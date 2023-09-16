"""
Tools for working with Open Ephys raw data files.
"""
from __future__ import annotations

import contextlib
import logging
import os
import pathlib
import shutil
from typing import Union

import crc32c
import rich.progress
import upath

logger = logging.getLogger(__name__)

PathLike = Union[str, bytes, os.PathLike, pathlib.Path, upath.UPath]


def from_pathlike(pathlike) -> upath.UPath:
    """
    >>> from_pathlike('s3://aind-data-bucket/experiment2_Record Node 102#probeA.png')
    S3Path('s3://aind-data-bucket/experiment2_Record Node 102#probeA.png')
    """
    if isinstance(pathlike, upath.UPath):
        return pathlike
    path: str = os.fsdecode(pathlike)
    # UPath will do rsplit('#')[0] on path
    if "#" in (p := pathlib.Path(path)).name:
        return upath.UPath(path).with_name(p.name)
    if "#" in p.parent.as_posix():
        raise ValueError(
            f"Path {p} contains '#' in a parent dir, which we don't have a fix for yet"
        )
    return upath.UPath(path)


def checksum(path: PathLike, show_progress_bar=True) -> str:
    path = from_pathlike(path)
    hasher = crc32c.crc32c

    def formatted(x):
        return f"{x:08X}"

    blocks_per_chunk = 4096
    multi_part_threshold_gb = 0.2
    if file_size(path) < multi_part_threshold_gb * 1024**3:
        return formatted(hasher(path.read_bytes()))
    hash = 0

    with open(path, "rb") as f, get_progress():
        progress: rich.progress.Progress = globals()["progress"]
        task = progress.add_task(
            f"Checksumming {path.name}",
            total=file_size(path),
            visible=show_progress_bar,
        )
        for chunk in iter(lambda: f.read(blocks_per_chunk), b""):
            progress.update(task, advance=blocks_per_chunk)
            hash = hasher(chunk, hash)
    progress.update(task, visible=False)
    return formatted(hash)


def get_progress() -> rich.progress.Progress | contextlib.nullcontext[None]:
    if "progress" not in globals():
        globals()["progress"] = rich.progress.Progress(
            rich.progress.TextColumn("{task.description}", justify="right"),
            rich.progress.BarColumn(),
            rich.progress.TimeRemainingColumn(),
            rich.progress.FileSizeColumn(),
            rich.progress.TotalFileSizeColumn(),
        )
        return globals()["progress"]
    else:
        return contextlib.nullcontext()


def get_copy_task(src) -> rich.progress.TaskID:
    get_progress()
    progress: rich.progress.Progress = globals()["progress"]
    if not progress.tasks:
        task = progress.add_task(
            description="[cyan]Getting file sizes",
            start=False,
        )
        progress.update(task, description="[cyan]Copying files", total=size(src))
        progress.start_task(task)
    return progress.tasks[0].id


def checksums_match(*paths: PathLike) -> bool:
    checksums = tuple(checksum(p) for p in paths)
    return all(c == checksums[0] for c in checksums)


def copy(src: PathLike, dest: PathLike, max_attempts: int = 2) -> None:
    """Copy `src` to `dest` with checksum validation.

    - copies recursively if `src` is a directory
    - if dest already exists, checksums are compared, copying is skipped if they match
    - attempts to copy up to 3 times if checksums don't match
    - replaces existing symlinks with actual files
    - creates parent dirs if needed
    """
    src, dest = from_pathlike(src), from_pathlike(dest)

    with get_progress():
        progress: rich.progress.Progress = globals()["progress"]
        task = get_copy_task(src)

        if dest.exists() and dest.is_symlink():
            dest.unlink()  # we'll replace symlink with src file

        if src.is_dir():  # copy files recursively
            for path in src.iterdir():
                copy(path, dest / path.name)
            return

        if (
            not dest.suffix
        ):  # dest is a folder, but might not exist yet so can't use `is_dir`
            dest = dest / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)

        if not dest.exists():
            shutil.copy2(src, dest)
            logger.debug(f"Copied {src} to {dest}")

        for _ in range(max_attempts):
            if checksums_match(src, dest):
                break
            shutil.copy2(src, dest)
        else:
            raise OSError(
                f"Failed to copy {src} to {dest} with checksum-validation after {max_attempts} attempts"
            )
        progress.update(task, advance=size(src))
        logger.debug(f"Copy of {src} at {dest} validated with checksum")


def move(src: PathLike, dest: PathLike, **rmtree_kwargs) -> None:
    """Copy `src` to `dest` with checksum validation, then delete `src`."""
    src, dest = from_pathlike(src), from_pathlike(dest)
    copy(src, dest)
    if src.is_dir():
        shutil.rmtree(src, **rmtree_kwargs)
    else:
        src.unlink()
    logger.debug(f"Deleted {src}")


def symlink(src: PathLike, dest: PathLike) -> None:
    """Create symlink at `dest` pointing to file at `src`.

    - creates symlinks recursively if `src` is a directory
    - creates parent dirs if needed (as folders, not symlinks)
    - skips if symlink already exists and points to `src`
    - replaces existing file or symlink pointing to a different location
    """
    src, dest = from_pathlike(src), from_pathlike(dest)
    if src.is_dir():
        for path in src.iterdir():
            symlink(src, dest / path.name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() and dest.resolve() == src.resolve():
        logger.debug(f"Symlink already exists to {src} from {dest}")
        return
    with contextlib.suppress(FileNotFoundError):
        dest.unlink()
    with contextlib.suppress(FileExistsError):
        dest.symlink_to(src)
    logger.debug(f"Created symlink to {src} from {dest}")


def size(path: PathLike) -> int:
    """Return the size of a file or directory in bytes"""
    path = from_pathlike(path)
    return dir_size(path) if path.is_dir() else file_size(path)


def size_gb(path: PathLike) -> float:
    """Return the size of a file or directory in GB"""
    return round(size(path) / 1024**3, 1)


def ctime(path: PathLike) -> float:
    path = from_pathlike(path)
    with contextlib.suppress(AttributeError):
        return path.stat().st_ctime
    with contextlib.suppress(AttributeError):
        return path.stat()["LastModified"].timestamp()
    raise RuntimeError(f"Could not get size of {path}")


def file_size(path: PathLike) -> int:
    path = from_pathlike(path)
    with contextlib.suppress(AttributeError):
        return path.stat().st_size
    with contextlib.suppress(AttributeError):
        return path.stat()["size"]
    raise RuntimeError(f"Could not get size of {path}")


def dir_size(path: PathLike) -> int:
    """Return the size of a directory in bytes"""
    path = from_pathlike(path)
    if not path.is_dir():
        raise ValueError(f"{path} is not a directory")
    dir_size = 0
    dir_size += sum(file_size(f) for f in path.rglob("*") if f.is_file())
    return dir_size


def dir_size_gb(path: PathLike) -> float:
    """Return the size of a directory in GB"""
    return round(dir_size(path) / 1024**3, 1)


def free_gb(path: PathLike) -> float:
    "Return free space at `path`, to .1 GB. Raises FileNotFoundError if `path` not accessible."
    path = from_pathlike(path)
    return round(shutil.disk_usage(path).free / 1024**3, 1)


if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )

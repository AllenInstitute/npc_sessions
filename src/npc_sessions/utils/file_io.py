"""
Tools for working with Open Ephys raw data files.
"""
from __future__ import annotations

import contextlib
import functools
import logging
import os
import pathlib
import shutil
from typing import Union

import crc32c
import hdmf_zarr
import pynwb
import rich.progress
import upath

logger = logging.getLogger(__name__)

PathLike = Union[str, bytes, os.PathLike, pathlib.Path, upath.UPath]


def from_pathlike(pathlike) -> upath.UPath:
    """
    >>> from_pathlike('s3://aind-data-bucket/experiment2_Record Node 102#probeA.png')
    S3Path('s3://aind-data-bucket/experiment2_Record Node 102#probeA.png')

    >>> from_pathlike('s3://codeocean-s3datasetsbucket-1u41qdg42ur9/4797cab2-9ea2-4747-8d15-5ba064837c1c/postprocessed/experiment1_Record Node 102#Neuropix-PXI-100.ProbeA-AP_recording1/template_metrics/params.json')
    S3Path('s3://codeocean-s3datasetsbucket-1u41qdg42ur9/4797cab2-9ea2-4747-8d15-5ba064837c1c/postprocessed/experiment1_Record Node 102#Neuropix-PXI-100.ProbeA-AP_recording1/template_metrics/params.json')
    """
    if isinstance(pathlike, upath.UPath):
        return pathlike
    path: str = os.fsdecode(pathlike)
    # UPath will do rsplit('#')[0] on path
    if "#" in (p := pathlib.Path(path)).name:
        return upath.UPath(path).with_name(p.name)
    if "#" in p.parent.as_posix():
        if p.parent.as_posix().count("#") > 1:
            raise ValueError(
                f"Path {p} contains multiple '#' in a parent dirs, which we don't have a fix for yet"
            )
        for parent in p.parents:
            if "#" in parent.name:
                new = upath.UPath(path).with_name(parent.name)
                for part in p.relative_to(parent).parts:
                    new = next(
                        new.glob(part)
                    )  # we can't create or join the problem-#, so we have to 'discover' it
                return new
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


def write_nwb(
    path: PathLike,
    nwb: pynwb.NWBFile,
) -> None:
    """Switches IO based on path suffix
    - use `.nwb` to save with `NWBHDF5IO`
    - use `.nwb.zarr` to save with `NWBZarrIO` (required for reading nwb.zarr files anyway)
    - can't currently write original hdf5 NWB to cloud directly
    """
    path = from_pathlike(path)
    if path.suffix == ".zarr":
        nwb_io_class = hdmf_zarr.NWBZarrIO
        if ".nwb" not in path.name:
            path = path.with_suffix(".nwb.zarr")
    else:
        assert path.suffix == ".nwb"
        nwb_io_class = pynwb.NWBHDF5IO
        if not path.as_uri().startswith("file"):
            raise ValueError(f"Must use a local path for {nwb_io_class!r}")

    with nwb_io_class(path=path.as_posix(), mode="w") as nwb_io:
        nwb_io.write(nwb)


_NOT_FOUND = object()


class cached_property(functools.cached_property):
    """Copy of stlib functools.cached_property minus faulty thread lock.

    Issue described here: https://github.com/python/cpython/issues/87634

    This version will make concurrent tasks across multiple instances faster, but
    each instance's cached properties will no longer be thread-safe - ie. don't
    dispatch the same instance to multiple threads without implementing your own lock.
    """

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        if self.attrname is None:
            raise TypeError(
                "Cannot use cached_property instance without calling __set_name__ on it."
            )
        try:
            cache = instance.__dict__
        except (
            AttributeError
        ):  # not all objects have __dict__ (e.g. class defines slots)
            msg = (
                f"No '__dict__' attribute on {type(instance).__name__!r} "
                f"instance to cache {self.attrname!r} property."
            )
            raise TypeError(msg) from None
        val = cache.get(self.attrname, _NOT_FOUND)
        ## these lines are removed from the original cached_property ------
        # with self.lock:
        #     # check if another thread filled cache while we awaited lock
        #     val = cache.get(self.attrname, _NOT_FOUND)
        # -----------------------------------------------------------------
        if val is _NOT_FOUND:
            val = self.func(instance)
            try:
                cache[self.attrname] = val
            except TypeError:
                msg = (
                    f"The '__dict__' attribute on {type(instance).__name__!r} instance "
                    f"does not support item assignment for caching {self.attrname!r} property."
                )
                raise TypeError(msg) from None
        return val


if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )

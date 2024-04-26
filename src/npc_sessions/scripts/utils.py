from __future__ import annotations

import argparse
import contextlib
import logging
import logging.config
import subprocess
import sys
from typing import Literal

import npc_lims
import psutil

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": True,
    }
)

logger = logging.getLogger()

MEMORY_PER_EPHYS_SESSION = 2 * 1024**3
"""Conservative estimate of a whole ephys session in memory."""

MEMORY_PER_TRAINING_SESSION = 0.2 * 1024**3

DEFAULT_CONTAINER_MEMORY = 7 * 1024**3
"""Default available memory in codeocean capsule or github action runner, in case
we can't query it.

https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners/about-github-hosted-runners#supported-runners-and-hardware-resources
"""


def get_max_workers(session_type: Literal["ephys", "training", "all"] = "ephys") -> int:
    if session_type == "training":
        memory_per_session = MEMORY_PER_TRAINING_SESSION
    else:
        memory_per_session = MEMORY_PER_EPHYS_SESSION

    return int(
        min(
            psutil.cpu_count(),
            get_available_memory() * 0.8 // memory_per_session,
        )
    )


def get_available_memory() -> int:
    """Assumes linux means containerized - get cgroups memory if possible."""
    if sys.platform == "linux":
        with contextlib.suppress(ValueError):
            return get_available_container_memory()
        return DEFAULT_CONTAINER_MEMORY
    return psutil.virtual_memory().available


def get_available_container_memory() -> int:
    """Available memory in the container, in bytes.

    `psutil.virtual_memory()` gives system memory, not container memory.
    In a github action or codeocean capsule, psutil will overestimate the available memory.
    """
    if sys.platform != "linux":
        raise NotImplementedError("Only implemented for linux")

    def _format(out: bytes) -> int:
        return int(out.decode().strip("\n"))

    def _run(cmd: list[str]) -> bytes:
        return subprocess.run(cmd, capture_output=True).stdout

    limit = _format(_run(["cat", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]))
    usage = _format(_run(["cat", "/sys/fs/cgroup/memory/memory.usage_in_bytes"]))
    return limit - usage


def setup(
    nwb: bool = False,
) -> dict[
    Literal["session_type", "skip_existing", "version", "parallel", "zarr_nwb"],
    str | bool | None,
]:
    args = parse_args()
    kwargs = vars(args)
    logging.basicConfig(
        level=kwargs.pop("log_level"),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    if not nwb:
        kwargs.pop("zarr_nwb")
    logger.info(f"Using parsed args {args}")
    return kwargs  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    """
    >>> parse_args()
    Namespace(session_type='ephys', skip_existing=True, version=None, parallel=False, log_level='INFO', max_workers=None, zarr_nwb=True)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session-type",
        choices=("training", "ephys", "all"),
        default="ephys",
        help="Type of sessions to process and write to cache (default 'ephys')",
    )
    parser.add_argument(
        "--overwrite",
        dest="skip_existing",
        action="store_false",
        help="Flag to skip existing files in cache",
    )
    parser.add_argument(
        "--version",
        type=str,
        metavar="v?.?.?",
        help="Subfolder that cached files are written to - default uses the current version of npc_sessions",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Flag to run in parallel",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        type=str,
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Level for the root logger (default INFO)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        metavar="N",
        help="If --parallel is used, override the number of workers in ProcessPool - default calculates appropriate number based on available memory",
    )
    parser.add_argument(
        "--hdf5-nwb",
        dest="zarr_nwb",
        action="store_false",
        help="Flag to store NWB files in hdf5 format instead of the default zarr",
    )

    return parser.parse_args()


def get_session_infos(
    session_type: Literal["training", "ephys", "all"]
) -> tuple[npc_lims.SessionInfo, ...]:
    if session_type not in ("training", "ephys", "all"):
        raise ValueError(f"{session_type=} must be one of 'training', 'ephys', 'all'")
    if session_type == "all":
        session_infos = npc_lims.get_session_info(issues=[])
    else:
        session_infos = tuple(
            s
            for s in npc_lims.get_session_info(issues=[])
            if s.is_ephys == is_ephys(session_type=session_type)
        )
    return session_infos


def is_ephys(session_type: Literal["training", "ephys", "all"]) -> bool | None:
    """
    >>> is_ephys("training")
    False
    >>> is_ephys("ephys")
    True
    >>> is_ephys("all")


    """
    if session_type not in ("training", "ephys", "all"):
        raise ValueError(f"{session_type=} must be one of 'training', 'ephys', 'all'")
    if session_type == "all":
        return None
    return session_type == "ephys"


if __name__ == "__main__":
    import doctest

    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )

from __future__ import annotations

import argparse
import logging
import logging.config
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

MEMORY_PER_SESSION = 4 * 1024**3
"""Conservative estimate of a whole ephys session in memory. Will be much less for
training sessions."""


def get_max_workers() -> int:
    return min(
        psutil.cpu_count(), psutil.virtual_memory().available * 0.7 / MEMORY_PER_SESSION
    )


def setup() -> (
    dict[
        Literal["session_type", "skip_existing", "version", "parallel"],
        str | bool | None,
    ]
):
    args = parse_args()
    kwargs = vars(args)
    logging.basicConfig(
        level=kwargs.pop("log_level"),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logger.info(f"Using parsed args {args}")
    return kwargs  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    """
    >>> parse_args()
    Namespace(session_type='ephys', skip_existing=True, version=None, parallel=False, log_level='INFO')
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
        default=None,
        type=str,
        help="Subfolder that cached files are written to - if None, uses the current version of npc_sessions",
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
    return parser.parse_args()


def get_session_infos(
    session_type: Literal["training", "ephys", "all"]
) -> tuple[npc_lims.SessionInfo, ...]:
    if session_type not in ("training", "ephys", "all"):
        raise ValueError(f"{session_type=} must be one of 'training', 'ephys', 'all'")
    if session_type == "all":
        session_infos = npc_lims.get_session_info()
    else:
        session_infos = tuple(
            s
            for s in npc_lims.get_session_info()
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

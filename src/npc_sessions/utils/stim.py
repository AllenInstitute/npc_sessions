from __future__ import annotations

import json
import logging
from collections.abc import Container, Iterable
from typing import Literal, TypeVar

import numpy as np
import numpy.typing as npt
import upath


import npc_sessions.utils as utils

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    import doctest

    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )

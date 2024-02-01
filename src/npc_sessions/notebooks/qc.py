from __future__ import annotations

import importlib.resources
import logging
import os

import npc_session
import upath

import npc_sessions.utils as utils

logger = logging.getLogger(__name__)

MODULE_ROOT = upath.UPath(__file__).parent
PACKAGE_ROOT = utils.from_pathlike(importlib.resources.files("npc_sessions"))  # type: ignore[arg-type]


def write_qc_notebook(
    session_path_or_id: str,
    save_path: utils.PathLike = upath.UPath.cwd(),
    **session_kwargs,
) -> upath.UPath:
    """Find the given session's appropriate QC notebook in the site-packages
    folder, run it and save it at the given path.

    - currently assumes dynamic routing sessions

    >>> output_path = write_qc_notebook("664566_20230403", is_ephys=False)
    """
    QC_NOTEBOOK = next(MODULE_ROOT.rglob("dynamic*routing*qc*.ipynb"))

    # pass config to run in notebook via env vars
    env = {}
    env["NPC_SESSION_PATH_OR_ID"] = session_path_or_id
    for k, v in session_kwargs.items():
        env[f"NPC_SESSION_{k}"] = str(v)
        # notebook should eval or json.loads env vars to pass to session init
        # keys are auto-capitalized (on windows, at least), so also apply .lower()

    logger.info(
        f"running {QC_NOTEBOOK.name} for {npc_session.SessionRecord(session_path_or_id)}"
    )
    return utils.run_and_save_notebook(
        notebook_path=QC_NOTEBOOK,
        save_path=save_path,
        env=dict(os.environ) | env,  # merge with current env to reuse credentials etc.
    )


if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )

"""
Helper functions for getting paths from spikeinterface output (developed for use
with the aind kilosort 2.5 "pipeline" spike-sorting capsule).
"""

from __future__ import annotations

import dataclasses
import functools
import io
import json
from typing import Union

import npc_lims
import npc_session
import numpy as np
import numpy.typing as npt
import pandas as pd
import upath
from typing_extensions import TypeAlias

import npc_sessions.utils as utils

SpikeInterfaceData: TypeAlias = Union[
    str, npc_session.SessionRecord, utils.PathLike, "SpikeInterfaceKS25Data"
]


def get_spikeinterface_data(
    session_or_root_path: SpikeInterfaceData,
) -> SpikeInterfaceKS25Data:
    """Return a SpikeInterfaceKS25Data object for a session.

    >>> paths = get_spikeinterface_data('668759_20230711')
    >>> paths.root == get_spikeinterface_data('s3://codeocean-s3datasetsbucket-1u41qdg42ur9/4797cab2-9ea2-4747-8d15-5ba064837c1c').root
    True
    """
    if isinstance(session_or_root_path, SpikeInterfaceKS25Data):
        return session_or_root_path
    try:
        session = npc_session.SessionRecord(str(session_or_root_path))
        root = None
    except ValueError:
        session = None
        root = utils.from_pathlike(session_or_root_path)
    return SpikeInterfaceKS25Data(session=session, root=root)


@dataclasses.dataclass(unsafe_hash=True, eq=True)
class SpikeInterfaceKS25Data:
    """The root directory of the result data asset produced by the 'pipeline'
    KS2.5 sorting capsule contains `processing.json`, `postprocessed`,
    `spikesorted`, etc. This class just simplifies access to the data in those
    files and dirs.

    Provide a session ID or a root path:
    >>> paths = SpikeInterfaceKS25Data('668759_20230711')
    >>> paths.root
    S3Path('s3://codeocean-s3datasetsbucket-1u41qdg42ur9/4797cab2-9ea2-4747-8d15-5ba064837c1c')

    >>> paths.template_metrics_dict('probeA')
    {'metric_names': ['peak_to_valley', 'peak_trough_ratio', 'half_width', 'repolarization_slope', 'recovery_slope'], 'sparsity': None, 'peak_sign': 'neg', 'upsampling_factor': 10, 'window_slope_ms': 0.7}

    >>> paths.quality_metrics_df('probeA').columns
    Index(['num_spikes', 'firing_rate', 'presence_ratio', 'snr',
           'isi_violations_ratio', 'isi_violations_count', 'rp_contamination',
           'rp_violations', 'sliding_rp_violation', 'amplitude_cutoff',
           'drift_ptp', 'drift_std', 'drift_mad', 'isolation_distance', 'l_ratio',
           'd_prime'],
          dtype='object')
    """

    session: str | npc_session.SessionRecord | None = None
    root: upath.UPath | None = None

    def __post_init__(self) -> None:
        if self.root is None and self.session is None:
            raise ValueError("Must provide either session or root")
        if self.root is None:
            self.root = npc_lims.get_sorted_data_paths_from_s3(self.session)[0].parent

    @staticmethod
    def format_path(*path_components: utils.PathLike) -> upath.UPath:
        """SpikeInterface makes paths with '#' in them, which is not allowed in s3
        paths in general - run paths through this function to fix them."""
        return utils.from_pathlike("/".join(str(path) for path in path_components))

    @staticmethod
    def read_json(path: upath.UPath) -> dict:
        return json.loads(path.read_text())

    @staticmethod
    def read_csv(path: upath.UPath) -> pd.DataFrame:
        return pd.read_csv(path, index_col=0)

    def get_json(self, filename: str) -> dict:
        assert self.root is not None
        return self.read_json(self.format_path(self.root, filename))

    def get_path(self, dirname: str, probe: str | None) -> upath.UPath:
        """Return a path to a single dir or file: either `self.root/dirname` or, if `probe` is specified,
        the probe-specific sub-path within `self.root/dirname`."""
        assert self.root is not None
        if probe is None:
            return self.format_path(self.root, dirname)
        else:
            return next(
                path
                for path in self.format_path(self.root, dirname).iterdir()
                if npc_session.ProbeRecord(probe)
                == npc_session.ProbeRecord(path.as_posix())
            )

    # json data
    processing_json = functools.partialmethod(get_json, "processing.json")
    subject_json = functools.partialmethod(get_json, "subject.json")
    data_description_json = functools.partialmethod(get_json, "data_description.json")
    procedures_json = functools.partialmethod(get_json, "procedures.json")
    visualization_output_json = functools.partialmethod(
        get_json, "visualization_output.json"
    )

    # dirs
    drift_maps = functools.partialmethod(get_path, "drift_maps")
    output = functools.partialmethod(get_path, "output")
    postprocessed = functools.partialmethod(get_path, "postprocessed")
    sorting_precurated = functools.partialmethod(get_path, "sorting_precurated")
    spikesorted = functools.partialmethod(get_path, "spikesorted")

    @functools.cache
    def quality_metrics_dict(self, probe: str) -> dict:
        return self.read_json(
            self.format_path(
                self.postprocessed(probe), "quality_metrics", "params.json"
            )
        )

    @functools.cache
    def postprocessed_params_dict(self, probe: str) -> dict:
        return self.read_json(
            self.format_path(self.postprocessed(probe), "params.json")
        )

    @functools.cache
    def quality_metrics_df(self, probe: str) -> pd.DataFrame:
        return self.read_csv(
            self.format_path(
                self.postprocessed(probe), "quality_metrics", "metrics.csv"
            )
        )

    @functools.cache
    def template_metrics_dict(self, probe: str) -> dict:
        return self.read_json(
            self.format_path(
                self.postprocessed(probe), "template_metrics", "params.json"
            )
        )

    @functools.cache
    def template_metrics_df(self, probe: str) -> pd.DataFrame:
        return self.read_csv(
            self.format_path(
                self.postprocessed(probe), "template_metrics", "metrics.csv"
            )
        )

    @functools.cache
    def templates_average(self, probe: str) -> npt.NDArray[np.floating]:
        return np.load(
            io.BytesIO(
                self.format_path(
                    self.postprocessed(probe), "templates_average.npy"
                ).read_bytes()
            )
        )

    @functools.cache
    def templates_std(self, probe: str) -> npt.NDArray[np.floating]:
        return np.load(
            io.BytesIO(
                self.format_path(
                    self.postprocessed(probe), "templates_std.npy"
                ).read_bytes()
            )
        )

    @functools.cache
    def sorting_cached(self, probe: str) -> dict[str, npt.NDArray]:
        return np.load(
            io.BytesIO(
                self.format_path(
                    self.spikesorted(probe), "sorting_cached.npz"
                ).read_bytes()
            ),
            allow_pickle=True,
        )

    @functools.cache
    def default_qc(self, probe: str) -> npt.NDArray[np.floating]:
        return np.load(
            io.BytesIO(
                self.format_path(
                    self.sorting_precurated(probe), "properties", "default_qc.npy"
                ).read_bytes()
            )
        )

    @functools.cache
    def unit_locations(self, probe: str) -> npt.NDArray[np.floating]:
        return np.load(
            io.BytesIO(
                self.format_path(
                    self.postprocessed(probe), "unit_locations", "unit_locations.npy"
                ).read_bytes()
            )
        )

    @functools.cache
    def sorting_json(self, probe: str) -> dict:
        return self.read_json(
            self.format_path(self.postprocessed(probe), "sorting.json")
        )

    @functools.cache
    def electrode_locations_xy(self, probe: str) -> npt.NDArray[np.floating]:
        return np.array(
            self.sorting_json(probe)["annotations"]["__sorting_info__"]["recording"][
                "properties"
            ]["location"]
        )


if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )

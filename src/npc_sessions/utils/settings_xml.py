"""
>>> path = upath.UPath('s3://aind-ephys-data/ecephys_668759_2023-07-11_13-07-32/ecephys_clipped/Record Node 102/settings.xml')
>>> et = ET.parse(io.BytesIO(path.read_bytes()))
>>> _hostname(et)
'W10DT05516'
>>> _date_time(et)
(datetime.date(2023, 7, 11), datetime.time(13, 7, 53))
>>> _open_ephys_version(et)
'0.6.4'
>>> _settings_xml_md5(path)
'5c1b33293cb7c5f72df56fbdd4b72fc3'
>>> _probe_serial_numbers(et)
(18194810652, 18005123131, 18005102491, 19192719021, 18005118602, 19192719061)
>>> _probe_letters(et)
('A', 'B', 'C', 'D', 'E', 'F')
>>> _probe_types(et)
('Neuropixels 1.0', 'Neuropixels 1.0', 'Neuropixels 1.0', 'Neuropixels 1.0', 'Neuropixels 1.0', 'Neuropixels 1.0')
>>> isinstance(get_settings_xml_data(path), SettingsXmlInfo)
True
>>> _probe_serial_number_to_channel_pos_xy(et)[18194810652]['CH0']
(27, 0)
"""

from __future__ import annotations

import dataclasses
import datetime
import doctest
import functools
import hashlib
import io
import xml.etree.ElementTree as ET
from typing import Literal

import upath

import npc_sessions.utils as utils


@dataclasses.dataclass
class SettingsXmlInfo:
    """Info from a settings.xml file from an Open Ephys recording."""

    path: upath.UPath
    probe_serial_numbers: tuple[int, ...]
    probe_types: tuple[str, ...]
    probe_letters: tuple[Literal["A", "B", "C", "D", "E", "F"], ...]
    hostname: str
    date: datetime.date
    start_time: datetime.time
    open_ephys_version: str
    channel_pos_xy: tuple[dict[str, tuple[int, int]], ...]

    def __eq__(self, other) -> bool:
        if not isinstance(other, SettingsXmlInfo):
            return NotImplemented
        # files from multiple nodes can be created at slightly different times, so their `date`
        # fields may differ. Everything else should be identical.
        return all(
            getattr(self, field.name) == getattr(other, field.name)
            for field in dataclasses.fields(self)
            if field.name not in ("path", "start_time")
        )


def get_settings_xml_etree(path: utils.PathLike) -> ET.ElementTree:
    """Info from a settings.xml file from an Open Ephys recording."""
    return ET.parse(io.BytesIO(utils.from_pathlike(path).read_bytes()))


def get_settings_xml_data(path: utils.PathLike | SettingsXmlInfo) -> SettingsXmlInfo:
    if isinstance(path, SettingsXmlInfo):
        return path
    et = get_settings_xml_etree(path)
    return SettingsXmlInfo(
        path=utils.from_pathlike(path),
        probe_serial_numbers=_probe_serial_numbers(et),
        probe_types=_probe_types(et),
        probe_letters=_probe_letters(et),
        hostname=_hostname(et),
        date=_date_time(et)[0],
        start_time=_date_time(et)[1],
        open_ephys_version=_open_ephys_version(et),
        channel_pos_xy=tuple(_probe_serial_number_to_channel_pos_xy(et).values()),
    )


def _get_tag_text(et: ET.ElementTree, tag: str) -> str | None:
    result = [
        element.text for element in et.getroot().iter() if element.tag == tag.upper()
    ]
    if not (result and any(result)):
        result = [element.attrib.get(tag.lower()) for element in et.getroot().iter()]
    return str(result[0]) if (result and any(result)) else None


def _get_tag_attrib(et: ET.ElementTree, tag: str, attrib: str) -> str | None:
    result = [
        element.attrib.get(attrib)
        for element in et.getroot().iter()
        if element.tag == tag.upper()
    ]
    return str(result[0]) if (result and any(result)) else None


def _hostname(et: ET.ElementTree) -> str:
    result = (
        # older, pre-0.6.x:
        _get_tag_text(et, "machine")
        # newer, 0.6.x:
        or _get_tag_attrib(et, "MACHINE", "name")
    )
    if not result:
        raise LookupError(f"No hostname: {result!r}")
    return result


@functools.cache
def _date_time(et: ET.ElementTree) -> tuple[datetime.date, datetime.time]:
    """Date and recording start time."""
    result = _get_tag_text(et, "date")
    if not result:
        raise LookupError(f"No datetime found: {result!r}")
    dt = datetime.datetime.strptime(result, "%d %b %Y %H:%M:%S")
    return dt.date(), dt.time()


@functools.cache
def _probe_attrib_dicts(et: ET.ElementTree) -> tuple[dict[str, str], ...]:
    return tuple(
        probe_dict.attrib
        for probe_dict in et.getroot().iter()
        if "probe_serial_number" in probe_dict.attrib
    )


def _probe_attrib(et: ET.ElementTree, attrib: str) -> tuple[str, ...]:
    return tuple(probe[attrib] for probe in _probe_attrib_dicts(et))


def _probe_serial_numbers(et: ET.ElementTree) -> tuple[int, ...]:
    return tuple(int(_) for _ in _probe_attrib(et, "probe_serial_number"))


def _probe_types(et: ET.ElementTree) -> tuple[str, ...]:
    try:
        return _probe_attrib(et, "probe_name")
    except KeyError:
        return tuple("unknown" for _ in _probe_attrib_dicts(et))


def _probe_serial_number_to_channel_pos_xy(
    et: ET.ElementTree,
) -> dict[int, dict[str, tuple[int, int]]]:
    x_pos_iter = tuple(a for a in et.getroot().iter() if a.tag == "ELECTRODE_XPOS")
    y_pos_iter = tuple(a for a in et.getroot().iter() if a.tag == "ELECTRODE_YPOS")

    return dict(
        zip(
            _probe_serial_numbers(et),
            (
                {k: (int(x.attrib[k]), int(y.attrib[k])) for k in x.attrib.keys()}
                for x, y in zip(x_pos_iter, y_pos_iter)
            ),
        )
    )


def _probe_idx(et: ET.ElementTree) -> tuple[int, ...]:
    """Try to reconstruct index from probe slot and port.

    assumptions:
        - 2 slots
        - 3 ports in use per slot
        - probes ABCDEF connected in sequence
    """
    slots, ports = _probe_attrib(et, "slot"), _probe_attrib(et, "port")
    result = tuple(
        (int(s) - int(min(slots))) * len(set(ports)) + int(p) - 1
        for s, p in zip(slots, ports)
    )
    if not all(idx in range(6) for idx in result):
        raise ValueError(f"probe_idx: {result!r}, slots: {slots}, ports: {ports}")
    return result


def _probe_letters(
    et: ET.ElementTree,
) -> tuple[Literal["A", "B", "C", "D", "E", "F"], ...]:
    return tuple("ABCDEF"[idx] for idx in _probe_idx(et))  # type: ignore [misc]


def _open_ephys_version(et: ET.ElementTree) -> str:
    result = _get_tag_text(et, "version")
    if not result:
        raise LookupError(f"No version found: {result!r}")
    return result


def _settings_xml_md5(path: str | upath.UPath) -> str:
    return hashlib.md5(upath.UPath(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )

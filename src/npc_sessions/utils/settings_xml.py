"""
>>> path = upath.UPath('s3://aind-ephys-data/ecephys_668759_2023-07-11_13-07-32/ecephys_clipped/Record Node 102/settings.xml')
>>> et = ET.parse(io.BytesIO(path.read_bytes()))
>>> hostname(et)
'W10DT05516'
>>> date_time(et)
(datetime.date(2023, 7, 11), datetime.time(13, 7, 53))
>>> open_ephys_version(et)
'0.6.4'
>>> settings_xml_md5(path)
'5c1b33293cb7c5f72df56fbdd4b72fc3'
>>> probe_serial_numbers(et)
(18194810652, 18005123131, 18005102491, 19192719021, 18005118602, 19192719061)
>>> probe_letters(et)
('A', 'B', 'C', 'D', 'E', 'F')
>>> probe_types(et)
('Neuropixels 1.0', 'Neuropixels 1.0', 'Neuropixels 1.0', 'Neuropixels 1.0', 'Neuropixels 1.0', 'Neuropixels 1.0')
>>> isinstance(settings_xml_info_from_path(path), SettingsXmlInfo)
True
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


@dataclasses.dataclass
class SettingsXmlInfo:
    """Info from a settings.xml file from an Open Ephys recording."""

    path: upath.UPath
    probe_serial_numbers: tuple[int, ...]
    probe_types: tuple[str, ...]
    probe_letters: Literal["A", "B", "C", "D", "E", "F"]
    hostname: str
    date: datetime.date
    start_time: datetime.time
    open_ephys_version: str


def settings_xml_info_from_path(path: str | upath.UPath) -> SettingsXmlInfo:
    """Info from a settings.xml file from an Open Ephys recording."""
    path = upath.UPath(path)
    et = ET.parse(io.BytesIO(path.read_bytes()))
    return SettingsXmlInfo(
        path=path,
        probe_serial_numbers=probe_serial_numbers(et),
        probe_types=probe_types(et),
        probe_letters=probe_letters(et),
        hostname=hostname(et),
        date=date_time(et)[0],
        start_time=date_time(et)[1],
        open_ephys_version=open_ephys_version(et),
    )


def get_tag_text(et: ET.ElementTree, tag: str) -> str | None:
    result = [
        element.text for element in et.getroot().iter() if element.tag == tag.upper()
    ]
    if not (result and any(result)):
        result = [element.attrib.get(tag.lower()) for element in et.getroot().iter()]
    return str(result[0]) if (result and any(result)) else None


def get_tag_attrib(et: ET.ElementTree, tag: str, attrib: str) -> str | None:
    result = [
        element.attrib.get(attrib)
        for element in et.getroot().iter()
        if element.tag == tag.upper()
    ]
    return str(result[0]) if (result and any(result)) else None


def hostname(et: ET.ElementTree) -> str:
    result = (
        # older, pre-0.6.x:
        get_tag_text(et, "machine")
        # newer, 0.6.x:
        or get_tag_attrib(et, "MACHINE", "name")
    )
    if not result:
        raise LookupError(f"No hostname: {result!r}")
    return result


@functools.cache
def date_time(et: ET.ElementTree) -> tuple[datetime.date, datetime.time]:
    """Date and recording start time."""
    result = get_tag_text(et, "date")
    if not result:
        raise LookupError(f"No datetime found: {result!r}")
    dt = datetime.datetime.strptime(result, "%d %b %Y %H:%M:%S")
    return dt.date(), dt.time()


@functools.cache
def probe_attrib_dicts(et: ET.ElementTree) -> tuple[dict[str, str], ...]:
    return tuple(
        probe_dict.attrib
        for probe_dict in et.getroot().iter()
        if "probe_serial_number" in probe_dict.attrib
    )


def probe_attrib(et: ET.ElementTree, attrib: str) -> tuple[str, ...]:
    return tuple(probe[attrib] for probe in probe_attrib_dicts(et))


def probe_serial_numbers(et: ET.ElementTree) -> tuple[int, ...]:
    return tuple(int(_) for _ in probe_attrib(et, "probe_serial_number"))


def probe_types(et: ET.ElementTree) -> tuple[str, ...]:
    try:
        return probe_attrib(et, "probe_name")
    except KeyError:
        return tuple("unknown" for _ in probe_attrib_dicts(et))


def probe_idx(et: ET.ElementTree) -> tuple[int, ...]:
    """Try to reconstruct index from probe slot and port.

    Normally 2 slots: each with 3 ports in use.
    """
    slots, ports = probe_attrib(et, "slot"), probe_attrib(et, "port")
    result = tuple(
        (int(s) - int(min(slots))) * len(set(ports)) + int(p) - 1
        for s, p in zip(slots, ports)
    )
    if not all(idx in range(6) for idx in result):
        raise ValueError(f"probe_idx: {result!r}, slots: {slots}, ports: {ports}")
    return result


def probe_letters(et: ET.ElementTree) -> Literal["A", "B", "C", "D", "E", "F"]:
    return tuple("ABCDEF"[idx] for idx in probe_idx(et))  # type: ignore


def open_ephys_version(et: ET.ElementTree) -> str:
    result = get_tag_text(et, "version")
    if not result:
        raise LookupError(f"No version found: {result!r}")
    return result


def settings_xml_md5(path: str | upath.UPath) -> str:
    return hashlib.md5(upath.UPath(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )

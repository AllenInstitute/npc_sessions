from __future__ import annotations

from typing import Any, Callable

bad_tag = "[bold red]"
good_tag = "[bold green]"
cautious_tag = "[bold yellow]"


def bad_string(base: str) -> str:
    return bad_tag + base + bad_tag


def good_string(base: str) -> str:
    return good_tag + base + good_tag


def cautious_string(base: str) -> str:
    return cautious_tag + base + cautious_tag


def determine_string_valence(
    value: Any, good_criterion: bool, bad_criterion: bool
) -> Callable[[str], str]:
    if good_criterion:
        return good_string

    elif bad_criterion:
        return bad_string

    else:
        return cautious_string


def add_valence_to_string(
    basestring: str, value: float | int, good_criterion: bool, bad_criterion: bool
) -> str:
    valence_func = determine_string_valence(value, good_criterion, bad_criterion)

    return valence_func(basestring)

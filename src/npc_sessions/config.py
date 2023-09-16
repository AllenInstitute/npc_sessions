from __future__ import annotations

session_kwargs = {
    "670248_2023-08-02": {"suppress_errors": True},
    "628801_2022-09-20": {"is_video": False},
}
session_issues: dict[str, str | None] = dict.fromkeys(
    (
        "660023_2023-08-08",
        "666986_2023-08-14",
        "644867_2023-02-21",
        "628801_2022-09-20",
    ),
    None,
)
"""Sessions with known issues, optionally mapped to github issue urls"""

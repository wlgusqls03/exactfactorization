"""Shared output-directory policy for dated calculation results."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re


_DATE_DIRECTORY = re.compile(r"^\d{8}$")


def dated_results_dir(outdir: str | Path, date: str | None = None) -> Path:
    """Place an output directory below ``results/YYYYMMDD``.

    Relative paths already beginning with ``results`` keep the part below
    that directory. Other relative paths are treated as names below
    ``results``. Absolute paths remain absolute; the date is inserted directly
    after a ``results`` component when present, or immediately before the last
    path component otherwise.

    A path whose date position already contains an eight-digit directory is
    returned unchanged, so callers can safely apply this helper more than
    once. ``date`` exists primarily to make the path policy deterministic in
    tests; normal runs use the server's local calendar date.
    """
    run_date = date or datetime.now().astimezone().strftime("%Y%m%d")
    if not _DATE_DIRECTORY.fullmatch(run_date):
        raise ValueError(f"date must have YYYYMMDD form, got {run_date!r}")

    path = Path(outdir)
    parts = path.parts

    if path.is_absolute():
        try:
            results_index = parts.index("results")
        except ValueError:
            if any(_DATE_DIRECTORY.fullmatch(part) for part in parts):
                return path
            parent, name = path.parent, path.name
            return parent / run_date / name

        date_index = results_index + 1
        if date_index < len(parts) and _DATE_DIRECTORY.fullmatch(parts[date_index]):
            return path
        return Path(*parts[:date_index], run_date, *parts[date_index:])

    relative_parts = parts[1:] if parts and parts[0] == "results" else parts
    if relative_parts and _DATE_DIRECTORY.fullmatch(relative_parts[0]):
        return Path("results", *relative_parts)
    return Path("results", run_date, *relative_parts)

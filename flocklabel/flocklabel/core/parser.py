"""Read the recording date and time from a video's file name.

The default, ``auto``, finds a date-time anywhere in the name, with or without
separators, and an optional end time after it, e.g.

* ``2025-10-01_16-59-45_to_16-59-47__ch00__SWF_1000381605376.mp4``
* ``ch01_20251014073000_20251014080000.mp4``
* ``NVR 2025-10-14 07.30.00.mp4``

Channels are ignored. A custom regex needs a named group ``start`` and may
have ``end`` (a full date-time, or a time of day on the same date).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

_DATE = r"\d{4}[-_.]?\d{2}[-_.]?\d{2}"
_TIME = r"\d{2}[-_.:]?\d{2}[-_.:]?\d{2}"

PRESETS: dict[str, str] = {
    "auto": rf"(?P<start>{_DATE}[-_. T]?{_TIME})(?:[^0-9]{{1,6}}(?P<end>(?:{_DATE}[-_. T]?)?{_TIME})(?!\d))?",
}
PRESET_EXAMPLES = {"auto": "2025-10-01_16-59-45_to_16-59-47__ch00__SWF_1000381605376.mp4"}


@dataclass(frozen=True)
class ParsedName:
    start: datetime
    end: Optional[datetime]


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def to_datetime(text: str) -> datetime:
    """Parse a 14-digit date-time written with any separators."""
    d = _digits(text)
    if len(d) != 14:
        raise ValueError(f"Not a date-time: {text!r}")
    return datetime.strptime(d, "%Y%m%d%H%M%S")


def parse_name(name: str, pattern: str = "auto") -> Optional[ParsedName]:
    """Start (and end, if present) of the recording named ``name``, or ``None``."""
    try:
        match = re.compile(PRESETS.get(pattern, pattern), re.IGNORECASE).search(name.replace("\\", "/"))
    except re.error:
        raise
    if not match or not match.groupdict().get("start"):
        return None
    try:
        start = to_datetime(match.group("start"))
    except ValueError:
        return None
    end = None
    raw_end = match.groupdict().get("end")
    if raw_end:
        d = _digits(raw_end)
        try:
            if len(d) == 14:
                end = datetime.strptime(d, "%Y%m%d%H%M%S")
            elif len(d) == 6:
                end = datetime.combine(start.date(), datetime.strptime(d, "%H%M%S").time())
                if end < start:                       # crosses midnight
                    end += timedelta(days=1)
        except ValueError:
            end = None
    if end is not None and end <= start:
        end = None
    return ParsedName(start=start, end=end)

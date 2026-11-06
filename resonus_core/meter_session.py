"""Reader for resonus-meter measurement bundles (``splmeter-session`` format 1.x).

resonus-meter (Rust) saves a measurement as a folder: ``measurement.md`` (Markdown with
YAML front matter and a fenced ``yaml {#summary}`` block) plus CSV files, the format
``splmeter-session`` version 1.2 (resonus-meter ``docs/FILE_FORMAT.md``). This module
loads such a bundle into numpy arrays and the common data model: the band levels as
:class:`~resonus_core.data.BandResults`, provenance as
:class:`~resonus_core.data.Metadata` (decision D-S1; the bundle stays the meter's
canonical format, so the Rust app needs no HDF5 library).

Reading rules (from the format document):

- ``format`` must be ``"splmeter-session"``; ``version`` is a ``"major.minor"`` string,
  or the integer 1 (format 1.0). A higher major version than :data:`SUPPORTED_MAJOR`
  is rejected; unknown front-matter keys, CSV columns and marker kinds are tolerated.
- File paths come from the front matter's ``files`` entries, relative to the bundle.
- A blank CSV cell is a missing value: NaN in float arrays.
- Times are UTC ISO 8601 with a ``Z`` suffix.

The levels are what the meter computed (IEC 61672-1:2013 weightings and time
weightings, IEC 61260-1:2014 bands: methods only, see resonus-meter
``docs/COMPLIANCE.md``); this module does not recompute them.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import ArrayLike, NDArray

from .bands import BandSet, band_index, nominal_frequency
from .data import BandResults, LevelHistory, Metadata

FORMAT = "splmeter-session"
"""Value of the front matter's ``format`` key."""
SUPPORTED_MAJOR = 1
"""Newest major format version this reader accepts."""
LOG_LEVELS = ("LAF", "LCF", "LZF")
"""Level columns of ``log.csv`` (Fast A/C/Z at each interval end)."""
BAND_PREFIX = "LZeq_"
"""Prefix of the band columns of ``rta_log.csv`` and ``rta_segments.csv``."""
_SUMMARY = re.compile(r"```yaml \{#summary\}\n(.*?)\n```", re.S)
_METADATA_KEYS = ("device", "calibration", "log_interval_s", "reference_pressure_uPa",
                  "end_utc", "duration_s", "tags", "author")  # fmt: skip


@dataclass(frozen=True)
class Marker:
    """One row of ``markers.csv``.

    Parameters
    ----------
    id
        Marker id (from 1).
    t_s
        Session time in s.
    utc
        Wall-clock time, timezone-aware UTC.
    kind
        ``start``, ``stop``, ``user``, ``reset``, ``interval_change``, ``pause``,
        ``resume``, or a kind of a newer format version (kept as written).
    label
        Label text.
    note
        Note text, ``""`` if none.
    """

    id: int
    t_s: float
    utc: datetime
    kind: str
    label: str
    note: str


@dataclass(frozen=True)
class MeterSession:
    """A loaded measurement bundle.

    Parameters
    ----------
    path
        The bundle folder.
    version
        Format version as ``"major.minor"`` (``"1.0"`` for the integer 1).
    front_matter
        The YAML front matter as parsed (``name``, ``start_utc``, ``device``,
        ``calibration``, ``files``, ``rta``, ...).
    body
        The Markdown after the front matter (title, description, summary block).
    summary
        The parsed ``yaml {#summary}`` block; ``{}`` if the bundle has none.
    log_t_s
        ``log.csv`` session times in s, shape ``(n,)``.
    log_utc
        ``log.csv`` wall-clock times (UTC), one per row.
    log_marker
        ``log.csv`` ``marker_id``, 0 where the cell is blank (ids start at 1), int64.
    log_clip
        ``log.csv`` ``clip`` as bool.
    log_levels
        The :data:`LOG_LEVELS` columns of ``log.csv`` in dB, each shape ``(n,)``.
    markers
        The rows of ``markers.csv`` in file order.
    segments
        Every column of ``segments.csv`` as float64 (blank = NaN), keyed by the header
        name, e.g. ``"duration_s"``, ``"LAeq"``, ``"clip"``.
    bands
        The recorded bands (front matter ``rta``), ``None`` without band levels.
    band_log
        ``rta_log.csv`` band LZeq in dB, shape ``(n, len(bands))``; ``None`` without
        band levels.
    band_segments
        ``rta_segments.csv`` band LZeq in dB, shape ``(segments, len(bands))``; ``None``
        without band levels.
    """

    path: Path
    version: str
    front_matter: dict[str, Any]
    body: str
    summary: dict[str, Any]
    log_t_s: NDArray[np.float64]
    log_utc: tuple[datetime, ...]
    log_marker: NDArray[np.int64]
    log_clip: NDArray[np.bool_]
    log_levels: dict[str, NDArray[np.float64]]
    markers: tuple[Marker, ...]
    segments: dict[str, NDArray[np.float64]]
    bands: BandSet | None = None
    band_log: NDArray[np.float64] | None = None
    band_segments: NDArray[np.float64] | None = None

    @property
    def name(self) -> str:
        """The measurement name (front matter ``name``)."""
        return str(self.front_matter["name"])

    @property
    def start_utc(self) -> datetime:
        """The start time (front matter ``start_utc``), timezone-aware UTC."""
        return parse_utc(str(self.front_matter["start_utc"]))

    def metadata(self) -> Metadata:
        """Return the provenance in the common data model.

        - ``software``, ``software_version``: front matter ``software`` split at its
          last space if the last word starts with a digit (the meter writes
          ``"<name> <version>"``, e.g. ``"SPL Meter 0.1.0"`` -> ``("SPL Meter",
          "0.1.0")``); otherwise the whole text and ``""``.
        - ``created``: front matter ``start_utc`` as written.
        - ``description``: front matter ``name``.
        - ``device``: front matter ``device.name``, ``""`` if absent.
        - ``standards``: ``()``: the bundle does not state any.
        - ``extra``: ``{"format": FORMAT, "version": self.version, "bundle":
          self.path.name}`` plus the front-matter keys ``device``, ``calibration``,
          ``log_interval_s``, ``reference_pressure_uPa``, ``end_utc``, ``duration_s``,
          ``tags``, ``author`` that are present, unchanged.

        Returns
        -------
        Metadata
        """
        fm = self.front_matter
        software = str(fm.get("software", ""))
        name, _, version = software.rpartition(" ")
        if not name or not version[:1].isdigit():
            name, version = software, ""
        extra: dict[str, Any] = {"format": FORMAT, "version": self.version}
        extra["bundle"] = self.path.name
        extra.update({key: fm[key] for key in _METADATA_KEYS if key in fm})
        device = fm.get("device")
        return Metadata(
            software=name,
            software_version=version,
            created=str(fm["start_utc"]),
            description=str(fm.get("name", "")),
            device=str(device.get("name", "")) if isinstance(device, dict) else "",
            standards=(),
            extra=extra,
        )

    def level_history(self) -> LevelHistory:
        """Return the ``log.csv`` levels as :class:`~resonus_core.data.LevelHistory`.

        ``time = log_t_s``, ``levels = log_levels`` (the :data:`LOG_LEVELS` columns),
        ``units`` ``"dB"`` for each, ``metadata = self.metadata()``, ``markers =
        tuple((m.t_s, m.label) for m in markers)``, ``start_utc`` = front matter
        ``start_utc`` as written.

        Returns
        -------
        LevelHistory
        """
        # Prepare units dict: each level name maps to "dB"
        units = {name: "dB" for name in LOG_LEVELS}
        # Prepare markers tuple
        markers = tuple((m.t_s, m.label) for m in self.markers)
        # Start UTC from front matter
        start_utc = str(self.front_matter.get("start_utc", ""))
        return LevelHistory(
            self.log_t_s,
            self.log_levels,
            units,
            self.metadata(),
            markers,
            start_utc,
        )

    def band_results(self, segment: int | None = None) -> BandResults:
        """Return band levels as :class:`~resonus_core.data.BandResults`.

        Parameters
        ----------
        segment
            ``None``: the summary's ``rta_LZeq_db`` (the saved range; ``null`` -> NaN).
            Otherwise the row of ``rta_segments.csv`` whose ``segment`` number equals it.

        Returns
        -------
        BandResults
            ``fraction = bands.fraction``, ``center = bands.center`` (exact),
            ``labels = tuple(bands.labels)``, ``values = {"LZeq": levels}``,
            ``units = {"LZeq": "dB"}``, ``metadata = self.metadata()``.

        Raises
        ------
        ValueError
            ``"bundle has no band levels"`` if ``bands`` is None;
            ``"summary has no rta_LZeq_db"`` if ``segment`` is None and the summary lacks
            it or its length differs from the bands; ``f"no segment {segment}"`` if no
            row has that number.
        """
        if self.bands is None or self.band_segments is None:
            raise ValueError("bundle has no band levels")
        if segment is None:
            raw = self.summary.get("rta_LZeq_db")
            if not isinstance(raw, list) or len(raw) != len(self.bands):
                raise ValueError("summary has no rta_LZeq_db")
            levels = np.array([np.nan if v is None else float(v) for v in raw])
        else:
            rows = np.flatnonzero(self.segments["segment"] == segment)
            if rows.size == 0:
                raise ValueError(f"no segment {segment}")
            levels = self.band_segments[rows[0]]
        return BandResults(
            self.bands.fraction,
            self.bands.center,
            tuple(self.bands.labels),
            {"LZeq": levels},
            {"LZeq": "dB"},
            self.metadata(),
        )


def parse_version(value: object) -> tuple[int, int]:
    """Return ``(major, minor)`` of a front-matter ``version`` value.

    The integer 1 is format 1.0. A string must be ``"<major>.<minor>"`` with two
    non-negative decimal integers.

    Raises
    ------
    ValueError
        ``f"invalid version {value!r}"`` for anything else (including ``True`` and other
        integers than 1).
    """
    if type(value) is int and value == 1:
        return (1, 0)
    if isinstance(value, str):
        match = re.fullmatch(r"(\d+)\.(\d+)", value)
        if match:
            return (int(match[1]), int(match[2]))
    raise ValueError(f"invalid version {value!r}")


def parse_utc(text: str) -> datetime:
    """Parse a bundle time stamp (UTC ISO 8601, ``Z`` suffix) to an aware datetime.

    Raises
    ------
    ValueError
        ``f"invalid time {text!r}"`` if it does not parse or has no time zone.
    """
    try:
        t = datetime.fromisoformat(text)
    except ValueError:
        raise ValueError(f"invalid time {text!r}") from None
    if t.tzinfo is None:
        raise ValueError(f"invalid time {text!r}")
    return t


def combined_leq(leq_db: ArrayLike, duration_s: ArrayLike) -> float:
    """Energy mean of segment levels weighted by their durations.

    ``10 * log10(sum(T_i * 10**(L_i / 10)) / sum(T_i))`` over the segments whose level is
    not NaN (resonus-meter ``docs/FILE_FORMAT.md``, "To combine segments").

    Parameters
    ----------
    leq_db
        Segment levels in dB, 1-D; NaN for an empty segment.
    duration_s
        Segment durations in s, same shape.

    Returns
    -------
    float
        Combined level in dB.

    Raises
    ------
    ValueError
        ``"shapes differ"`` if the shapes differ; ``"no segment with a level"`` if every
        level is NaN or the durations of the others sum to <= 0.
    """
    leq = np.asarray(leq_db, dtype=np.float64)
    duration = np.asarray(duration_s, dtype=np.float64)
    if leq.shape != duration.shape:
        raise ValueError("shapes differ")
    used = ~np.isnan(leq)
    total = float(np.sum(duration[used]))
    if not used.any() or total <= 0:
        raise ValueError("no segment with a level")
    energy = np.sum(duration[used] * 10 ** (leq[used] / 10))
    return float(10 * np.log10(energy / total))


def read_session(path: str | Path) -> MeterSession:
    r"""Read a resonus-meter measurement bundle.

    Parameters
    ----------
    path
        The bundle folder (containing ``measurement.md``).

    Returns
    -------
    MeterSession

    Raises
    ------
    FileNotFoundError
        If ``measurement.md`` or a listed CSV file is missing.
    ValueError
        With these messages (``{...}`` filled in):

        - ``"no front matter"``: the text does not start with ``"---\n"`` or has no
          closing ``"\n---\n"`` line; ``"front matter is not a mapping"``;
        - ``f"not a {FORMAT} bundle"``: ``format`` missing or different;
        - ``f"invalid version {value!r}"``, ``f"unsupported major version {version}"``;
        - ``f"missing files.{key}"`` for ``log``, ``markers``, ``segments`` (and
          ``rta_log``, ``rta_segments`` when the front matter has ``rta``);
          ``f"file outside the bundle: {name}"`` for an absolute path or one that leaves
          the folder;
        - ``f"{file}: missing column {column}"``;
        - ``f"{file}: line {line}: {message}"`` for a cell that does not parse
          (``line`` counts the header as line 1);
        - ``"invalid rta block"``: ``bands_per_octave`` not 1 or 3, or ``bands_hz`` not
          a non-empty list of nominal mid-band frequencies of consecutive bands;
        - ``f"{file}: {n} rows, expected {m}"``: ``rta_log.csv`` and ``log.csv`` (or
          ``rta_segments.csv`` and ``segments.csv``) differ in length.
    """
    folder = Path(path)
    text = (folder / "measurement.md").read_text(encoding="utf-8")
    end = text.find("\n---\n", 3)
    if not text.startswith("---\n") or end < 0:
        raise ValueError("no front matter")
    front_matter = yaml.safe_load(text[4 : end + 1])
    if not isinstance(front_matter, dict):
        raise ValueError("front matter is not a mapping")
    body = text[end + 5 :]
    if front_matter.get("format") != FORMAT:
        raise ValueError(f"not a {FORMAT} bundle")
    major, minor = parse_version(front_matter.get("version"))
    version = f"{major}.{minor}"
    if major > SUPPORTED_MAJOR:
        raise ValueError(f"unsupported major version {version}")
    files = front_matter.get("files")
    files = files if isinstance(files, dict) else {}
    match = _SUMMARY.search(body)
    summary = (yaml.safe_load(match[1]) or {}) if match else {}

    log = _Table.read(folder, files, "log", ("t_s", "utc", "marker_id", "clip", *LOG_LEVELS))
    log_rows = log.parse(
        lambda r: (
            float(r["t_s"]),
            parse_utc(r["utc"]),
            int(r["marker_id"]) if r["marker_id"] else 0,
            r["clip"] == "1",
            [_number(r[c]) for c in LOG_LEVELS],
        )
    )
    marker_table = _Table.read(
        folder, files, "markers", ("id", "t_s", "utc", "kind", "label", "note")
    )
    markers = marker_table.parse(
        lambda r: Marker(
            int(r["id"]), float(r["t_s"]), parse_utc(r["utc"]), r["kind"], r["label"], r["note"]
        )
    )
    segment_table = _Table.read(folder, files, "segments", ("segment", "duration_s"))
    segment_rows = segment_table.parse(lambda r: [_number(r[c]) for c in segment_table.header])
    segments = {
        column: np.array([row[i] for row in segment_rows], dtype=np.float64).reshape(-1)
        for i, column in enumerate(segment_table.header)
    }

    bands = band_log = band_segments = None
    if "rta" in front_matter:
        bands = _band_set(front_matter["rta"])
        band_log = _Table.read(folder, files, "rta_log", ()).bands(bands, len(log_rows))
        band_segments = _Table.read(folder, files, "rta_segments", ()).bands(
            bands, len(segment_rows)
        )
    return MeterSession(
        path=folder,
        version=version,
        front_matter=front_matter,
        body=body,
        summary=summary if isinstance(summary, dict) else {},
        log_t_s=np.array([row[0] for row in log_rows], dtype=np.float64),
        log_utc=tuple(row[1] for row in log_rows),
        log_marker=np.array([row[2] for row in log_rows], dtype=np.int64),
        log_clip=np.array([row[3] for row in log_rows], dtype=np.bool_),
        log_levels={
            column: np.array([row[4][i] for row in log_rows], dtype=np.float64)
            for i, column in enumerate(LOG_LEVELS)
        },
        markers=tuple(markers),
        segments=segments,
        bands=bands,
        band_log=band_log,
        band_segments=band_segments,
    )


def _number(cell: str) -> float:
    """Parse a CSV cell: blank -> NaN, else ``float``."""
    return np.nan if cell == "" else float(cell)


def _band_set(rta: object) -> BandSet:
    """Return the bands of a front-matter ``rta`` block, or raise "invalid rta block"."""
    try:
        if not isinstance(rta, dict):
            raise TypeError
        fraction = rta["bands_per_octave"]
        nominal = [float(f) for f in rta["bands_hz"]]
        if fraction not in (1, 3) or not nominal:
            raise ValueError
        index = band_index(nominal, fraction)
        if not np.array_equal(nominal_frequency(index, fraction), nominal):
            raise ValueError
        return BandSet(fraction, tuple(int(i) for i in index))
    except (ValueError, TypeError, KeyError):
        raise ValueError("invalid rta block") from None


@dataclass(frozen=True)
class _Table:
    """One CSV file of a bundle: its name as listed in ``files``, header and rows."""

    name: str
    header: list[str]
    rows: list[dict[str, str]]

    @classmethod
    def read(
        cls, folder: Path, files: dict[str, Any], key: str, columns: tuple[str, ...]
    ) -> _Table:
        """Read ``files[key]`` inside ``folder``; check the required ``columns``."""
        if key not in files:
            raise ValueError(f"missing files.{key}")
        name = str(files[key])
        path = folder / name
        if Path(name).is_absolute() or not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError(f"file outside the bundle: {name}")
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            header = list(reader.fieldnames or [])
            rows = list(reader)
        for column in columns:
            if column not in header:
                raise ValueError(f"{name}: missing column {column}")
        return cls(name, header, rows)

    def parse[T](self, convert: Callable[[dict[str, str]], T]) -> list[T]:
        """Convert every row; a ValueError names the file and line (header = line 1)."""
        out = []
        for line, row in enumerate(self.rows, start=2):
            try:
                out.append(convert(row))
            except ValueError as error:
                raise ValueError(f"{self.name}: line {line}: {error}") from None
        return out

    def bands(self, bands: BandSet, rows: int) -> NDArray[np.float64]:
        """Return the band columns as ``(rows, len(bands))``; check columns and row count."""
        by_frequency = {}
        for column in self.header:
            if column.startswith(BAND_PREFIX):
                try:
                    by_frequency[float(column[len(BAND_PREFIX) :])] = column
                except ValueError:
                    continue
        columns = []
        for f in bands.nominal:
            if float(f) not in by_frequency:
                raise ValueError(f"{self.name}: missing column {BAND_PREFIX}{f:g}")
            columns.append(by_frequency[float(f)])
        data = self.parse(lambda r: [_number(r[c]) for c in columns])
        if len(data) != rows:
            raise ValueError(f"{self.name}: {len(data)} rows, expected {rows}")
        return np.array(data, dtype=np.float64).reshape(rows, len(columns))

"""Common data model for measurement results, and its HDF5 file format.

Every app expresses its results in these types so the viewer can list, plot and
combine results from any app. One result per ``.h5`` file; the file is
self-describing (``kind``, format version, metadata as attributes).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import ArrayLike, NDArray

FORMAT_VERSION = 1


@dataclass(frozen=True)
class Metadata:
    """Provenance of a result.

    Parameters
    ----------
    software
        Name of the app that produced the result.
    software_version
        Version of that app.
    created
        ISO 8601 timestamp (UTC); set automatically if empty.
    description
        Free text, e.g. measurement position.
    device
        Audio device / measurement chain description.
    standards
        Standards (with edition and clause) the result was computed to.
    extra
        Further key/value pairs (JSON-serializable).
    """

    software: str
    software_version: str
    created: str = ""
    description: str = ""
    device: str = ""
    standards: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Fill in the creation time if it was not given."""
        if not self.created:
            object.__setattr__(self, "created", datetime.now(UTC).isoformat(timespec="seconds"))


@dataclass(frozen=True)
class ImpulseResponse:
    """Impulse response sampled at ``sample_rate``.

    Parameters
    ----------
    data
        Samples, 1-D.
    sample_rate
        Sample rate in Hz.
    unit
        Unit of the samples, e.g. ``"Pa/V"`` or ``"1"`` for uncalibrated data.
    metadata
        Provenance.
    """

    data: NDArray[np.float64]
    sample_rate: float
    unit: str
    metadata: Metadata

    def __post_init__(self) -> None:
        """Validate shape and sample rate."""
        object.__setattr__(self, "data", _as_1d(self.data, np.float64))
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

    @property
    def time(self) -> NDArray[np.float64]:
        """Time axis in s."""
        return np.arange(self.data.size) / self.sample_rate


@dataclass(frozen=True)
class FrequencyResponse:
    """Complex frequency response.

    Parameters
    ----------
    frequency
        Frequencies in Hz, 1-D, strictly increasing, non-negative.
    values
        Complex response at each frequency.
    unit
        Unit of the response, e.g. ``"Pa/V"`` or ``"1"``.
    metadata
        Provenance.
    """

    frequency: NDArray[np.float64]
    values: NDArray[np.complex128]
    unit: str
    metadata: Metadata

    def __post_init__(self) -> None:
        """Validate shapes and the frequency axis."""
        f = _as_1d(self.frequency, np.float64)
        v = _as_1d(self.values, np.complex128)
        if f.size != v.size:
            raise ValueError("frequency and values must have the same length")
        if np.any(f < 0) or np.any(np.diff(f) <= 0):
            raise ValueError("frequency must be non-negative and strictly increasing")
        object.__setattr__(self, "frequency", f)
        object.__setattr__(self, "values", v)

    def magnitude_db(self) -> NDArray[np.float64]:
        """Magnitude in dB re 1 unit, ``20*log10(|values|)``; zeros give -inf."""
        with np.errstate(divide="ignore"):
            return 20 * np.log10(np.abs(self.values))


@dataclass(frozen=True)
class BandResults:
    """Scalar results per frequency band, e.g. reverberation times or absorption coefficients.

    Parameters
    ----------
    fraction
        Bandwidth designator ``b`` of the 1/b-octave bands (1 = octave, 3 = one-third
        octave), at least 1.
    center
        Exact mid-band frequencies in Hz, 1-D, positive, strictly increasing.
    labels
        Nominal band labels (e.g. ``"125"``, ``"1k"``), one per band.
    values
        Quantity name -> values per band (1-D float, length = number of bands; NaN where
        not available), e.g. ``{"T20": ..., "T30": ...}``. At least one quantity.
    units
        Quantity name -> unit, e.g. ``{"T20": "s"}``; same keys as ``values``.
    metadata
        Provenance (standards with edition and clause belong in ``metadata.standards``).
    flags
        Optional warnings per band: empty, or one tuple of strings per band.

    Raises
    ------
    ValueError
        On inconsistent shapes or keys (see ``__post_init__``).
    """

    fraction: int
    center: NDArray[np.float64]
    labels: tuple[str, ...]
    values: dict[str, NDArray[np.float64]]
    units: dict[str, str]
    metadata: Metadata
    flags: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        """Validate and normalize the fields.

        - ``fraction`` must be an int >= 1.
        - ``center = _as_1d(center, np.float64)``; all > 0 and strictly increasing.
        - ``labels = tuple(str(s) for s in labels)``; ``len(labels) == n`` (n = number of
          bands = ``center.size``).
        - ``values``: non-empty; keys equal to the keys of ``units``; each value is
          converted with ``_as_1d(v, np.float64)`` and must have length n; stored as a new
          dict (``object.__setattr__``). ``units`` stored as a new ``dict[str, str]``.
        - ``flags = tuple(tuple(str(f) for f in band) for band in flags)``; empty or length n.

        Raises
        ------
        ValueError
            If any rule is violated.
        """
        # Validate fraction
        if not isinstance(self.fraction, int) or self.fraction < 1:
            raise ValueError("fraction must be an int >= 1")
        # Validate center
        center = _as_1d(self.center, np.float64)
        if np.any(center <= 0) or np.any(np.diff(center) <= 0):
            raise ValueError("center must be > 0 and strictly increasing")
        # Validate labels
        labels = tuple(str(s) for s in self.labels)
        n = center.size
        if len(labels) != n:
            raise ValueError("labels length must match number of bands")
        # Validate values and units
        if not self.values:
            raise ValueError("values must be non-empty")
        if set(self.values.keys()) != set(self.units.keys()):
            raise ValueError("values and units keys must match")
        values_dict = {}
        for name, v in self.values.items():
            arr = _as_1d(v, np.float64)
            if arr.size != n:
                raise ValueError(f"values[{name}] length must match number of bands")
            values_dict[name] = arr
        # Normalize units
        units_dict = {k: str(v) for k, v in self.units.items()}
        # Validate flags
        # Accept list or tuple for flags, convert to tuple of tuples of strings
        if not isinstance(self.flags, (tuple, list)):
            raise ValueError("flags must be a tuple or list")
        if len(self.flags) not in (0, n):
            raise ValueError("flags length must be 0 or number of bands")
        flags_tuple = tuple(tuple(str(f) for f in band) for band in self.flags)
        # Set attributes
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "values", values_dict)
        object.__setattr__(self, "units", units_dict)
        object.__setattr__(self, "flags", flags_tuple)


@dataclass(frozen=True)
class LevelHistory:
    """Levels over time, e.g. the log of a sound level meter (D-S1, Q-C8).

    Parameters
    ----------
    time
        Times in s from ``start_utc``, 1-D, finite, strictly increasing.
    levels
        Quantity name -> values (1-D float, length = ``time.size``; NaN = missing,
        +-inf allowed), e.g. ``{"LAF": ..., "LCF": ...}``. At least one quantity.
    units
        Quantity name -> unit, e.g. ``{"LAF": "dB"}``; same keys as ``levels``.
    metadata
        Provenance.
    markers
        ``(time_s, label)`` pairs (times finite, in s from ``start_utc``), e.g. the
        meter's markers; may be empty.
    start_utc
        ISO 8601 UTC time of ``time == 0``; ``""`` if unknown.

    Raises
    ------
    ValueError
        On inconsistent shapes or keys (see ``__post_init__``).
    """

    time: NDArray[np.float64]
    levels: dict[str, NDArray[np.float64]]
    units: dict[str, str]
    metadata: Metadata
    markers: tuple[tuple[float, str], ...] = ()
    start_utc: str = ""

    def __post_init__(self) -> None:
        """Validate and normalize the fields.

        - ``time = _as_1d(time, np.float64)``; all finite and strictly increasing, else
          ``ValueError("time must be finite and strictly increasing")``.
        - ``levels``: non-empty (``"levels must be non-empty"``); keys equal to the keys
          of ``units`` (``"levels and units keys must match"``); each value
          ``_as_1d(v, np.float64)`` of length ``time.size``
          (``f"levels[{name}] length must match time"``); stored as a new dict.
          ``units`` stored as a new ``dict[str, str]``.
        - ``markers = tuple((float(t), str(label)) for t, label in markers)``; every
          time finite, else ``ValueError("marker times must be finite")``.
        - ``start_utc = str(start_utc)``.

        Raises
        ------
        ValueError
            If any rule is violated.
        """
        # Validate time
        time_arr = _as_1d(self.time, np.float64)
        if not np.all(np.isfinite(time_arr)) or not np.all(np.diff(time_arr) > 0):
            raise ValueError("time must be finite and strictly increasing")
        # Validate levels
        if not self.levels:
            raise ValueError("levels must be non-empty")
        if set(self.levels.keys()) != set(self.units.keys()):
            raise ValueError("levels and units keys must match")
        levels_dict: dict[str, NDArray[np.float64]] = {}
        for name, v in self.levels.items():
            arr = _as_1d(v, np.float64)
            if arr.size != time_arr.size:
                raise ValueError(f"levels[{name}] length must match time")
            levels_dict[name] = arr
        # Normalize units
        units_dict = {k: str(v) for k, v in self.units.items()}
        # Normalize markers
        markers_tuple = tuple((float(t), str(label)) for t, label in self.markers)
        for t, _ in markers_tuple:
            if not np.isfinite(t):
                raise ValueError("marker times must be finite")
        # Normalize start_utc
        start_utc_str = str(self.start_utc)
        # Set attributes
        object.__setattr__(self, "time", time_arr)
        object.__setattr__(self, "levels", levels_dict)
        object.__setattr__(self, "units", units_dict)
        object.__setattr__(self, "markers", markers_tuple)
        object.__setattr__(self, "start_utc", start_utc_str)


@dataclass(frozen=True)
class TimeSignal:
    """A recorded signal, e.g. background noise in Pa (resonus-room Q-C9).

    Unlike :class:`ImpulseResponse` it is not a system response: apps store recordings
    with this kind instead of an impulse response with a role key.

    Parameters
    ----------
    data
        Samples, 1-D, non-empty (one channel per result, as :class:`ImpulseResponse`);
        NaN allowed.
    sample_rate
        Sample rate in Hz, > 0.
    unit
        Unit of the samples, e.g. ``"Pa"`` or ``"FS"`` (digital full scale).
    metadata
        Provenance.
    start_utc
        ISO 8601 UTC time of the first sample; ``""`` if unknown.

    Raises
    ------
    ValueError
        ``"expected a non-empty 1-D array"`` (from ``_as_1d``), ``"sample_rate must be
        positive"``.
    """

    data: NDArray[np.float64]
    sample_rate: float
    unit: str
    metadata: Metadata
    start_utc: str = ""

    def __post_init__(self) -> None:
        """Validate and normalize the fields.

        ``data = _as_1d(data, np.float64)``; ``sample_rate <= 0`` gives
        ``ValueError("sample_rate must be positive")``; store ``data``, ``float(
        sample_rate)``, ``str(unit)``, ``str(start_utc)`` (``object.__setattr__``).
        """
        data_arr = _as_1d(self.data, np.float64)
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        object.__setattr__(self, "data", data_arr)
        object.__setattr__(self, "sample_rate", float(self.sample_rate))
        object.__setattr__(self, "unit", str(self.unit))
        object.__setattr__(self, "start_utc", str(self.start_utc))

    @property
    def time(self) -> NDArray[np.float64]:
        """Time axis in s from the first sample: ``arange(data.size) / sample_rate``."""
        return np.arange(self.data.size) / self.sample_rate

    @property
    def duration(self) -> float:
        """Length in s, ``data.size / sample_rate``."""
        return self.data.size / self.sample_rate


Result = ImpulseResponse | FrequencyResponse | BandResults | LevelHistory | TimeSignal
_KINDS: dict[str, type] = {
    "ImpulseResponse": ImpulseResponse,
    "FrequencyResponse": FrequencyResponse,
    "BandResults": BandResults,
    "LevelHistory": LevelHistory,
    "TimeSignal": TimeSignal,
}


def save(path: str | Path, result: Result) -> None:
    """Write a result to an HDF5 file (overwrites).

    Parameters
    ----------
    path
        Target file, conventionally ``*.h5``.
    result
        Result to store.

    Notes
    -----
    BandResults are stored with attributes:
    - fraction
    - labels (JSON list)
    - quantities (JSON list of value names)
    - units (JSON dict)
    - flags (JSON list of lists)
    - center dataset
    - values group with one dataset per quantity.
    """
    with h5py.File(path, "w") as h5:
        h5.attrs["format"] = "resonus"
        h5.attrs["format_version"] = FORMAT_VERSION
        h5.attrs["kind"] = type(result).__name__
        if isinstance(result, (ImpulseResponse, FrequencyResponse)):
            h5.attrs["unit"] = result.unit
        h5.attrs["metadata"] = json.dumps(asdict(result.metadata))
        if isinstance(result, ImpulseResponse):
            h5.attrs["sample_rate"] = result.sample_rate
            h5.create_dataset("data", data=result.data)
        elif isinstance(result, FrequencyResponse):
            h5.create_dataset("frequency", data=result.frequency)
            h5.create_dataset("values", data=result.values)
        if isinstance(result, BandResults):
            h5.attrs["fraction"] = result.fraction
            h5.attrs["labels"] = json.dumps(list(result.labels))
            h5.attrs["quantities"] = json.dumps(list(result.values))
            h5.attrs["units"] = json.dumps(result.units)
            h5.attrs["flags"] = json.dumps([list(f) for f in result.flags])
            h5.create_dataset("center", data=result.center)
            group = h5.create_group("values")
            for name in result.values:
                group.create_dataset(name, data=result.values[name])
            return
        if isinstance(result, TimeSignal):
            h5.attrs["unit"] = result.unit
            h5.attrs["sample_rate"] = result.sample_rate
            h5.attrs["start_utc"] = result.start_utc
            h5.create_dataset("data", data=result.data)
        if isinstance(result, LevelHistory):
            _save_level_history(h5, result)


def load(path: str | Path) -> Result:
    """Read a result written by :func:`save`.

    Raises
    ------
    ValueError
        If the file is not a resonus result or has an unsupported version.
    """
    with h5py.File(path, "r") as h5:
        if h5.attrs.get("format") != "resonus":
            raise ValueError(f"{path}: not a resonus result file")
        if int(h5.attrs["format_version"]) > FORMAT_VERSION:
            raise ValueError(
                f"{path}: format version {h5.attrs['format_version']} is newer than supported"
            )
        kind = str(h5.attrs["kind"])
        if kind not in _KINDS:
            raise ValueError(f"{path}: unknown result kind {kind!r}")
        meta_dict = json.loads(h5.attrs["metadata"])
        meta_dict["standards"] = tuple(meta_dict["standards"])
        meta = Metadata(**meta_dict)
        if kind == "LevelHistory":
            return _load_level_history(h5, meta)
        if kind == "BandResults":
            return BandResults(
                int(h5.attrs["fraction"]),
                h5["center"][()],
                tuple(json.loads(h5.attrs["labels"])),
                {name: h5["values"][name][()] for name in json.loads(h5.attrs["quantities"])},
                json.loads(h5.attrs["units"]),
                meta,
                tuple(tuple(f) for f in json.loads(h5.attrs["flags"])),
            )
        unit = str(h5.attrs["unit"])
        if kind == "ImpulseResponse":
            return ImpulseResponse(h5["data"][()], float(h5.attrs["sample_rate"]), unit, meta)
        if kind == "TimeSignal":
            return TimeSignal(
                h5["data"][()],
                float(h5.attrs["sample_rate"]),
                unit,
                meta,
                str(h5.attrs["start_utc"]),
            )
        return FrequencyResponse(h5["frequency"][()], h5["values"][()], unit, meta)


def _save_level_history(h5: h5py.File, result: LevelHistory) -> None:
    """Write the ``LevelHistory`` items (docs/FILE_FORMAT.md, kind ``LevelHistory``).

    Attributes ``quantities`` (JSON list of the level names, in order), ``units`` (JSON
    object), ``markers`` (JSON list of ``[time_s, label]``), ``start_utc`` (string);
    dataset ``time``; group ``levels`` with one dataset per quantity.
    """
    # Quantities and units
    h5.attrs["quantities"] = json.dumps(list(result.levels))
    h5.attrs["units"] = json.dumps(result.units)
    # Markers
    h5.attrs["markers"] = json.dumps([[t, label] for t, label in result.markers])
    # Start UTC
    h5.attrs["start_utc"] = result.start_utc
    # Time dataset
    h5.create_dataset("time", data=result.time)
    # Levels group
    levels_grp = h5.create_group("levels")
    for name, arr in result.levels.items():
        levels_grp.create_dataset(name, data=arr)


def _load_level_history(h5: h5py.File, meta: Metadata) -> LevelHistory:
    """Read what :func:`_save_level_history` wrote (markers back to tuples)."""
    quantities = json.loads(h5.attrs["quantities"])
    units = json.loads(h5.attrs["units"])
    markers_json = json.loads(h5.attrs["markers"])
    markers = tuple((float(t), str(label)) for t, label in markers_json)
    start_utc = str(h5.attrs["start_utc"])
    time_arr = h5["time"][()]
    levels_dict = {name: h5["levels"][name][()] for name in quantities}
    return LevelHistory(time_arr, levels_dict, units, meta, markers, start_utc)


def _as_1d(a: ArrayLike, dtype: type) -> NDArray[Any]:
    arr: NDArray[Any] = np.asarray(a, dtype=dtype)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("expected a non-empty 1-D array")
    return arr

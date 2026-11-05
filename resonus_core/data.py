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


Result = ImpulseResponse | FrequencyResponse | BandResults
_KINDS: dict[str, type] = {
    "ImpulseResponse": ImpulseResponse,
    "FrequencyResponse": FrequencyResponse,
    "BandResults": BandResults,
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
        return FrequencyResponse(h5["frequency"][()], h5["values"][()], unit, meta)


def _as_1d(a: ArrayLike, dtype: type) -> NDArray[Any]:
    arr: NDArray[Any] = np.asarray(a, dtype=dtype)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("expected a non-empty 1-D array")
    return arr

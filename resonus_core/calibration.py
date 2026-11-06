"""Level calibration of an input channel with an acoustic calibrator (D-C1).

The method is the one of resonus-meter (Rust, ``spl-dsp/src/calibration.rs`` and
``spl-engine/src/calibration_store``; review 2026-11-05), ported identically so both apps
calibrate the same way, plus the two corrections of IEC 61672-3:2013 (ČSN EN 61672-3 ed. 2)
9.1 and 10.2.

Level convention (the same as :mod:`resonus_core.signals`): samples are normalised to
full scale (amplitude 1), ``L_FS = 10 lg mean(x**2)`` in dB re full scale (a full-scale
sine reads -3.01 dB), and the sound pressure level is ``L_p = L_FS + C`` in dB re 20 µPa
with one calibration offset ``C`` per input (device, channel).

Calibration (:func:`propose`):

1. The calibrator tone is recorded raw (no weighting), by default ``CAPTURE_DURATION``.
2. Its frequency is searched in the first second within ±``FREQUENCY_TOLERANCE`` of the
   calibrator's nominal frequency (Hann window, power at 1-Hz bin centres, parabolic
   refinement on the log power), then its power is measured with a lock-in at that
   frequency on every whole second and averaged.
3. Quality checks: no clipping, signal and tone above ``MIN_LEVEL_DBFS``, tone-to-residual
   ratio at least ``MIN_TONE_TO_BACKGROUND_DB``, standard deviation of the 1-s tone levels
   below ``MAX_STABILITY_DB``.
4. ``C = L_cal + ΔL_ambient + ΔL_adjustment - L_tone``. ``L_cal`` is the calibrator's
   level (certificate); ``ΔL_ambient`` the effect of the ambient conditions on the level in
   the calibrator's coupler (IEC 61672-3:2013 10.2: "in accordance with the procedure and
   data given in the instruction manual for the sound calibrator", e.g. a pistonphone's
   barometric correction); ``ΔL_adjustment`` the manufacturer's data for this
   calibrator-microphone combination (9.1). Both default to 0 dB. IEC 60942:2017 (ČSN EN
   IEC 60942 ed. 2) 5.1.5, 5.1.7 and Table 1: only class LS/M and 1/M pistonphones need a
   correction, for static pressure only, stated in their instruction manual relative to the
   reference conditions of its Clause 4 (101.325 kPa, 23 °C, 50 %); all other calibrators
   need none. The proposal is shown to the user and saved only on confirmation. The
   calibrator must have stabilised before the capture (IEC 60942:2017 5.1.10: at most
   30 s, stated in its manual).

Records are kept in an append-only TOML history per input profile (:class:`CalibrationStore`,
the file format of resonus-meter's ``calibrations.toml``, schema version 1, with the two
correction fields added). The thresholds are the meter's design choices (its
specification 4.9); no standard prescribes them.
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import tomli_w
from numpy.typing import ArrayLike, NDArray

# Reference sound pressure in Pa, ISO 1683:2015 Table 1 (in air).
P0: Final = 20e-6
# Defaults and limits of the calibration capture (resonus-meter specification 4.9, design
# choices; D-C1):
CAPTURE_DURATION: Final = 8.0  # s
FREQUENCY_TOLERANCE: Final = 0.03  # ±3 % of the nominal calibrator frequency
MIN_LEVEL_DBFS: Final = -60.0  # dB re full scale, signal and tone
MIN_TONE_TO_BACKGROUND_DB: Final = 20.0  # dB
MAX_STABILITY_DB: Final = 0.1  # dB, standard deviation of the 1-s tone levels (strict <)
# |x| >= CLIP_LEVEL counts as clipped: the top code of a 16-bit converter, 32767/32768, so
# any converter of 16 bits or more reaches it at full scale. One value for the whole stack
# (user, 2026-11-05; design choice, resonus-room D-21); resonus-meter used 0.999 before.
CLIP_LEVEL: Final = 1.0 - 2.0**-15
# Calibration file schema version (resonus-meter ADR 0006).
SCHEMA_VERSION: Final = 1

FailureReason = Literal[
    "empty", "invalid_frequency", "no_tone", "too_quiet", "clipped", "noisy", "unstable"
]
# Messages of the failures, as resonus-meter's CalibrationError::fmt.
FAILURE_MESSAGES: Final[dict[str, str]] = {
    "empty": "no samples were captured",
    "invalid_frequency": "calibration frequency is outside the valid range",
    "no_tone": "no tone was found near the calibration frequency",
    "too_quiet": "calibration tone is below -60 dBFS",
    "clipped": "the calibration capture clipped",
    "noisy": "background noise is too high for calibration",
    "unstable": "calibration tone level is unstable",
}


class CalibrationError(ValueError):
    """A calibration capture is unsuitable.

    Attributes
    ----------
    reason : str
        One of ``FAILURE_MESSAGES`` (key); the exception message is its value.
    """

    def __init__(self, reason: FailureReason) -> None:
        super().__init__(FAILURE_MESSAGES[reason])
        self.reason = reason


@dataclass(frozen=True)
class CalibrationProposal:
    """A proposed calibration; saved only when the user confirms it.

    Parameters
    ----------
    calibrator_level_db
        Calibrator level ``L_cal`` in dB re 20 µPa (from its certificate).
    measured_level_dbfs
        Measured tone level ``L_tone`` in dB re full scale.
    frequency_hz
        Detected tone frequency in Hz (used for the lock-in).
    offset_db
        ``C = L_cal + ambient_correction_db + adjustment_correction_db - L_tone`` in dB.
    stability_db
        Standard deviation of the 1-s tone levels in dB.
    tone_to_background_db
        ``10 lg(tone power / residual power)`` in dB.
    peak_dbfs
        Largest ``|x|`` of the capture in dB re full scale.
    ambient_correction_db, adjustment_correction_db
        Corrections of IEC 61672-3:2013 10.2 and 9.1 in dB (0 if none).
    """

    calibrator_level_db: float
    measured_level_dbfs: float
    frequency_hz: float
    offset_db: float
    stability_db: float
    tone_to_background_db: float
    peak_dbfs: float
    ambient_correction_db: float = 0.0
    adjustment_correction_db: float = 0.0


def lock_in_power(x: ArrayLike, sample_rate: float, frequency: float) -> float:
    """Power of the component of ``x`` at ``frequency``: ``2 |sum x[n] e^(-j w n)|^2 / N^2``.

    ``w = 2 pi frequency / sample_rate``, ``n = 0..N-1``, ``N = len(x)``. For ``x =
    a sin(w n + phi)`` over whole periods this is ``a**2 / 2``, the tone's mean square.

    Raises
    ------
    ValueError
        ``"empty signal"`` for ``N == 0``.
    """
    a = np.asarray(x, dtype=np.float64)
    n = a.size
    if n == 0:
        raise ValueError("empty signal")
    w = 2 * np.pi * frequency / sample_rate
    s = np.sum(a * np.exp(-1j * w * np.arange(n)))
    return float(2 * abs(s) ** 2 / n**2)


def hann_power(x: ArrayLike, sample_rate: float, frequency: float) -> float:
    """Return ``|sum x[n] h[n] e^(-j w n)|^2``, Hann-windowed (tone search only).

    ``h[n] = 0.5 - 0.5 cos(2 pi n / (N - 1))``, ``w = 2 pi frequency / sample_rate``. Used
    only to locate the tone (relative values).

    Raises
    ------
    ValueError
        ``"signal needs at least 2 samples"`` for ``N < 2``.
    """
    a = np.asarray(x, dtype=np.float64)
    n = a.size
    if n < 2:
        raise ValueError("signal needs at least 2 samples")
    k = np.arange(n)
    h = 0.5 - 0.5 * np.cos(2 * np.pi * k / (n - 1))
    w = 2 * np.pi * frequency / sample_rate
    return float(abs(np.sum(a * h * np.exp(-1j * w * k))) ** 2)


def find_tone_frequency(x: ArrayLike, sample_rate: float, frequency: float) -> float:
    """Frequency of the calibrator tone near ``frequency``, from the first second of ``x``.

    1. ``w = x[:round(sample_rate)]``; fewer samples: ``CalibrationError("empty")``.
    2. ``df = sample_rate / len(w)``; bins ``k = ceil(frequency * (1 - FREQUENCY_TOLERANCE)
       / df)`` to ``floor(frequency * (1 + FREQUENCY_TOLERANCE) / df)``; if the first is not
       below the last: ``CalibrationError("no_tone")``.
    3. ``p[k] = hann_power(w, sample_rate, k * df)``; the largest ``p`` (first one on a
       tie); if it is ``<= sys.float_info.min``: ``CalibrationError("no_tone")``.
    4. Largest at the first or last bin: return ``k * df``. Otherwise, with ``l, c, r`` the
       natural logs of the three powers around it (each ``max(p, float_info.min)``),
       ``d = l - 2c + r``; ``a = clip(0.5 (l - r) / d, -0.5, 0.5)`` if ``|d| >
       float_info.min`` else 0; return ``(k + a) * df``.
    """
    a = np.asarray(x, dtype=np.float64)
    length = round(sample_rate)
    if length < 2 or a.size < length:
        raise CalibrationError("empty")
    first = a[:length]
    df = sample_rate / length
    k_low = math.ceil(frequency * (1 - FREQUENCY_TOLERANCE) / df)
    k_high = math.floor(frequency * (1 + FREQUENCY_TOLERANCE) / df)
    if k_low >= k_high:
        raise CalibrationError("no_tone")
    powers = [hann_power(first, sample_rate, k * df) for k in range(k_low, k_high + 1)]
    i = int(np.argmax(powers))
    tiny = sys.float_info.min
    if powers[i] <= tiny:
        raise CalibrationError("no_tone")
    k = k_low + i
    if i == 0 or i == len(powers) - 1:
        return float(k * df)
    left, centre, right = (math.log(max(v, tiny)) for v in powers[i - 1 : i + 2])
    d = left - 2 * centre + right
    shift = min(max(0.5 * (left - right) / d, -0.5), 0.5) if abs(d) > tiny else 0.0
    return float((k + shift) * df)


def propose(
    samples: ArrayLike,
    sample_rate: float,
    frequency: float,
    calibrator_level_db: float,
    ambient_correction_db: float = 0.0,
    adjustment_correction_db: float = 0.0,
) -> CalibrationProposal:
    """Calibration proposal from a raw capture of a calibrator tone (1-D samples).

    Steps and checks in this order (each failure raises :class:`CalibrationError`):

    1. No samples: ``"empty"``. ``frequency`` not in ``(0, sample_rate / 2)``:
       ``"invalid_frequency"``.
    2. Any ``|x| >= CLIP_LEVEL``: ``"clipped"``. ``10 lg mean(x**2) < MIN_LEVEL_DBFS``
       (silence: ``-inf``): ``"too_quiet"``. ``peak_dbfs = 20 lg max|x|``.
    3. ``f = find_tone_frequency(x, sample_rate, frequency)``.
    4. ``L = round(sample_rate)``; the whole seconds ``x[i L:(i + 1) L]`` (a shorter tail
       is ignored); none: ``"empty"``. ``tone_power = mean`` of their
       :func:`lock_in_power` at ``f``; ``L_tone = 10 lg tone_power``; ``< MIN_LEVEL_DBFS``:
       ``"too_quiet"``.
    5. ``total_power = mean(x**2)`` over those whole seconds; ``residual = max(total_power
       - tone_power, sys.float_info.min)``; ``tone_to_background_db = 10 lg(tone_power /
       residual)``; ``< MIN_TONE_TO_BACKGROUND_DB``: ``"noisy"``.
    6. ``stability_db`` = sample standard deviation (``ddof=1``) of the 1-s levels ``10 lg
       p_i`` (0 for one second); ``>= MAX_STABILITY_DB``: ``"unstable"``.
    7. ``offset_db = calibrator_level_db + ambient_correction_db +
       adjustment_correction_db - L_tone``.

    Parameters
    ----------
    samples
        Raw (unweighted) samples, full scale 1.
    sample_rate
        Sample rate in Hz.
    frequency
        Nominal calibrator frequency in Hz (e.g. 1000; 250 for pistonphones).
    calibrator_level_db
        Calibrator level in dB re 20 µPa (e.g. 94 or 114).
    ambient_correction_db
        IEC 61672-3:2013 10.2 correction for the ambient conditions, from the
        calibrator's instruction manual (e.g. the static-pressure correction of a class
        LS/M or 1/M pistonphone, IEC 60942:2017 5.1.5).
    adjustment_correction_db
        IEC 61672-3:2013 9.1 adjustment data for the calibrator-microphone combination.

    Returns
    -------
    CalibrationProposal

    References
    ----------
    resonus-meter specification 4.9 (method); IEC 61672-3:2013, 9.1 and 10.2
    (corrections); IEC 60942:2017, 4, 5.1.5, 5.1.7 (when corrections apply);
    ISO 1683:2015 (reference sound pressure).
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.size == 0:
        raise CalibrationError("empty")
    if not 0 < frequency < sample_rate / 2:
        raise CalibrationError("invalid_frequency")
    peak = float(np.max(np.abs(x)))
    if peak >= CLIP_LEVEL:
        raise CalibrationError("clipped")
    mean_square = float(np.mean(x**2))
    if mean_square <= 0 or 10 * math.log10(mean_square) < MIN_LEVEL_DBFS:
        raise CalibrationError("too_quiet")
    f = find_tone_frequency(x, sample_rate, frequency)
    length = round(sample_rate)
    seconds = x.size // length
    if seconds == 0:
        raise CalibrationError("empty")
    powers = [
        lock_in_power(x[i * length : (i + 1) * length], sample_rate, f) for i in range(seconds)
    ]
    tone_power = float(np.mean(powers))
    if tone_power <= 0 or 10 * math.log10(tone_power) < MIN_LEVEL_DBFS:
        raise CalibrationError("too_quiet")
    tone_level = 10 * math.log10(tone_power)
    total_power = float(np.mean(x[: seconds * length] ** 2))
    residual = max(total_power - tone_power, sys.float_info.min)
    tone_to_background = 10 * math.log10(tone_power / residual)
    if tone_to_background < MIN_TONE_TO_BACKGROUND_DB:
        raise CalibrationError("noisy")
    levels = [10 * math.log10(max(p, sys.float_info.min)) for p in powers]
    stability = float(np.std(levels, ddof=1)) if seconds > 1 else 0.0
    if stability >= MAX_STABILITY_DB:
        raise CalibrationError("unstable")
    offset = calibrator_level_db + ambient_correction_db + adjustment_correction_db - tone_level
    return CalibrationProposal(
        calibrator_level_db=calibrator_level_db,
        measured_level_dbfs=tone_level,
        frequency_hz=f,
        offset_db=offset,
        stability_db=stability,
        tone_to_background_db=tone_to_background,
        peak_dbfs=20 * math.log10(peak),
        ambient_correction_db=ambient_correction_db,
        adjustment_correction_db=adjustment_correction_db,
    )


def to_pascal(samples: ArrayLike, offset_db: float) -> NDArray[np.float64]:
    """Sound pressure in Pa of samples in full-scale units: ``x * P0 * 10**(offset_db / 20)``.

    Follows from ``L_p = L_FS + C``: ``20 lg(p_rms / P0) = 10 lg mean(x**2) + C``.
    Works for any shape (float64 result).
    """
    return np.asarray(samples, dtype=np.float64) * P0 * 10 ** (offset_db / 20)


# Records and store ---------------------------------------------------------------------


@dataclass(frozen=True)
class InputProfile:
    """The input a calibration belongs to: only ``device_id`` and ``channel_index`` identify it.

    Parameters
    ----------
    device_id
        Stable device identifier (resonus-core: ``DeviceInfo.key``, ``"host API: name"``).
    device_name
        Display name at the time of calibration (for the user only).
    channel_index
        Zero-based input channel (resonus-meter convention; settings use 1-based numbers).
    sample_rate_hz
        Sample rate in Hz at the time of calibration (for the user only).
    """

    device_id: str
    device_name: str
    channel_index: int
    sample_rate_hz: int


CalibrationKind = Literal["calibrator", "manual"]

# Optional float fields of a record, in file order after "offset_db".
OPTIONAL_FIELDS: Final = (
    "calibrator_level_db",
    "tone_frequency_hz",
    "tone_level_dbfs",
    "stability_db",
    "tone_to_background_db",
    "peak_dbfs",
    "ambient_correction_db",
    "adjustment_correction_db",
)


@dataclass(frozen=True)
class CalibrationRecord:
    """One saved calibration (one ``[[calibration]]`` table of the history file).

    Parameters
    ----------
    saved_utc
        When the record was saved (or restored); timezone-aware UTC.
    calibrated_utc
        When the calibration was measured or entered; timezone-aware UTC.
    device_id, device_name, channel_index, sample_rate_hz
        As :class:`InputProfile`.
    kind
        ``"calibrator"`` (measured) or ``"manual"`` (entered).
    offset_db
        Calibration offset ``C`` in dB (``L_p = L_FS + C``).
    calibrator_level_db, tone_frequency_hz, tone_level_dbfs, stability_db,
    tone_to_background_db, peak_dbfs
        From the proposal (None for manual records).
    ambient_correction_db, adjustment_correction_db
        Corrections of IEC 61672-3:2013 10.2 / 9.1 in dB (None if not used).
    ambient_source, adjustment_source
        Where the corrections come from (free text, e.g. "pistonphone barometer,
        1013 hPa"); empty if not used.
    note
        Free text.
    """

    saved_utc: datetime
    calibrated_utc: datetime
    device_id: str
    device_name: str
    channel_index: int
    sample_rate_hz: int
    kind: CalibrationKind
    offset_db: float
    calibrator_level_db: float | None = None
    tone_frequency_hz: float | None = None
    tone_level_dbfs: float | None = None
    stability_db: float | None = None
    tone_to_background_db: float | None = None
    peak_dbfs: float | None = None
    ambient_correction_db: float | None = None
    adjustment_correction_db: float | None = None
    ambient_source: str = ""
    adjustment_source: str = ""
    note: str = ""

    @classmethod
    def from_proposal(
        cls,
        profile: InputProfile,
        proposal: CalibrationProposal,
        saved_utc: datetime,
        note: str = "",
        ambient_source: str = "",
        adjustment_source: str = "",
    ) -> CalibrationRecord:
        """Record of a confirmed calibrator capture (``kind="calibrator"``).

        ``saved_utc == calibrated_utc``; profile fields from ``profile``;
        ``calibrator_level_db``, ``offset_db``, ``stability_db``,
        ``tone_to_background_db``, ``peak_dbfs`` from the proposal, ``tone_frequency_hz =
        proposal.frequency_hz``, ``tone_level_dbfs = proposal.measured_level_dbfs``; each
        correction: its value if it is not 0, else None (and then its source "").
        """
        return cls(
            saved_utc=saved_utc,
            calibrated_utc=saved_utc,
            device_id=profile.device_id,
            device_name=profile.device_name,
            channel_index=profile.channel_index,
            sample_rate_hz=profile.sample_rate_hz,
            kind="calibrator",
            offset_db=proposal.offset_db,
            calibrator_level_db=proposal.calibrator_level_db,
            tone_frequency_hz=proposal.frequency_hz,
            tone_level_dbfs=proposal.measured_level_dbfs,
            stability_db=proposal.stability_db,
            tone_to_background_db=proposal.tone_to_background_db,
            peak_dbfs=proposal.peak_dbfs,
            ambient_correction_db=proposal.ambient_correction_db
            if proposal.ambient_correction_db != 0.0
            else None,
            adjustment_correction_db=proposal.adjustment_correction_db
            if proposal.adjustment_correction_db != 0.0
            else None,
            ambient_source=ambient_source if proposal.ambient_correction_db != 0.0 else "",
            adjustment_source=adjustment_source if proposal.adjustment_correction_db != 0.0 else "",
            note=note,
        )

    @classmethod
    def manual(
        cls, profile: InputProfile, offset_db: float, saved_utc: datetime, note: str = ""
    ) -> CalibrationRecord:
        """Record of a manually entered offset (``kind="manual"``, optional fields None)."""
        return cls(
            saved_utc=saved_utc,
            calibrated_utc=saved_utc,
            device_id=profile.device_id,
            device_name=profile.device_name,
            channel_index=profile.channel_index,
            sample_rate_hz=profile.sample_rate_hz,
            kind="manual",
            offset_db=offset_db,
            calibrator_level_db=None,
            tone_frequency_hz=None,
            tone_level_dbfs=None,
            stability_db=None,
            tone_to_background_db=None,
            peak_dbfs=None,
            ambient_correction_db=None,
            adjustment_correction_db=None,
            ambient_source="",
            adjustment_source="",
            note=note,
        )

    def restored(self, saved_utc: datetime) -> CalibrationRecord:
        """Copy with a new ``saved_utc``; everything else (``calibrated_utc``) kept."""
        return dataclasses.replace(self, saved_utc=saved_utc)

    def belongs_to(self, profile: InputProfile) -> bool:
        """Return True if ``device_id`` and ``channel_index`` equal the profile's."""
        return self.device_id == profile.device_id and self.channel_index == profile.channel_index

    def to_dict(self) -> dict[str, Any]:
        """TOML table: all fields in declaration order; times as RFC 3339 strings.

        ``saved_utc`` / ``calibrated_utc``: :func:`format_time`. Optional fields that are
        None and empty source strings are left out; ``note`` is always written.
        """
        result: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if f.name in ("saved_utc", "calibrated_utc"):
                result[f.name] = format_time(value)
            elif value is None:
                continue
            elif isinstance(value, str) and value == "" and f.name.endswith("_source"):
                continue
            else:
                result[f.name] = value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationRecord:
        """Inverse of :meth:`to_dict`; unknown keys are ignored, missing optional ones default.

        Times: :func:`parse_time` (a TOML datetime that ``tomllib`` already parsed is also
        accepted if timezone-aware).

        Raises
        ------
        ValueError
            ``f"calibration record: missing {key}"`` for a missing required key (the eight
            fields before ``calibrator_level_db``); ``f"calibration record: unknown kind
            {kind!r}"``; from :func:`parse_time`.
        """
        required = [
            "saved_utc",
            "calibrated_utc",
            "device_id",
            "device_name",
            "channel_index",
            "sample_rate_hz",
            "kind",
            "offset_db",
        ]
        for key in required:
            if key not in data:
                raise ValueError(f"calibration record: missing {key}")
        kind = data["kind"]
        if kind not in ("calibrator", "manual"):
            raise ValueError(f"calibration record: unknown kind {kind!r}")

        def parse_time_field(name: str) -> datetime:
            val = data[name]
            if isinstance(val, datetime):
                if val.tzinfo is None:
                    raise ValueError(f"invalid time {val!r}")
                return val.astimezone(UTC)
            return parse_time(val)

        saved_utc = parse_time_field("saved_utc")
        calibrated_utc = parse_time_field("calibrated_utc")

        optional: dict[str, Any] = {
            "calibrator_level_db": None,
            "tone_frequency_hz": None,
            "tone_level_dbfs": None,
            "stability_db": None,
            "tone_to_background_db": None,
            "peak_dbfs": None,
            "ambient_correction_db": None,
            "adjustment_correction_db": None,
            "ambient_source": "",
            "adjustment_source": "",
            "note": "",
        }
        optional = {key: data.get(key, default) for key, default in optional.items()}

        return cls(
            saved_utc=saved_utc,
            calibrated_utc=calibrated_utc,
            device_id=data["device_id"],
            device_name=data["device_name"],
            channel_index=data["channel_index"],
            sample_rate_hz=data["sample_rate_hz"],
            kind=kind,
            offset_db=data["offset_db"],
            calibrator_level_db=optional["calibrator_level_db"],
            tone_frequency_hz=optional["tone_frequency_hz"],
            tone_level_dbfs=optional["tone_level_dbfs"],
            stability_db=optional["stability_db"],
            tone_to_background_db=optional["tone_to_background_db"],
            peak_dbfs=optional["peak_dbfs"],
            ambient_correction_db=optional["ambient_correction_db"],
            adjustment_correction_db=optional["adjustment_correction_db"],
            ambient_source=optional["ambient_source"],
            adjustment_source=optional["adjustment_source"],
            note=optional["note"],
        )


def format_time(t: datetime) -> str:
    """RFC 3339 UTC string as resonus-meter writes it: ``"2026-11-05T18:30:00Z"``.

    ``t`` converted to UTC; microseconds written only if non-zero (``.ffffff``).

    Raises
    ------
    ValueError
        ``"time must be timezone-aware"`` for a naive datetime.
    """
    if t.tzinfo is None or t.utcoffset() is None:
        raise ValueError("time must be timezone-aware")
    utc = t.astimezone(UTC)
    s = utc.strftime("%Y-%m-%dT%H:%M:%S")
    if utc.microsecond:
        s += f".{utc.microsecond:06d}"
    return s + "Z"


def parse_time(text: str) -> datetime:
    """Parse an RFC 3339 time (``Z`` or an offset; fractional seconds allowed) to aware UTC.

    Raises
    ------
    ValueError
        ``f"invalid time {text!r}"`` if it cannot be parsed or has no offset.
    """
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        raise ValueError(f"invalid time {text!r}") from None
    if dt.tzinfo is None:
        raise ValueError(f"invalid time {text!r}")
    return dt.astimezone(UTC)


class CalibrationStoreError(Exception):
    """The calibration history cannot be read, parsed or written."""


class CalibrationStore:
    """Append-only calibration history in one TOML file (resonus-meter ADR 0006).

    File: ``version = 1`` and an array of tables ``[[calibration]]``
    (:meth:`CalibrationRecord.to_dict`). The active record of a profile is its last
    appended record. Opening never writes; an unreadable file is never overwritten (use
    :meth:`unsaved` to continue in memory).

    Attributes
    ----------
    path : Path or None
        The history file; None for an in-memory store.
    records : list of CalibrationRecord
        In file order (oldest first).
    """

    def __init__(self, path: Path | None, records: list[CalibrationRecord]) -> None:
        self.path = path
        self.records = records

    @classmethod
    def open(cls, path: str | Path) -> CalibrationStore:
        """Read the history; a missing or blank file is an empty history (not created).

        Raises
        ------
        CalibrationStoreError
            ``f"cannot read {path}: {exc}"`` (OSError), ``f"cannot parse {path}: {exc}"``
            (TOML or record error), ``f"calibration file version {v} is newer than this
            application supports"`` for ``version > SCHEMA_VERSION``.
        """
        p = Path(path)
        try:
            text = p.read_text(encoding="utf-8")
        except FileNotFoundError:
            return cls(p, [])
        except OSError as exc:
            raise CalibrationStoreError(f"cannot read {p}: {exc}") from exc
        if not text.strip():
            return cls(p, [])
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            raise CalibrationStoreError(f"cannot parse {p}: {exc}") from exc
        version = data.get("version", 1)
        if version > SCHEMA_VERSION:
            raise CalibrationStoreError(
                f"calibration file version {version} is newer than this application supports"
            )
        records_data = data.get("calibration", [])
        records: list[CalibrationRecord] = []
        for rec_data in records_data:
            try:
                rec = CalibrationRecord.from_dict(rec_data)
            except (TypeError, ValueError) as exc:
                raise CalibrationStoreError(f"cannot parse {p}: {exc}") from exc
            records.append(rec)
        return cls(p, records)

    @classmethod
    def unsaved(cls) -> CalibrationStore:
        """Return an empty history kept in memory only (``path=None``)."""
        return cls(None, [])

    def active(self, profile: InputProfile) -> CalibrationRecord | None:
        """Return the profile's last appended record, or None."""
        for rec in reversed(self.records):
            if rec.belongs_to(profile):
                return rec
        return None

    def history(self, profile: InputProfile) -> list[CalibrationRecord]:
        """Return the profile's records, newest first."""
        return [rec for rec in reversed(self.records) if rec.belongs_to(profile)]

    def append(self, record: CalibrationRecord) -> None:
        """Append ``record`` and, with a path, rewrite the file atomically.

        The record stays in memory even if writing fails. Writing: ``{"version":
        SCHEMA_VERSION, "calibration": [r.to_dict() for r in records]}`` with
        ``tomli_w.dumps`` to ``path.with_name(path.name + ".tmp")`` (parent directories
        created), flushed and ``os.fsync``-ed, then ``os.replace`` onto ``path``.

        Raises
        ------
        CalibrationStoreError
            ``f"cannot save {path}: {exc}"`` on an OSError.
        """
        self.records.append(record)
        if self.path is None:
            return
        data = {"version": SCHEMA_VERSION, "calibration": [r.to_dict() for r in self.records]}
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            tmp.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(tomli_w.dumps(data))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except OSError as exc:
            raise CalibrationStoreError(f"cannot save {self.path}: {exc}") from exc

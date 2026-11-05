"""Impulse-response measurement orchestration: settings -> excitation -> play/record -> results.

One call runs a complete measurement on any :class:`~resonus_core.audio.AudioIO`
(sound card or simulation) and returns results in the common data model.

- Sweeps: ``repetitions`` sweeps, each followed by ``tail`` seconds of silence, are
  played back to back; the recordings are averaged synchronously and deconvolved.
- Periodic noise and MLS: the period (``duration``) is looped ``repetitions + 1`` times;
  the first period is discarded (settling), the rest averaged and deconvolved circularly.
- Channels: the io records ``input_channels`` in this order, followed by the reference
  channel if ``reference_channel > 0``, and plays ``output_channels`` (:func:`open_io`).
  ``output_mode="simultaneous"`` plays the excitation on all outputs at once (one IR per
  input); ``"sequential"`` plays it on one output after the other (one run per output,
  one IR per input and output, output-major). Each result carries
  ``metadata.extra["channel"]`` with the channel numbers and labels and the combined
  label, e.g. ``"s1m1"`` (resonus-room D-17, user decisions).
- Reference channel (``reference_channel > 0``): the recorded loopback channel is used
  as the excitation for deconvolution (removes sound-card latency and response, as REW's
  loopback timing reference); results are returned for the measurement inputs only.

Impulse responses are returned causal (t >= 0, ``ir_length`` seconds). Frequency
responses are computed from a segment that also contains ``PRE_RINGING`` seconds before
t = 0 and are phase-referenced to t = 0: band-limited deconvolution is nearly zero-phase,
and cutting at t = 0 would lose the pre-ringing (about 0.75 dB in band for a
20 Hz-20 kHz sweep at 48 kHz). The ringing of the lower band edge lasts about its
period, so 0.1 s covers band edges down to about 20 Hz (design choice; 10 ms left up to
0.4 dB error around 100-130 Hz in testing).
When the deconvolved signal is shorter (short noise periods), the pre-window is reduced to
what is available beside the impulse response.

Linear sweeps spend little time at low frequencies (Müller & Massarani 2001), so their
low-frequency accuracy is limited, and a fade-in attenuates up to
``f_start + fade_in*(f_stop - f_start)/duration`` - keep fades short for linear sweeps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from resonus_core import __version__
from resonus_core.audio import AudioIO, SounddeviceIO
from resonus_core.data import FrequencyResponse, ImpulseResponse, Metadata
from resonus_core.deconv import Deconvolved, deconvolve
from resonus_core.devices import COMMON_SAMPLE_RATES, find_device, list_devices
from resonus_core.noise import periodic_deconvolve, synchronous_average
from resonus_core.settings import Settings, param
from resonus_core.signals import Excitation, exp_sweep, linear_sweep, mls, pink_noise, white_noise

EXCITATIONS = ("exp_sweep", "sync_sweep", "linear_sweep", "white_noise", "pink_noise", "mls")
METHODS = ("auto", "inverse", "regularized", "novak")
OUTPUT_MODES = ("simultaneous", "sequential")
PERIODIC_EXCITATIONS = ("white_noise", "pink_noise", "mls")
PRE_RINGING = 0.1  # s of negative time in frequency responses: ~2 periods of a 20 Hz band edge
# Shown when input and output devices differ (resonus-room D-17: clocks may drift).
DEVICE_DRIFT_WARNING = (
    "Different input and output devices: their clocks may drift. Use one interface, "
    "or a loopback reference on the input device."
)


@dataclass(frozen=True)
class MeasurementSettings(Settings):
    """Settings of an impulse-response measurement (GUI preferences, CLI TOML).

    Channels: ``input_channels`` / ``output_channels`` are 1-based channel numbers of
    the input / output device, with one label each (``input_labels``, ``output_labels``;
    e.g. ``m1``, ``m1L``, ``s1``). ``reference_channel`` is the input channel number of
    the loopback (0 = none); it is recorded in addition to ``input_channels`` and must
    not be one of them. ``output_mode``: ``"simultaneous"`` plays the excitation on all
    outputs at once (one IR per input), ``"sequential"`` plays it on one output after the
    other (one IR per input and output).

    Relations checked after the field checks (ValueError, messages containing the quoted
    words):

    1. ``len(input_labels) == len(input_channels)`` ("input_labels") and likewise for
       outputs ("output_labels"); no label empty ("empty").
    2. ``reference_channel`` not in ``input_channels`` ("reference").
    3. ``output_mode == "sequential"`` with a reference and more than one output is not
       supported ("sequential"): the loopback is wired to one output only.
    4. Periodic excitations (``PERIODIC_EXCITATIONS``): ``round(ir_length *
       sample_rate)`` must not exceed the period ("period"): ``round(duration *
       sample_rate)`` samples for noise, ``2**ceil(log2(duration * sample_rate + 1)) -
       1`` for MLS (as :func:`make_excitation`). A longer IR would fold back (time
       aliasing; Mueller & Massarani 2001, 2.1).

    Sweeps are not limited by the excitation length; an IR longer than the silence after
    the sweep is reported by :func:`measurement_warnings`.

    References
    ----------
    S. Mueller, P. Massarani, "Transfer-function measurement with sweeps", J. Audio Eng.
    Soc. 49(6), 2001, section 2.1 and Fig. 6f.
    """

    excitation: str = param("exp_sweep", label="Excitation", choices=EXCITATIONS, group="Signal")
    sample_rate: int = param(
        48000, label="Sample rate", unit="Hz", choices=COMMON_SAMPLE_RATES, group="Device"
    )
    f_start: float = param(
        20.0, label="Start frequency", unit="Hz", minimum=0.0, maximum=96000.0, group="Signal"
    )
    f_stop: float = param(
        20000.0, label="Stop frequency", unit="Hz", minimum=1.0, maximum=192000.0, group="Signal"
    )
    duration: float = param(
        3.0,
        label="Length",
        unit="s",
        minimum=0.01,
        maximum=600.0,
        group="Signal",
        help="Sweep length, or the period of noise/MLS",
    )
    level: float = param(
        -12.0,
        label="Level",
        unit="dB re 1",
        minimum=-100.0,
        maximum=0.0,
        group="Signal",
        help="Peak level for sweeps and MLS, RMS level for noise",
    )
    fade_in: float = param(
        0.0,
        label="Fade-in",
        unit="s",
        minimum=0.0,
        maximum=60.0,
        group="Signal",
        help="Linear sweeps: attenuates up to f_start + fade_in*(f_stop-f_start)/duration",
    )
    fade_out: float = param(
        0.0, label="Fade-out", unit="s", minimum=0.0, maximum=60.0, group="Signal"
    )
    tail: float = param(
        1.0, label="Silence after sweep", unit="s", minimum=0.0, maximum=60.0, group="Signal"
    )
    repetitions: int = param(1, label="Averages", minimum=1, maximum=1000, group="Signal")
    ir_length: float = param(
        1.0,
        label="IR length",
        unit="s",
        minimum=0.001,
        maximum=600.0,
        group="Processing",
        help="Sweeps: at most the silence after the sweep; noise/MLS: at most the period",
    )
    reference_channel: int = param(
        0,
        label="Reference channel",
        minimum=0,
        maximum=256,
        group="Channels",
        help="Input channel with the loopback; 0 = none",
    )
    method: str = param("auto", label="Deconvolution", choices=METHODS, group="Processing")
    input_device: str = param(
        "", label="Input device", group="Device", help="'<host API>: <name>'; empty = default"
    )
    output_device: str = param(
        "", label="Output device", group="Device", help="'<host API>: <name>'; empty = default"
    )
    input_channels: tuple[int, ...] = param(
        (1,),
        label="Input channels",
        minimum=1,
        maximum=256,
        min_items=1,
        unique=True,
        group="Channels",
    )
    input_labels: tuple[str, ...] = param(
        ("m1",), label="Input labels", min_items=1, unique=True, group="Channels"
    )
    output_channels: tuple[int, ...] = param(
        (1,),
        label="Output channels",
        minimum=1,
        maximum=256,
        min_items=1,
        unique=True,
        group="Channels",
    )
    output_labels: tuple[str, ...] = param(
        ("s1",), label="Output labels", min_items=1, unique=True, group="Channels"
    )
    output_mode: str = param(
        "simultaneous",
        label="Several outputs",
        choices=OUTPUT_MODES,
        group="Channels",
        help="simultaneous: all at once; sequential: one after another",
    )

    def __post_init__(self) -> None:
        """Validate the fields, then the relations of the class docstring."""
        super().__post_init__()
        # 1. Labels per channel
        for kind in ("input", "output"):
            ch = getattr(self, f"{kind}_channels")
            lbl = getattr(self, f"{kind}_labels")
            if len(ch) != len(lbl):
                raise ValueError(f"{kind}_labels: one label per channel required")
            if any(label == "" for label in lbl):
                raise ValueError(f"{kind}_labels: empty label")
        # 2. Reference channel not a measurement input
        if self.reference_channel and self.reference_channel in self.input_channels:
            raise ValueError("reference_channel must not be a measurement input")
        # 3. Sequential outputs with reference not supported
        if (
            self.output_mode == "sequential"
            and self.reference_channel
            and len(self.output_channels) > 1
        ):
            raise ValueError("loopback reference with sequential outputs is not supported")
        # 4. IR length must not exceed excitation period
        if self.excitation in PERIODIC_EXCITATIONS:
            n = self.duration * self.sample_rate
            period = round(n)
            if self.excitation == "mls":
                period = 2 ** math.ceil(math.log2(n + 1)) - 1
            if round(self.ir_length * self.sample_rate) > period:
                raise ValueError("ir_length exceeds the excitation period (time aliasing)")


def measurement_warnings(settings: MeasurementSettings) -> list[str]:
    """Return warnings about settings that are valid but probably not intended.

    In this order:

    1. Sweeps (excitation not in ``PERIODIC_EXCITATIONS``) with ``ir_length > tail``:
       ``f"IR length {ir_length:g} s exceeds the silence after the sweep ({tail:g} s): the
       decay at the highest frequencies is cut after {tail:g} s (Mueller & Massarani 2001,
       2.1)"``.
    2. ``input_device`` and ``output_device`` both non-empty and different: ``"Different
       input and output devices: their clocks may drift. Use one interface, or a
       loopback reference on the input device."``
    """
    warnings: list[str] = []
    if settings.excitation not in PERIODIC_EXCITATIONS and settings.ir_length > settings.tail:
        warnings.append(
            f"IR length {settings.ir_length:g} s exceeds the silence after the sweep "
            f"({settings.tail:g} s): the decay at the highest frequencies is cut after "
            f"{settings.tail:g} s (Mueller & Massarani 2001, 2.1)"
        )
    if (
        settings.input_device
        and settings.output_device
        and settings.input_device != settings.output_device
    ):
        warnings.append(DEVICE_DRIFT_WARNING)
    return warnings


def open_io(settings: MeasurementSettings) -> SounddeviceIO:
    """Open the sound card described by the settings.

    Device keys are resolved with :func:`~resonus_core.devices.find_device` (empty: None,
    the default device; ``devices = list_devices()`` is queried once). Recorded inputs are
    ``input_channels`` followed by ``reference_channel`` if it is not 0 (the order
    :func:`measure` expects). If neither device key is set: ``SounddeviceIO(sample_rate,
    input_channels=..., output_channels=...)``; otherwise with ``input_device=`` /
    ``output_device=`` indices.

    Raises
    ------
    ValueError
        If a device key is not found.
    """
    inputs = settings.input_channels + (
        (settings.reference_channel,) if settings.reference_channel else ()
    )
    if not settings.input_device and not settings.output_device:
        return SounddeviceIO(
            settings.sample_rate,
            input_channels=inputs,
            output_channels=settings.output_channels,
        )
    devs = list_devices()
    i = find_device(settings.input_device, devs).index if settings.input_device else None
    o = find_device(settings.output_device, devs).index if settings.output_device else None
    return SounddeviceIO(
        settings.sample_rate,
        input_channels=inputs,
        output_channels=settings.output_channels,
        input_device=i,
        output_device=o,
    )


def make_excitation(settings: MeasurementSettings) -> Excitation:
    """Generate the excitation described by the settings.

    - ``exp_sweep`` / ``sync_sweep``: :func:`~resonus_core.signals.exp_sweep` with
      ``peak_dbfs=level`` (``synchronized=True`` for ``sync_sweep``);
    - ``linear_sweep``: :func:`~resonus_core.signals.linear_sweep`;
    - ``white_noise`` / ``pink_noise``: periodic noise of ``round(duration*sample_rate)``
      samples with ``rms_db=level``;
    - ``mls``: :func:`~resonus_core.signals.mls` with
      ``n_bits = ceil(log2(duration*sample_rate + 1))`` and ``peak_dbfs=level``.

    Raises
    ------
    ValueError
        If the generator rejects the parameters.
    """
    fs = settings.sample_rate
    k = settings.excitation
    if k in ("exp_sweep", "sync_sweep"):
        return exp_sweep(
            settings.f_start,
            settings.f_stop,
            settings.duration,
            fs,
            peak_dbfs=settings.level,
            fade_in=settings.fade_in,
            fade_out=settings.fade_out,
            synchronized=(k == "sync_sweep"),
        )
    if k == "linear_sweep":
        return linear_sweep(
            settings.f_start,
            settings.f_stop,
            settings.duration,
            fs,
            peak_dbfs=settings.level,
            fade_in=settings.fade_in,
            fade_out=settings.fade_out,
        )
    n = round(settings.duration * fs)
    if k == "white_noise":
        return white_noise(n, fs, rms_db=settings.level, periodic=True)
    if k == "pink_noise":
        return pink_noise(n, fs, rms_db=settings.level, periodic=True)
    if k == "mls":
        n_bits = int(np.ceil(np.log2(settings.duration * fs + 1)))
        return mls(n_bits, fs, peak_dbfs=settings.level)
    raise ValueError(f"Unsupported excitation: {k}")


@dataclass(frozen=True)
class MeasurementResult:
    """Everything a measurement produced.

    Parameters
    ----------
    settings
        Settings used.
    excitation
        Played excitation (one sweep or one period).
    recording
        Raw recording of the first run ``(n, io.n_inputs)`` (``recordings[0]``).
    deconvolved
        Deconvolution of the first run's averaged recording (measurement inputs only,
        2-D; ``deconvolutions[0]``).
    impulse_responses, frequency_responses
        One per measurement input and run, run-major: simultaneous mode ``[m1, m2, ...]``,
        sequential mode ``[s1m1, s1m2, ..., s2m1, ...]`` (reference channel excluded).
    recordings, deconvolutions
        One per run: one in simultaneous mode, one per output in sequential mode.
    """

    settings: MeasurementSettings
    excitation: Excitation
    recording: NDArray[np.float64]
    deconvolved: Deconvolved
    impulse_responses: list[ImpulseResponse]
    frequency_responses: list[FrequencyResponse]
    recordings: list[NDArray[np.float64]] = field(default_factory=list)
    deconvolutions: list[Deconvolved] = field(default_factory=list)


def measure(
    settings: MeasurementSettings,
    io: AudioIO,
    software: str = "resonus-core",
    software_version: str = __version__,
) -> MeasurementResult:
    """Run a complete measurement (see the module docstring).

    Steps:

    1. ``io.sample_rate != settings.sample_rate``: ValueError "sample rate mismatch".
       With ``n_in = len(input_channels)``, ``ref = bool(reference_channel)``, ``n_out =
       len(output_channels)``: ``io.n_inputs != n_in + ref``: ValueError ``f"io inputs:
       expected {n_in + ref}, got {io.n_inputs}"``; ``io.n_outputs != n_out``: ValueError
       ``f"io outputs: expected {n_out}, got {io.n_outputs}"``.
    2. The drive signal as before: sweeps ``np.tile(block, repetitions)`` with ``block =
       sweep + tail silence``; periodic ``np.tile(e.signal, repetitions + 1)``.
    3. Runs: simultaneous: one run, ``io.playrec(drive)`` (1-D: every output). Sequential:
       for output ``j = 0..n_out-1`` a ``(len(drive), n_out)`` zero array with column
       ``j`` = ``drive``. ``n_out == 1`` gives the same single run in both modes.
    4. Per run as before: synchronous average; measurement inputs ``avg[:, :n_in]``; the
       reference (if any) is the last column ``avg[:, n_in]``; deconvolution, linear IR,
       frequency response with ``PRE_RINGING``.
    5. Per run and input ``k``: metadata ``extra = {"settings": settings.to_dict(),
       "excitation": e.kind, "channel": {"input": input_channels[k], "input_label":
       input_labels[k], "outputs": [output channels of the run], "output_labels":
       [their labels], "label": "+".join(output_labels) + input_labels[k]}}`` (lists, not
       tuples, so they survive JSON); IR and FR of input ``k`` share it.
    6. ``MeasurementResult(settings, e, recordings[0], deconvolutions[0], irs, frs,
       recordings, deconvolutions)``.

    Parameters
    ----------
    settings
        Measurement settings.
    io
        Audio device; its sample rate must equal ``settings.sample_rate``.
    software, software_version
        Identity of the calling app, stored in the result metadata.

    Returns
    -------
    MeasurementResult

    Raises
    ------
    ValueError
        On a sample-rate mismatch, io channel counts that do not match the channel
        settings, an IR longer than the deconvolved signal, or invalid settings.
    """
    # 1. Sample rate check
    if io.sample_rate != settings.sample_rate:
        raise ValueError("sample rate mismatch")
    n_in = len(settings.input_channels)
    n_out = len(settings.output_channels)
    # the io records input_channels, then the reference (open_io): reference = last column
    ref = bool(settings.reference_channel)
    if io.n_inputs != n_in + ref:
        raise ValueError(f"io inputs: expected {n_in + ref}, got {io.n_inputs}")
    if io.n_outputs != n_out:
        raise ValueError(f"io outputs: expected {n_out}, got {io.n_outputs}")
    meas = slice(0, n_in)
    ref_index = n_in
    # 2. Create excitation
    e = make_excitation(settings)
    fs = e.sample_rate
    if e.kind in ("exp_sweep", "linear_sweep"):
        block = np.concatenate([e.signal, np.zeros(round(settings.tail * fs))])
        drive = np.tile(block, settings.repetitions)
    else:
        drive = np.tile(e.signal, settings.repetitions + 1)
    # 3. Runs
    if settings.output_mode == "simultaneous":
        runs = [list(range(n_out))]
    else:
        runs = [[j] for j in range(n_out)]
    recordings: list[NDArray[np.float64]] = []
    deconvolutions: list[Deconvolved] = []
    irs: list[ImpulseResponse] = []
    frs: list[FrequencyResponse] = []
    for run in runs:
        if settings.output_mode == "simultaneous":
            rec = io.playrec(drive)
        else:
            drive2d = np.zeros((len(drive), n_out), dtype=drive.dtype)
            drive2d[:, run[0]] = drive
            rec = io.playrec(drive2d)
        if e.kind in ("exp_sweep", "linear_sweep"):
            avg = synchronous_average(rec, block.size, n_discard=0)
            if ref:
                x = Excitation(avg[:, ref_index], fs, "reference", {})
                d = deconvolve(
                    avg[:, meas], x, method="regularized", band=(settings.f_start, settings.f_stop)
                )
            else:
                d = deconvolve(
                    avg[:, meas], e, method=None if settings.method == "auto" else settings.method
                )
        else:
            avg = synchronous_average(rec, e.n_samples, n_discard=1)
            if ref:
                x = Excitation(avg[:, ref_index], fs, "reference", {"periodic": True})
            else:
                x = e
            d = periodic_deconvolve(avg[:, meas], x)
        recordings.append(rec)
        deconvolutions.append(d)
        # 4. Extract IR and FR
        n_ir = round(settings.ir_length * fs)
        lin = d.linear_ir(n_ir)
        if lin.ndim == 1:
            lin = lin[:, None]
        pre = min(round(PRE_RINGING * fs), d.signal.shape[0] - n_ir)
        seg = np.roll(d.signal, pre - d.zero_index, axis=0)[: pre + n_ir]
        f = np.fft.rfftfreq(pre + n_ir, 1 / fs)
        H = np.fft.rfft(seg, axis=0) * np.exp(1j * 2 * np.pi * f * pre / fs)[:, None]
        if H.ndim == 1:
            H = H[:, None]
        # 5. Metadata per input
        for k in range(n_in):
            meta = Metadata(
                software,
                software_version,
                device=io.describe(),
                extra={
                    "settings": settings.to_dict(),
                    "excitation": e.kind,
                    "channel": {
                        "input": settings.input_channels[k],
                        "input_label": settings.input_labels[k],
                        "outputs": [settings.output_channels[j] for j in run],
                        "output_labels": [settings.output_labels[j] for j in run],
                        "label": "+".join(settings.output_labels[j] for j in run)
                        + settings.input_labels[k],
                    },
                },
            )
            irs.append(ImpulseResponse(lin[:, k], fs, "1", meta))
            frs.append(FrequencyResponse(f, H[:, k], "1", meta))
    return MeasurementResult(
        settings, e, recordings[0], deconvolutions[0], irs, frs, recordings, deconvolutions
    )

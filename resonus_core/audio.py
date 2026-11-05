"""Audio I/O: one multichannel protocol, a real implementation (sounddevice), a simulation.

Apps only talk to :class:`AudioIO`, so measurement code runs unchanged on a sound card,
in tests, in CLI ``--simulate`` mode and in notebooks. Signals are 2-D arrays of shape
``(n_samples, n_channels)``; channel numbers are 1-based, as printed on sound cards.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np
import scipy.signal
from numpy.typing import ArrayLike, NDArray

# Block length of playrec_blocks in samples (design choice, resonus-room D-21): 1024
# samples are 21 ms at 48 kHz, so a level meter fed from the blocks updates ~47 times/s.
STREAM_BLOCK_SIZE = 1024

BlockCallback = Callable[[NDArray[np.float64]], None]


class MeasurementAborted(Exception):
    """Raised by an :class:`AudioIO` call after :meth:`AudioIO.abort` (stopped by the user)."""


class AudioIO(Protocol):
    """Multichannel play/record interface used by all measurement apps."""

    @property
    def sample_rate(self) -> float:
        """Sample rate in Hz."""
        ...

    @property
    def n_inputs(self) -> int:
        """Number of recorded channels."""
        ...

    @property
    def n_outputs(self) -> int:
        """Number of playback channels."""
        ...

    def playrec(self, signal: ArrayLike) -> NDArray[np.float64]:
        """Play and record simultaneously.

        ``signal`` is 1-D (same signal on every output) or 2-D ``(n, n_outputs)``.
        Returns ``(n, n_inputs)``.
        """
        ...

    def rec(self, n_samples: int) -> NDArray[np.float64]:
        """Record ``n_samples`` samples; returns ``(n_samples, n_inputs)``."""
        ...

    def playrec_blocks(
        self, signal: ArrayLike, on_block: BlockCallback, block_size: int = STREAM_BLOCK_SIZE
    ) -> NDArray[np.float64]:
        """Play and record like :meth:`playrec`, passing the recording on block by block.

        ``on_block`` gets each recorded block, shape ``(k, n_inputs)`` with ``k ==
        block_size`` except for a shorter last block, in order; the blocks together are
        the returned recording ``(n, n_inputs)``. On a sound card ``on_block`` runs in the
        audio thread: it must return quickly (e.g. compute a level, hand it to the GUI
        with ``GLib.idle_add``). An exception from ``on_block`` stops playback and is
        raised by this call.
        """
        ...

    def abort(self) -> None:
        """Stop the running play/record call; it and every later call raise an exception.

        May be called from any thread, also before or between calls, and more than once.
        The running call (if any) stops as soon as possible and raises
        :class:`MeasurementAborted`, as do all later :meth:`playrec`, :meth:`rec` and
        :meth:`playrec_blocks` calls: an aborted object is used up (open a new one).
        """
        ...

    def describe(self) -> str:
        """Return a one-line description of the device for result metadata."""
        ...


def _as_output_array(signal: ArrayLike, n_outputs: int) -> NDArray[np.float64]:
    """Return ``signal`` as ``(n, n_outputs)``; a 1-D signal is played on every output.

    Raises
    ------
    ValueError
        If the signal is empty, not 1-D/2-D, or has the wrong number of columns.
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim not in (1, 2) or x.shape[0] == 0:
        raise ValueError("signal must be a non-empty 1-D or 2-D array")
    if x.ndim == 1:
        return np.tile(x[:, None], (1, n_outputs))
    if x.shape[1] != n_outputs:
        raise ValueError(f"signal must have {n_outputs} columns, got {x.shape[1]}")
    return x


def _check_channels(channels: Sequence[int], name: str) -> tuple[int, ...]:
    if not channels:
        raise ValueError(f"{name} must not be empty")
    if any(ch < 1 for ch in channels):
        raise ValueError(f"{name}: channel numbers are 1-based")
    if len(set(channels)) != len(channels):
        raise ValueError(f"{name}: channel numbers must be unique")
    return tuple(channels)


class SounddeviceIO:
    """Real sound card through `sounddevice` (PortAudio).

    Parameters
    ----------
    sample_rate
        Sample rate in Hz, > 0.
    device
        PortAudio device index or name for both directions; None uses the system default.
    input_channels, output_channels
        1-based channel numbers used for recording and playback; non-empty, unique.
    input_device, output_device
        Separate devices for recording and playback (index or name; None = default).
        If either is given, ``device`` must be None and PortAudio gets the pair
        ``(input_device, output_device)``; :meth:`rec` uses ``input_device``. Both must
        belong to the same host API; different devices may drift (no shared clock).

    Raises
    ------
    ValueError
        On invalid sample rate or channel lists, or ``device`` given together with
        ``input_device``/``output_device``.
    """

    def __init__(
        self,
        sample_rate: float,
        device: int | str | None = None,
        input_channels: Sequence[int] = (1,),
        output_channels: Sequence[int] = (1,),
        *,
        input_device: int | str | None = None,
        output_device: int | str | None = None,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        if device is not None and (input_device is not None or output_device is not None):
            raise ValueError("device cannot be set together with input_device or output_device")
        self._sample_rate = float(sample_rate)
        self._device = device
        self._input_channels = _check_channels(input_channels, "input_channels")
        self._output_channels = _check_channels(output_channels, "output_channels")
        self._pair = (
            (input_device, output_device)
            if input_device is not None or output_device is not None
            else None
        )
        self._aborted = threading.Event()

    @property
    def sample_rate(self) -> float:
        """Sample rate in Hz."""
        return self._sample_rate

    @property
    def n_inputs(self) -> int:
        """Number of recorded channels."""
        return len(self._input_channels)

    @property
    def n_outputs(self) -> int:
        """Number of playback channels."""
        return len(self._output_channels)

    def playrec(self, signal: ArrayLike) -> NDArray[np.float64]:
        """Play and record simultaneously (blocking); returns ``(n, n_inputs)``."""
        import sounddevice

        x = _as_output_array(signal, self.n_outputs)
        self._check_aborted()
        result = sounddevice.playrec(
            x,
            samplerate=self._sample_rate,
            device=self._device if self._pair is None else self._pair,
            input_mapping=list(self._input_channels),
            output_mapping=list(self._output_channels),
            dtype="float64",
            blocking=True,
        )
        self._check_aborted()
        return np.asarray(result, dtype=np.float64)

    def rec(self, n_samples: int) -> NDArray[np.float64]:
        """Record (blocking); returns ``(n_samples, n_inputs)``."""
        import sounddevice

        if n_samples <= 0:
            raise ValueError("n_samples must be > 0")
        self._check_aborted()
        result = sounddevice.rec(
            n_samples,
            samplerate=self._sample_rate,
            device=self._device if self._pair is None else self._pair[0],
            mapping=list(self._input_channels),
            dtype="float64",
            blocking=True,
        )
        self._check_aborted()
        return np.asarray(result, dtype=np.float64)

    def playrec_blocks(
        self, signal: ArrayLike, on_block: BlockCallback, block_size: int = STREAM_BLOCK_SIZE
    ) -> NDArray[np.float64]:
        """Play and record through a ``sounddevice.Stream``, block by block (blocking).

        A raw stream has no channel mapping and takes float32 samples, so: ``x =
        _as_output_array(signal, n_outputs)``; the stream opens ``channels=(max(
        input_channels), max(output_channels))``, ``dtype="float32"``, ``blocksize=
        block_size``, ``samplerate``, ``device`` as :meth:`playrec`, a ``callback`` and
        ``finished_callback`` that sets a ``threading.Event``; the call waits for it.

        ``callback(indata, outdata, frames, time, status)``, with ``i`` samples done so
        far and ``k = min(frames, n - i)``: ``outdata`` zero, then ``outdata[:k, o - 1] =
        x[i:i + k, j]`` for output channel ``o = output_channels[j]``; ``block =
        indata[:k, [c - 1 for c in input_channels]]`` as float64 (a copy); store it in the
        recording, ``on_block(block)``; raise ``sounddevice.CallbackStop`` once ``i + k ==
        n``. An exception from ``on_block`` is kept, ``sounddevice.CallbackAbort`` is
        raised, and the kept exception is raised after the stream has finished. Under- and
        overflows (``status``) are not reported.

        Raises
        ------
        ValueError
            ``"block_size must be > 0"``; from :func:`_as_output_array`.
        """
        if block_size <= 0:
            raise ValueError("block_size must be > 0")
        import sounddevice

        x = _as_output_array(signal, self.n_outputs)
        self._check_aborted()
        n = x.shape[0]
        recording = np.zeros((n, self.n_inputs), dtype=np.float64)
        pos = [0]
        in_idx = [c - 1 for c in self._input_channels]
        exc = None
        done = threading.Event()

        def callback(
            indata: NDArray[np.float32],
            outdata: NDArray[np.float32],
            frames: int,
            _time: Any,
            _status: Any,
        ) -> None:
            i = pos[0]
            k = min(frames, n - i)
            outdata[:] = 0
            if self._aborted.is_set():
                raise sounddevice.CallbackAbort
            for j, o in enumerate(self._output_channels):
                outdata[:k, o - 1] = x[i : i + k, j]
            block = indata[:k, in_idx].astype(np.float64, copy=False)
            recording[i : i + k, :] = block
            pos[0] += k
            try:
                on_block(block)
            except Exception as e:
                nonlocal exc
                exc = e
                raise sounddevice.CallbackAbort from None
            if pos[0] >= n:
                raise sounddevice.CallbackStop

        def finished() -> None:
            done.set()

        with sounddevice.Stream(
            channels=(max(self._input_channels), max(self._output_channels)),
            dtype="float32",
            blocksize=block_size,
            samplerate=self._sample_rate,
            device=self._device if self._pair is None else self._pair,
            callback=callback,
            finished_callback=finished,
        ):
            done.wait()  # closing the stream earlier would stop playback

        self._check_aborted()
        if exc:
            raise exc
        return recording

    def describe(self) -> str:
        """Return the device name(s), channel lists and sample rate.

        One device: ``f"{name}, in {inputs}, out {outputs}, {fs:g} Hz"`` (unchanged).
        Device pair: ``f"in {input name} {inputs}, out {output name} {outputs}, {fs:g}
        Hz"``. Names from ``sounddevice.query_devices(device)["name"]``, ``"default
        device"`` for None.
        """
        import sounddevice

        if self._pair is None:
            name = (
                sounddevice.query_devices(self._device)["name"]
                if self._device is not None
                else "default device"
            )
            return (
                f"{name}, in {list(self._input_channels)}, out {list(self._output_channels)}, "
                f"{self._sample_rate:g} Hz"
            )
        else:
            in_name = (
                sounddevice.query_devices(self._pair[0])["name"]
                if self._pair[0] is not None
                else "default device"
            )
            out_name = (
                sounddevice.query_devices(self._pair[1])["name"]
                if self._pair[1] is not None
                else "default device"
            )
            return (
                f"in {in_name} {list(self._input_channels)}, "
                f"out {out_name} {list(self._output_channels)}, {self._sample_rate:g} Hz"
            )

    def abort(self) -> None:
        """Stop the running call from any thread (see :meth:`AudioIO.abort`).

        ``self._aborted.set()``, then ``sounddevice.stop()``, which stops a running
        ``sounddevice.playrec``/``rec`` (they then return early). :meth:`playrec` and
        :meth:`rec` call :meth:`_check_aborted` before and after the sounddevice call;
        :meth:`playrec_blocks` calls it before opening the stream and after it finished,
        and its stream callback raises ``sounddevice.CallbackAbort`` (before writing the
        block) once ``self._aborted`` is set. An abort takes precedence over an exception
        from ``on_block``.
        """
        self._aborted.set()
        import sounddevice

        sounddevice.stop()

    def _check_aborted(self) -> None:
        """Raise ``MeasurementAborted("measurement stopped")`` if :meth:`abort` was called."""
        if self._aborted.is_set():
            raise MeasurementAborted("measurement stopped")

    @staticmethod
    def devices() -> list[dict[str, Any]]:
        """List available PortAudio devices."""
        import sounddevice

        return [dict(d) for d in sounddevice.query_devices()]


class SimulatedIO:
    """Simulated multichannel loopback through known impulse responses.

    Parameters
    ----------
    sample_rate
        Sample rate in Hz, > 0.
    impulse_responses
        None: one input, one output, unit impulse (a wire).
        1-D array: one input, one output, this impulse response.
        3-D array ``(n_inputs, n_outputs, length)``: path from each output to each input.
    latency_samples
        Extra delay between playback and recording, in samples, >= 0.
    noise_rms
        RMS of independent white noise added to each recorded channel, >= 0.
    seed
        Seed for the noise generator (reproducible results).

    Raises
    ------
    ValueError
        On invalid parameters or a 2-D ``impulse_responses`` (ambiguous).
    """

    def __init__(
        self,
        sample_rate: float,
        impulse_responses: ArrayLike | None = None,
        latency_samples: int = 0,
        noise_rms: float = 0.0,
        seed: int = 0,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        if latency_samples < 0:
            raise ValueError("latency_samples must be >= 0")
        if noise_rms < 0:
            raise ValueError("noise_rms must be >= 0")
        if impulse_responses is None:
            h = np.ones((1, 1, 1))
        else:
            h = np.asarray(impulse_responses, dtype=np.float64)
            if h.ndim == 1:
                h = h.reshape(1, 1, -1)
            elif h.ndim != 3:
                raise ValueError("impulse_responses must be None, 1-D or 3-D")
        if h.shape[2] == 0:
            raise ValueError("impulse_responses must not be empty")
        self._h = h
        self._sample_rate = float(sample_rate)
        self._latency_samples = int(latency_samples)
        self._noise_rms = float(noise_rms)
        self._rng = np.random.default_rng(seed)
        self._aborted = threading.Event()

    @property
    def sample_rate(self) -> float:
        """Sample rate in Hz."""
        return self._sample_rate

    @property
    def n_inputs(self) -> int:
        """Number of recorded channels."""
        return int(self._h.shape[0])

    @property
    def n_outputs(self) -> int:
        """Number of playback channels."""
        return int(self._h.shape[1])

    def playrec(self, signal: ArrayLike) -> NDArray[np.float64]:
        """Convolve with the simulated paths, delay, add noise; returns ``(n, n_inputs)``."""
        x = _as_output_array(signal, self.n_outputs)
        n = x.shape[0]
        self._check_aborted()
        out = np.zeros((n, self.n_inputs))
        for i in range(self.n_inputs):
            for o in range(self.n_outputs):
                out[:, i] += scipy.signal.fftconvolve(x[:, o], self._h[i, o])[:n]
        if self._latency_samples:
            out = np.concatenate([np.zeros((self._latency_samples, self.n_inputs)), out])[:n]
        self._check_aborted()
        return out + self._noise(n)

    def playrec_blocks(
        self, signal: ArrayLike, on_block: BlockCallback, block_size: int = STREAM_BLOCK_SIZE
    ) -> NDArray[np.float64]:
        """Run the simulated :meth:`playrec`, then pass its rows to ``on_block`` in blocks.

        ``recording = self.playrec(signal)``; ``on_block(recording[i:i + block_size])``
        for ``i = 0, block_size, ...`` (no real-time pacing); returns ``recording``.

        Raises
        ------
        ValueError
            ``"block_size must be > 0"``; from :meth:`playrec`.
        """
        if block_size <= 0:
            raise ValueError("block_size must be > 0")
        recording = self.playrec(signal)
        for i in range(0, recording.shape[0], block_size):
            self._check_aborted()
            on_block(recording[i : i + block_size])
        self._check_aborted()
        return recording

    def rec(self, n_samples: int) -> NDArray[np.float64]:
        """Record noise only (nothing is playing); returns ``(n_samples, n_inputs)``."""
        if n_samples <= 0:
            raise ValueError("n_samples must be > 0")
        self._check_aborted()
        return self._noise(n_samples)

    def abort(self) -> None:
        """Make the running and all later calls raise (see :meth:`AudioIO.abort`).

        ``self._aborted.set()``. :meth:`playrec` and :meth:`rec` call
        :meth:`_check_aborted` before and after the simulation, :meth:`playrec_blocks`
        before each block is passed on and before returning.
        """
        self._aborted.set()

    def _check_aborted(self) -> None:
        """Raise ``MeasurementAborted("measurement stopped")`` if :meth:`abort` was called."""
        if self._aborted.is_set():
            raise MeasurementAborted("measurement stopped")

    def describe(self) -> str:
        """Return a description of the simulation parameters."""
        return (
            f"simulated device, {self._sample_rate:g} Hz, "
            f"{self.n_inputs} in / {self.n_outputs} out, "
            f"IR {self._h.shape[2]} samples, latency {self._latency_samples} samples, "
            f"noise rms {self._noise_rms:g}"
        )

    def _noise(self, n: int) -> NDArray[np.float64]:
        if self._noise_rms > 0:
            return self._noise_rms * self._rng.standard_normal((n, self.n_inputs))
        return np.zeros((n, self.n_inputs))

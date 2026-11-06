"""SounddeviceIO on its own streams; Stop without sounddevice.stop() (resonus-room Q-C10).

resonus-room Q-C10 (user decision 2026-11-06, "fix it"): ``abort()`` used to call
``sounddevice.stop()`` from another thread while the measurement thread was blocked in
``sounddevice.playrec(..., blocking=True)``; both threads then stopped/closed the same
PortAudio stream, which could hang. Now ``playrec`` and ``rec`` run on their own
``sounddevice.Stream`` / ``InputStream`` whose callback stops on the abort flag, and
``abort()`` only sets the flag. The fake module has no ``playrec``, ``rec`` or ``stop``,
so any call to them fails the tests.
"""

import threading

import numpy as np
import pytest

import resonus_core.audio as audio
from resonus_core.audio import STREAM_FINISH_TIMEOUT, MeasurementAborted, SounddeviceIO

FS = 48000.0


def test_timeout_constant():
    assert STREAM_FINISH_TIMEOUT == 10.0


def test_playrec_runs_on_a_stream(fake_stream):
    io = SounddeviceIO(FS, device=3, input_channels=(1, 2), output_channels=(3,))
    x = np.arange(1, 11, dtype=float)
    y = io.playrec(x)
    assert fake_stream["class"] == "Stream"
    kw = fake_stream["kw"]
    assert kw["channels"] == (2, 3) and kw["dtype"] == "float32"
    assert kw["blocksize"] == audio.STREAM_BLOCK_SIZE
    assert kw["samplerate"] == FS and kw["device"] == 3
    assert y.shape == (10, 2) and y.dtype == np.float64
    np.testing.assert_array_equal(y[:, 0], np.arange(10))
    np.testing.assert_array_equal(y[:, 1], np.arange(10) + 1000)
    out = np.concatenate(fake_stream["out"])
    np.testing.assert_array_equal(out[:10, 2], x)
    assert fake_stream["stopped_by"] == "CallbackStop"


def test_playrec_broadcasts_1d_signal(fake_stream):
    io = SounddeviceIO(FS, output_channels=(1, 2))
    io.playrec(np.arange(4.0))
    out = np.concatenate(fake_stream["out"])
    np.testing.assert_array_equal(out[:4], np.column_stack([np.arange(4.0)] * 2))


def test_rec_runs_on_an_input_stream(fake_stream):
    io = SounddeviceIO(FS, device=4, input_channels=(2,))
    y = io.rec(5)
    assert fake_stream["class"] == "InputStream"
    kw = fake_stream["kw"]
    assert kw["channels"] == 2 and kw["dtype"] == "float32"
    assert kw["blocksize"] == audio.STREAM_BLOCK_SIZE
    assert kw["samplerate"] == FS and kw["device"] == 4
    assert y.shape == (5, 1) and y.dtype == np.float64
    np.testing.assert_array_equal(y[:, 0], np.arange(5) + 1000)  # input 2 = column 1
    assert fake_stream["stopped_by"] == "CallbackStop"


def test_rec_uses_the_input_device_of_a_pair(fake_stream):
    SounddeviceIO(FS, input_device=1, output_device=5).rec(3)
    assert fake_stream["kw"]["device"] == 1


def test_rec_validation(fake_stream):
    with pytest.raises(ValueError, match="n_samples must be > 0"):
        SounddeviceIO(FS).rec(0)
    assert "kw" not in fake_stream


@pytest.mark.parametrize("call", ["playrec", "rec"])
def test_abort_before_opens_no_stream(fake_stream, call):
    io = SounddeviceIO(FS)
    io.abort()  # no sounddevice.stop(): the fake module has none
    with pytest.raises(MeasurementAborted, match="measurement stopped"):
        io.playrec(np.ones(3)) if call == "playrec" else io.rec(3)
    assert "kw" not in fake_stream


@pytest.mark.parametrize("call", ["playrec", "rec"])
def test_abort_from_another_thread_ends_the_stream(fake_stream, call):
    # 1 s of audio in 1024-sample blocks of 2 ms each in the fake: the abort must end it
    # at the next block, not after the signal
    io = SounddeviceIO(FS)
    threading.Timer(0.02, io.abort).start()
    with pytest.raises(MeasurementAborted):
        io.playrec(np.ones(48000 * 10)) if call == "playrec" else io.rec(48000 * 10)
    assert fake_stream["stopped_by"] == "CallbackAbort"
    assert len(fake_stream["out"]) < 200


@pytest.mark.parametrize("call", ["playrec", "rec"])
def test_stalled_device_times_out(fake_stream, monkeypatch, call):
    # a device that never calls back: the wait ends after the signal length plus
    # STREAM_FINISH_TIMEOUT; closing the stream then finishes it
    monkeypatch.setattr(audio, "STREAM_FINISH_TIMEOUT", 0.05)
    fake_stream["stall"] = True
    io = SounddeviceIO(FS)
    with pytest.raises(RuntimeError, match="audio stream did not finish"):
        io.playrec(np.ones(48)) if call == "playrec" else io.rec(48)


def test_stalled_device_with_abort_reports_the_abort(fake_stream, monkeypatch):
    monkeypatch.setattr(audio, "STREAM_FINISH_TIMEOUT", 0.05)
    fake_stream["stall"] = True
    io = SounddeviceIO(FS)
    threading.Timer(0.01, io.abort).start()
    with pytest.raises(MeasurementAborted):
        io.playrec(np.ones(48))


def test_playrec_blocks_times_out_too(fake_stream, monkeypatch):
    monkeypatch.setattr(audio, "STREAM_FINISH_TIMEOUT", 0.05)
    fake_stream["stall"] = True
    with pytest.raises(RuntimeError, match="audio stream did not finish"):
        SounddeviceIO(FS).playrec_blocks(np.ones(48), lambda _b: None, 16)

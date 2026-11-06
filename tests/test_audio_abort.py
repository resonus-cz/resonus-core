"""AudioIO.abort: stop a running play/record call (resonus-room D-22)."""

import threading

import numpy as np
import pytest

from resonus_core.audio import MeasurementAborted, SimulatedIO, SounddeviceIO
from resonus_core.measure import MeasurementSettings, measure

FS = 48000.0


# ------------------------------------------------------------------ SimulatedIO
def test_simulated_abort_before_and_after():
    io = SimulatedIO(FS)
    np.testing.assert_array_equal(io.playrec(np.ones(4)), np.ones((4, 1)))  # not aborted
    io.abort()
    io.abort()  # idempotent
    for call in (
        lambda: io.playrec(np.ones(4)),
        lambda: io.rec(4),
        lambda: io.playrec_blocks(np.ones(4), lambda _b: None, 2),
    ):
        with pytest.raises(MeasurementAborted, match="measurement stopped"):
            call()


def test_simulated_abort_from_block_callback():
    io = SimulatedIO(FS)
    blocks = []

    def on_block(b):
        blocks.append(b)
        if len(blocks) == 2:
            io.abort()

    with pytest.raises(MeasurementAborted):
        io.playrec_blocks(np.ones(10), on_block, 2)
    assert len(blocks) == 2  # no block after the abort


def test_measure_propagates_abort():
    io = SimulatedIO(FS)
    io.abort()
    with pytest.raises(MeasurementAborted):
        measure(MeasurementSettings(), io)


def test_exception_type():
    assert issubclass(MeasurementAborted, Exception)
    assert not issubclass(MeasurementAborted, (ValueError, RuntimeError))


# ------------------------------------------------------------------ SounddeviceIO (mocked)
def test_sounddevice_stream_abort_from_other_thread(fake_stream, monkeypatch):
    io = SounddeviceIO(FS)
    first = threading.Event()
    blocks = []

    def on_block(b):
        blocks.append(b)
        first.set()

    def press_stop():
        first.wait(5)
        io.abort()

    threading.Thread(target=press_stop, daemon=True).start()
    # 2500 blocks of 2 ms each: about 5 s if the abort does not work
    with pytest.raises(MeasurementAborted):
        io.playrec_blocks(np.ones(10**4), on_block, 4)
    assert fake_stream["stopped_by"] == "CallbackAbort"
    assert 1 <= len(blocks) < 2000


def test_sounddevice_stream_abort_beats_block_error(fake_stream):
    io = SounddeviceIO(FS)

    def on_block(_b):
        io.abort()
        raise RuntimeError("meter failed")

    with pytest.raises(MeasurementAborted):
        io.playrec_blocks(np.ones(10), on_block, 4)


def test_sounddevice_stream_abort_before(fake_stream):
    io = SounddeviceIO(FS)
    io.abort()
    with pytest.raises(MeasurementAborted):
        io.playrec_blocks(np.ones(10), lambda _b: None, 4)
    assert "kw" not in fake_stream  # no stream opened

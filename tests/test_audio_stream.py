"""playrec_blocks: streaming play/record with a block callback (resonus-room D-21)."""

import numpy as np
import pytest

from resonus_core.audio import STREAM_BLOCK_SIZE, SimulatedIO, SounddeviceIO

FS = 48000.0


def test_block_size_constant():
    assert STREAM_BLOCK_SIZE == 1024


# ------------------------------------------------------------------ SimulatedIO
def test_simulated_blocks_match_playrec():
    h = np.zeros((2, 1, 4))
    h[0, 0, 0], h[1, 0, 2] = 1.0, 0.5
    x = np.random.default_rng(1).standard_normal(2500)
    blocks = []
    got = SimulatedIO(FS, h, noise_rms=1e-3, seed=7).playrec_blocks(x, blocks.append, 1000)
    want = SimulatedIO(FS, h, noise_rms=1e-3, seed=7).playrec(x)
    np.testing.assert_array_equal(got, want)
    assert [b.shape for b in blocks] == [(1000, 2), (1000, 2), (500, 2)]
    np.testing.assert_array_equal(np.concatenate(blocks), got)


def test_simulated_default_block_and_errors():
    io = SimulatedIO(FS)
    blocks = []
    io.playrec_blocks(np.ones(STREAM_BLOCK_SIZE + 1), blocks.append)
    assert [len(b) for b in blocks] == [STREAM_BLOCK_SIZE, 1]
    with pytest.raises(ValueError, match="block_size"):
        io.playrec_blocks(np.ones(10), blocks.append, 0)

    def boom(_block):
        raise RuntimeError("meter failed")

    with pytest.raises(RuntimeError, match="meter failed"):
        io.playrec_blocks(np.ones(10), boom, 4)


# ------------------------------------------------------------------ SounddeviceIO (mocked)
def test_sounddevice_stream_mapping_and_blocks(fake_stream):
    io = SounddeviceIO(FS, device=2, input_channels=(2, 4), output_channels=(3,))
    x = np.arange(1, 6, dtype=float)  # 5 samples, block size 2 -> blocks 2, 2, 1
    blocks = []
    rec = io.playrec_blocks(x, blocks.append, 2)
    kw = fake_stream["kw"]
    assert kw["channels"] == (4, 3) and kw["dtype"] == "float32"
    assert kw["blocksize"] == 2 and kw["samplerate"] == FS and kw["device"] == 2
    assert fake_stream["stopped_by"] == "CallbackStop"
    # inputs 2 and 4 are columns 1 and 3 of the stream
    np.testing.assert_array_equal(rec[:, 0], np.arange(5) + 1000)
    np.testing.assert_array_equal(rec[:, 1], np.arange(5) + 3000)
    assert rec.dtype == np.float64 and rec.shape == (5, 2)
    assert [b.shape for b in blocks] == [(2, 2), (2, 2), (1, 2)]
    np.testing.assert_array_equal(np.concatenate(blocks), rec)
    # output 3 is column 2; other columns and the rest of the last block are zero
    out = np.concatenate(fake_stream["out"])
    np.testing.assert_array_equal(out[:5, 2], x)
    assert np.all(out[:, :2] == 0) and np.all(out[5:] == 0)


def test_sounddevice_stream_device_pair(fake_stream):
    io = SounddeviceIO(FS, input_device=1, output_device=5)
    io.playrec_blocks(np.ones(3), lambda _b: None, 8)
    assert fake_stream["kw"]["device"] == (1, 5)
    assert fake_stream["kw"]["channels"] == (1, 1)


def test_sounddevice_stream_errors(fake_stream):
    io = SounddeviceIO(FS)
    with pytest.raises(ValueError, match="block_size"):
        io.playrec_blocks(np.ones(3), lambda _b: None, 0)

    def boom(_block):
        raise RuntimeError("meter failed")

    with pytest.raises(RuntimeError, match="meter failed"):
        io.playrec_blocks(np.ones(10), boom, 4)
    assert fake_stream["stopped_by"] == "CallbackAbort"

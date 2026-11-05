import numpy as np
import pytest

from resonus_core import data
from resonus_core.data import FrequencyResponse, ImpulseResponse, Metadata

META = Metadata("test", "1.0", description="pos 1", standards=("ISO 18233:2006",), extra={"k": 1})


def test_ir_roundtrip(tmp_path):
    ir = ImpulseResponse(np.array([1.0, 0.5, 0.25]), 48000, "1", META)
    data.save(tmp_path / "ir.h5", ir)
    back = data.load(tmp_path / "ir.h5")
    assert isinstance(back, ImpulseResponse)
    np.testing.assert_array_equal(back.data, ir.data)
    assert back.sample_rate == 48000 and back.metadata == META
    np.testing.assert_allclose(back.time, [0, 1 / 48000, 2 / 48000])


def test_fr_roundtrip(tmp_path):
    fr = FrequencyResponse(np.array([0.0, 100.0, 1000.0]), np.array([1, 1j, 0]), "Pa/V", META)
    data.save(tmp_path / "fr.h5", fr)
    back = data.load(tmp_path / "fr.h5")
    assert isinstance(back, FrequencyResponse)
    np.testing.assert_array_equal(back.values, fr.values)
    np.testing.assert_array_equal(back.magnitude_db()[:2], [0.0, 0.0])
    assert np.isneginf(back.magnitude_db()[2])


def test_created_timestamp_filled():
    assert Metadata("a", "1").created


def test_validation():
    with pytest.raises(ValueError):
        ImpulseResponse(np.array([]), 48000, "1", META)
    with pytest.raises(ValueError):
        ImpulseResponse(np.ones(3), 0, "1", META)
    with pytest.raises(ValueError):
        FrequencyResponse(np.array([10.0, 5.0]), np.ones(2), "1", META)
    with pytest.raises(ValueError):
        FrequencyResponse(np.array([1.0, 2.0]), np.ones(3), "1", META)


def test_foreign_file_rejected(tmp_path):
    import h5py

    with h5py.File(tmp_path / "x.h5", "w") as h5:
        h5.attrs["format"] = "other"
    with pytest.raises(ValueError, match="not a resonus"):
        data.load(tmp_path / "x.h5")

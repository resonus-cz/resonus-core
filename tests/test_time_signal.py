"""TimeSignal: a recorded signal in the data model (resonus-room Q-C9).

User decision 2026-11-06 ("add it"): a kind for recorded signals (e.g. background noise
in Pa), so apps no longer store recordings as ``ImpulseResponse`` with a role key.
"""

import h5py
import numpy as np
import pytest

from resonus_core import data
from resonus_core.data import ImpulseResponse, Metadata, TimeSignal

META = Metadata("test", "0", created="2026-11-06T10:00:00+00:00", extra={"position": "r1"})


def make(**kw):
    args = {"data": np.arange(6.0), "sample_rate": 48000.0, "unit": "Pa", "metadata": META}
    args.update(kw)
    return TimeSignal(**args)


def test_fields_and_defaults():
    s = make(data=[0, 1, 2])
    assert s.data.dtype == np.float64 and s.data.shape == (3,)
    assert s.sample_rate == 48000.0 and s.unit == "Pa" and s.start_utc == ""
    np.testing.assert_array_equal(s.time, [0.0, 1 / 48000, 2 / 48000])
    assert s.duration == pytest.approx(3 / 48000)
    assert isinstance(s.sample_rate, float)


def test_validation():
    with pytest.raises(ValueError, match="sample_rate must be positive"):
        make(sample_rate=0)
    with pytest.raises(ValueError):
        make(data=np.zeros((3, 2)))  # one channel per result, as ImpulseResponse
    with pytest.raises(ValueError, match="expected a non-empty 1-D array"):
        make(data=[])


def test_round_trip(tmp_path):
    s = make(data=np.array([0.5, -0.25, np.nan, 1e-3]), start_utc="2026-11-06T09:59:58+00:00")
    path = tmp_path / "noise.h5"
    data.save(path, s)
    back = data.load(path)
    assert isinstance(back, TimeSignal)
    np.testing.assert_array_equal(back.data, s.data)  # NaN kept
    assert back.sample_rate == 48000.0 and back.unit == "Pa"
    assert back.start_utc == "2026-11-06T09:59:58+00:00"
    assert back.metadata == META


def test_layout_readable_without_resonus_core(tmp_path):
    # docs/FILE_FORMAT.md, kind TimeSignal
    path = tmp_path / "noise.h5"
    data.save(path, make(start_utc="2026-11-06T09:59:58+00:00"))
    with h5py.File(path) as f:
        assert f.attrs["kind"] == "TimeSignal"
        assert f.attrs["unit"] == "Pa" and float(f.attrs["sample_rate"]) == 48000.0
        assert f.attrs["start_utc"] == "2026-11-06T09:59:58+00:00"
        np.testing.assert_array_equal(f["data"][()], np.arange(6.0))


def test_not_an_impulse_response(tmp_path):
    path = tmp_path / "noise.h5"
    data.save(path, make())
    assert not isinstance(data.load(path), ImpulseResponse)
    assert TimeSignal in data.Result.__args__

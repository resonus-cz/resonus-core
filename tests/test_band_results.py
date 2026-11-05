import h5py
import numpy as np
import pytest

from resonus_core import data
from resonus_core.data import BandResults, Metadata

META = Metadata("app", "1.0", description="pos 1", standards=("ISO 3382-2:2008 7.3",))


def make(**kw):
    args = dict(
        fraction=1,
        center=[125.89254117941672, 251.18864315095798, 501.18723362727224],
        labels=("125", "250", "500"),
        values={"T20": [1.2, np.nan, 0.8], "C80": [0.5, 1.0, 2.0]},
        units={"T20": "s", "C80": "dB"},
        metadata=META,
    )
    args.update(kw)
    return BandResults(**args)


def test_fields_are_normalized():
    r = make(flags=[["T20:range"], [], []])
    assert r.center.dtype == np.float64 and r.center.shape == (3,)
    assert r.values["T20"].dtype == np.float64 and np.isnan(r.values["T20"][1])
    assert r.labels == ("125", "250", "500")
    assert r.flags == (("T20:range",), (), ())
    assert r.units == {"T20": "s", "C80": "dB"}


def test_save_load_roundtrip(tmp_path):
    r = make(fraction=3, flags=(("a", "b"), (), ("c",)))
    path = tmp_path / "bands.h5"
    data.save(path, r)
    back = data.load(path)
    assert isinstance(back, BandResults)
    assert back.fraction == 3
    np.testing.assert_array_equal(back.center, r.center)
    assert back.labels == r.labels
    assert set(back.values) == {"T20", "C80"}
    for name in r.values:
        np.testing.assert_array_equal(back.values[name], r.values[name])
    assert back.units == r.units
    assert back.flags == r.flags
    assert back.metadata == r.metadata
    with h5py.File(path) as h5:
        assert h5.attrs["kind"] == "BandResults"
        assert h5.attrs["format_version"] == data.FORMAT_VERSION


def test_roundtrip_without_flags(tmp_path):
    r = make()
    data.save(tmp_path / "b.h5", r)
    assert data.load(tmp_path / "b.h5").flags == ()


def test_existing_kinds_still_roundtrip(tmp_path):
    ir = data.ImpulseResponse(np.array([1.0, 0.5]), 48000, "1", META)
    data.save(tmp_path / "ir.h5", ir)
    back = data.load(tmp_path / "ir.h5")
    assert isinstance(back, data.ImpulseResponse) and back.unit == "1"


@pytest.mark.parametrize(
    "kw",
    [
        {"fraction": 0},
        {"center": [125.0, 125.0, 500.0]},
        {"center": [0.0, 250.0, 500.0]},
        {"center": []},
        {"labels": ("125", "250")},
        {"values": {}, "units": {}},
        {"values": {"T20": [1.0, 2.0]}, "units": {"T20": "s"}},
        {"values": {"T20": [1.0, 2.0, 3.0]}, "units": {"T30": "s"}},
        {"values": {"T20": [[1.0, 2.0, 3.0]]}, "units": {"T20": "s"}},
        {"flags": ((), ())},
    ],
)
def test_invalid(kw):
    with pytest.raises(ValueError):
        make(**kw)

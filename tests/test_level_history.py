"""LevelHistory: the time-history kind of the data model (Q-C8)."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from resonus_core import data
from resonus_core.data import LevelHistory, Metadata
from resonus_core.meter_session import LOG_LEVELS, read_session

META = Metadata("test", "1.0", created="2026-11-05T12:00:00+00:00", extra={"k": 1})
FIXTURE = Path(__file__).parent / "data" / "meter_session"


def history(**kw):
    args = dict(
        time=[1.0, 2.0, 4.0],
        levels={"LAF": [40.0, np.nan, -np.inf], "LCF": [50, 51, 52]},
        units={"LAF": "dB", "LCF": "dB"},
        metadata=META,
    )
    args.update(kw)
    return LevelHistory(**args)


def test_fields_normalised():
    h = history(markers=[(0, "Start"), (2.5, 7)], start_utc="2026-11-05T12:00:00.000Z")
    assert h.time.dtype == np.float64 and h.time.tolist() == [1.0, 2.0, 4.0]
    assert list(h.levels) == ["LAF", "LCF"] and h.levels["LCF"].dtype == np.float64
    assert np.isnan(h.levels["LAF"][1]) and np.isneginf(h.levels["LAF"][2])
    assert h.markers == ((0.0, "Start"), (2.5, "7"))
    assert isinstance(h.markers, tuple) and isinstance(h.markers[0], tuple)
    assert h.start_utc == "2026-11-05T12:00:00.000Z"
    assert history().markers == () and history().start_utc == ""


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"time": [1.0, 1.0, 2.0]}, "time must be finite and strictly increasing"),
        ({"time": [1.0, np.nan, 2.0]}, "time must be finite and strictly increasing"),
        ({"time": [2.0, 1.0, 3.0]}, "time must be finite and strictly increasing"),
        ({"time": []}, "expected a non-empty 1-D array"),
        ({"levels": {}, "units": {}}, "levels must be non-empty"),
        ({"units": {"LAF": "dB"}}, "levels and units keys must match"),
        ({"levels": {"LAF": [1, 2], "LCF": [1, 2, 3]}}, "levels[LAF] length must match time"),
        ({"markers": [(np.inf, "x")]}, "marker times must be finite"),
    ],
)
def test_invalid(kw, message):
    with pytest.raises(ValueError) as info:
        history(**kw)
    assert str(info.value) == message


def test_roundtrip(tmp_path):
    h = history(markers=[(0.0, "Start"), (2.5, 'Door "1"')], start_utc="2026-11-05T12:00:00Z")
    data.save(tmp_path / "log.h5", h)
    back = data.load(tmp_path / "log.h5")
    assert isinstance(back, LevelHistory)
    np.testing.assert_array_equal(back.time, h.time)
    assert list(back.levels) == ["LAF", "LCF"]
    for name in h.levels:
        np.testing.assert_array_equal(back.levels[name], h.levels[name])
    assert back.units == h.units and back.metadata == META
    assert back.markers == h.markers and back.start_utc == h.start_utc


def test_file_layout(tmp_path):
    # docs/FILE_FORMAT.md, kind LevelHistory: readable without resonus-core
    data.save(tmp_path / "log.h5", history(markers=[(0.5, "M")], start_utc="t0"))
    with h5py.File(tmp_path / "log.h5") as f:
        assert f.attrs["format"] == "resonus" and f.attrs["format_version"] == 1
        assert f.attrs["kind"] == "LevelHistory"
        assert json.loads(f.attrs["quantities"]) == ["LAF", "LCF"]
        assert json.loads(f.attrs["units"]) == {"LAF": "dB", "LCF": "dB"}
        assert json.loads(f.attrs["markers"]) == [[0.5, "M"]]
        assert f.attrs["start_utc"] == "t0"
        assert f["time"][()].tolist() == [1.0, 2.0, 4.0]
        assert f["levels"]["LCF"][()].tolist() == [50.0, 51.0, 52.0]


def test_from_meter_session(tmp_path):
    s = read_session(FIXTURE)
    h = s.level_history()
    assert isinstance(h, LevelHistory)
    np.testing.assert_array_equal(h.time, s.log_t_s)
    assert list(h.levels) == list(LOG_LEVELS)
    np.testing.assert_array_equal(h.levels["LAF"], s.log_levels["LAF"])
    assert h.units == {name: "dB" for name in LOG_LEVELS}
    assert h.metadata == s.metadata()
    assert h.markers == ((0.0, "Start"), (3.0, 'Door: "opened" ü'), (5.0, "Pause"),
                         (7.0, "Resume"), (9.0, "Stop"))  # fmt: skip
    assert h.start_utc == "2026-11-05T20:40:42.515Z"
    data.save(tmp_path / "log.h5", h)
    assert data.load(tmp_path / "log.h5").metadata == h.metadata

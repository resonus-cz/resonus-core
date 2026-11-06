"""Reader for resonus-meter bundles (D-S1).

The fixture ``tests/data/meter_session`` was saved by resonus-meter's demo engine
(commit b49480a, 2026-11-05): 9 s of the 1 kHz demo sine, a user marker at 3 s, a pause
from 5 s to 7 s, one-third-octave band levels 25 Hz-10 kHz.
"""

import math
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from resonus_core.data import BandResults, Metadata
from resonus_core.meter_session import (
    FORMAT,
    LOG_LEVELS,
    SUPPORTED_MAJOR,
    Marker,
    MeterSession,
    combined_leq,
    parse_utc,
    parse_version,
    read_session,
)

FIXTURE = Path(__file__).parent / "data" / "meter_session"

# A format 1.0 bundle (integer version, no Fmax columns, no band levels), written by hand
# from resonus-meter docs/FILE_FORMAT.md.
V10_MD = """---
format: splmeter-session
version: 1
name: "Plant room"
start_utc: "2026-10-16T09:12:03.250Z"
end_utc: "2026-10-16T09:12:05.250Z"
duration_s: 2.000
software: "SPL Meter"
log_interval_s: 1.000
reference_pressure_uPa: 20
future_key: { a: 1 }
files:
  log: log.csv
  markers: markers.csv
  segments: segments.csv
---

# Plant room
"""
V10_LOG = "t_s,utc,marker_id,clip,LAF,LCF,LZF\n" \
    "1.000,2026-10-16T09:12:04.250Z,,0,41.05,49.00,54.90\n" \
    "2.000,2026-10-16T09:12:05.250Z,2,1,42.00,50.00,-inf\n"  # fmt: skip
V10_MARKERS = "id,t_s,utc,kind,label,note\n" \
    "1,0.000,2026-10-16T09:12:03.250Z,start,Start,\n" \
    '2,2.000,2026-10-16T09:12:05.250Z,future_kind,"AC, on","Chiller ""2"""\n'  # fmt: skip
V10_SEGMENTS = "segment,start_marker,end_marker,start_s,end_s,duration_s,clip,LAeq,LApeak," \
    "LCeq,LCpeak,LZeq,LZpeak,extra\n" \
    "1,1,2,0.000,2.000,2.000,1,41.50,60.20,49.50,64.00,55.00,66.50,7\n"  # fmt: skip


def write_v10(folder, md=V10_MD, log=V10_LOG, markers=V10_MARKERS, segments=V10_SEGMENTS):
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in [("measurement.md", md), ("log.csv", log), ("markers.csv", markers),
                       ("segments.csv", segments)]:  # fmt: skip
        (folder / name).write_text(text, encoding="utf-8")
    return folder


def fixture_copy(tmp_path):
    return Path(shutil.copytree(FIXTURE, tmp_path / "bundle"))


def edit(path, old, new):
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def test_constants():
    assert FORMAT == "splmeter-session" and SUPPORTED_MAJOR == 1
    assert LOG_LEVELS == ("LAF", "LCF", "LZF")


def test_reads_the_meter_fixture():
    s = read_session(FIXTURE)
    assert isinstance(s, MeterSession)
    assert s.path == FIXTURE and s.version == "1.2"
    assert s.name == 'Office: "north" #2'
    assert s.start_utc == datetime(2026, 11, 5, 20, 40, 42, 515000, tzinfo=UTC)
    assert s.front_matter["device"]["sample_rate_hz"] == 48000
    assert s.front_matter["files"]["rta_log"] == "rta_log.csv"
    assert "## Description" in s.body and "Line one." in s.body
    assert s.summary["A"]["Leq_db"] == -23.01 and s.summary["clip"] is False
    # log.csv: rows 1-5 s, none while paused (5-7 s), 8-9 s
    np.testing.assert_array_equal(s.log_t_s, [1, 2, 3, 4, 5, 8, 9])
    assert s.log_t_s.dtype == np.float64
    np.testing.assert_array_equal(s.log_marker, [1, 0, 0, 2, 0, 4, 0])
    assert s.log_marker.dtype == np.int64
    assert s.log_clip.dtype == np.bool_ and not s.log_clip.any()
    assert len(s.log_utc) == 7 and s.log_utc[-1] == datetime(
        2026, 11, 5, 20, 40, 51, 515000, tzinfo=UTC
    )
    assert list(s.log_levels) == list(LOG_LEVELS)
    for column in LOG_LEVELS:
        np.testing.assert_array_equal(s.log_levels[column], np.full(7, -23.01))
    assert [m.kind for m in s.markers] == ["start", "user", "pause", "resume", "stop"]
    door = s.markers[1]
    assert door == Marker(2, 3.0, datetime(2026, 11, 5, 20, 40, 45, 515000, tzinfo=UTC), "user",
                          'Door: "opened" ü', "")  # fmt: skip
    np.testing.assert_array_equal(s.segments["segment"], [1, 2, 3, 4])
    np.testing.assert_array_equal(s.segments["duration_s"], [3, 2, 2, 2])
    assert all(v.dtype == np.float64 for v in s.segments.values())
    assert "LZFmax" in s.segments and "clip" in s.segments
    assert math.isnan(s.segments["LAeq"][2])  # the pause
    assert s.bands is not None and s.bands.fraction == 3 and len(s.bands) == 27
    assert s.bands.labels[0] == "25" and s.bands.labels[16] == "1k" and s.bands.labels[-1] == "10k"
    assert s.band_log is not None and s.band_log.shape == (7, 27)
    assert s.band_segments is not None and s.band_segments.shape == (4, 27)
    assert np.isnan(s.band_segments[2]).all()
    assert s.band_log[0, 16] == -23.03 and s.band_segments[0, 16] == -23.02


def test_segments_recombine_to_the_summary():
    # resonus-meter FILE_FORMAT.md: the rounded file values recombine within ~0.01 dB
    s = read_session(FIXTURE)
    for weighting in "ACZ":
        combined = combined_leq(s.segments[f"L{weighting}eq"], s.segments["duration_s"])
        assert combined == pytest.approx(s.summary[weighting]["Leq_db"], abs=0.011)
    band = combined_leq(s.band_segments[:, 16], s.segments["duration_s"])
    assert band == pytest.approx(s.summary["rta_LZeq_db"][16], abs=0.011)


def test_band_results_summary_and_segment():
    s = read_session(FIXTURE)
    r = s.band_results()
    assert isinstance(r, BandResults)
    assert r.fraction == 3 and r.labels == tuple(s.bands.labels)
    np.testing.assert_allclose(r.center, s.bands.center)
    assert list(r.values) == ["LZeq"] and r.units == {"LZeq": "dB"}
    np.testing.assert_array_equal(r.values["LZeq"], s.summary["rta_LZeq_db"])
    assert r.metadata == s.metadata()
    first = s.band_results(segment=1)
    np.testing.assert_array_equal(first.values["LZeq"], s.band_segments[0])
    assert np.isnan(s.band_results(segment=3).values["LZeq"]).all()
    with pytest.raises(ValueError, match="^no segment 9$"):
        s.band_results(segment=9)


def test_metadata():
    m = read_session(FIXTURE).metadata()
    assert isinstance(m, Metadata)
    assert (m.software, m.software_version) == ("SPL Meter", "0.1.0")
    assert m.created == "2026-11-05T20:40:42.515Z"
    assert m.description == 'Office: "north" #2'
    assert m.device == "Demo sine source (1 kHz)"
    assert m.standards == ()
    assert m.extra["format"] == FORMAT and m.extra["version"] == "1.2"
    assert m.extra["bundle"] == "meter_session"
    assert m.extra["calibration"] == {"offset_db": 0.0, "type": "none"}
    assert m.extra["device"]["id"] == "demo.sine-1khz"
    assert m.extra["log_interval_s"] == 1.0 and m.extra["reference_pressure_uPa"] == 20
    assert m.extra["tags"] == ["demo"] and m.extra["duration_s"] == 9.0
    assert m.extra["end_utc"] == "2026-11-05T20:40:51.515Z"
    assert "author" not in m.extra and "files" not in m.extra and "rta" not in m.extra


def test_format_10_bundle(tmp_path):
    s = read_session(write_v10(tmp_path / "b"))
    assert s.version == "1.0" and s.name == "Plant room"
    assert s.bands is None and s.band_log is None and s.band_segments is None
    np.testing.assert_array_equal(s.log_marker, [0, 2])
    np.testing.assert_array_equal(s.log_clip, [False, True])
    assert s.log_levels["LZF"][1] == -np.inf
    assert s.markers[1].kind == "future_kind"  # unknown kinds kept
    assert s.markers[1].label == "AC, on" and s.markers[1].note == 'Chiller "2"'
    assert "LAFmax" not in s.segments and "extra" in s.segments
    assert s.segments["extra"][0] == 7.0  # unknown columns kept
    assert s.summary == {}
    assert s.front_matter["future_key"] == {"a": 1}
    m = s.metadata()
    assert (m.software, m.software_version) == ("SPL Meter", "")
    assert m.device == "" and "device" not in m.extra
    with pytest.raises(ValueError, match="^bundle has no band levels$"):
        s.band_results()


def test_files_are_found_through_the_front_matter(tmp_path):
    folder = fixture_copy(tmp_path)
    (folder / "data").mkdir()
    (folder / "log.csv").rename(folder / "data" / "levels.csv")
    edit(folder / "measurement.md", "  log: log.csv", "  log: data/levels.csv")
    assert read_session(str(folder)).log_t_s.size == 7


def test_combined_leq():
    assert combined_leq([60.0, 70.0], [1.0, 1.0]) == pytest.approx(67.40362689494245)
    assert combined_leq([60.0, np.nan, 70.0], [3.0, 5.0, 1.0]) == pytest.approx(65.11883360978874)
    assert combined_leq(np.array([50.0]), np.array([2.0])) == pytest.approx(50.0)
    with pytest.raises(ValueError, match="^shapes differ$"):
        combined_leq([60.0, 70.0], [1.0])
    for leq, duration in [([np.nan], [1.0]), ([], []), ([60.0], [0.0])]:
        with pytest.raises(ValueError, match="^no segment with a level$"):
            combined_leq(leq, duration)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1, (1, 0)), ("1.2", (1, 2)), ("1.0", (1, 0)), ("2.13", (2, 13))],
)
def test_parse_version(value, expected):
    assert parse_version(value) == expected


@pytest.mark.parametrize("value", [True, 2, 1.2, "1", "1.2.3", "a.b", "-1.0", "", None, "1.x"])
def test_parse_version_invalid(value):
    with pytest.raises(ValueError) as info:
        parse_version(value)
    assert str(info.value) == f"invalid version {value!r}"


def test_parse_utc():
    t = parse_utc("2026-10-16T09:12:03.250Z")
    assert t == datetime(2026, 10, 16, 9, 12, 3, 250000, tzinfo=UTC)
    assert t.utcoffset().total_seconds() == 0
    for bad in ["2026-10-16T09:12:03.250", "yesterday", ""]:
        with pytest.raises(ValueError) as info:
            parse_utc(bad)
        assert str(info.value) == f"invalid time {bad!r}"


def test_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_session(tmp_path)
    folder = fixture_copy(tmp_path)
    (folder / "segments.csv").unlink()
    with pytest.raises(FileNotFoundError):
        read_session(folder)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("---\nformat", "format", "no front matter"),
        ("  rta_segments: rta_segments.csv\n---\n", "  rta_segments: rta_segments.csv\n",
         "no front matter"),
        ("format: splmeter-session", "format: other", f"not a {FORMAT} bundle"),
        ("format: splmeter-session\n", "", f"not a {FORMAT} bundle"),
        ('version: "1.2"', 'version: "2.0"', "unsupported major version 2.0"),
        ('version: "1.2"', 'version: "x"', "invalid version 'x'"),
        ("  log: log.csv\n", "", "missing files.log"),
        ("  rta_log: rta_log.csv\n", "", "missing files.rta_log"),
        ("  log: log.csv", "  log: /etc/passwd", "file outside the bundle: /etc/passwd"),
        ("  log: log.csv", "  log: ../log.csv", "file outside the bundle: ../log.csv"),
        ("  bands_per_octave: 3", "  bands_per_octave: 2", "invalid rta block"),
        ("  bands_hz: [25, 31.5, 40,", "  bands_hz: [25, 40,", "invalid rta block"),
        ("  bands_hz: [25, 31.5,", "  bands_hz: [26, 31.5,", "invalid rta block"),
    ],
)  # fmt: skip
def test_invalid_measurement_md(tmp_path, old, new, message):
    folder = fixture_copy(tmp_path)
    edit(folder / "measurement.md", old, new)
    with pytest.raises(ValueError) as info:
        read_session(folder)
    assert str(info.value) == message


def test_front_matter_not_a_mapping(tmp_path):
    folder = fixture_copy(tmp_path)
    (folder / "measurement.md").write_text("---\n- a\n- b\n---\n", encoding="utf-8")
    with pytest.raises(ValueError, match="^front matter is not a mapping$"):
        read_session(folder)


@pytest.mark.parametrize(
    ("file", "old", "new", "message"),
    [
        ("log.csv", "t_s,utc,marker_id,clip,LAF", "t_s,utc,marker_id,clip,LAX",
         "log.csv: missing column LAF"),
        ("log.csv", "3.000,2026-11-05T20:40:45.515Z,,0,-23.01",
         "3.000,2026-11-05T20:40:45.515Z,,0,loud",
         "log.csv: line 4: could not convert string to float: 'loud'"),
        ("log.csv", "2.000,2026-11-05T20:40:44.515Z,,0", "2.000,2026-11-05T20:40:44.515,,0",
         "log.csv: line 3: invalid time '2026-11-05T20:40:44.515'"),
        ("markers.csv", "id,t_s,utc,kind,label,note", "id,t_s,utc,kind,label",
         "markers.csv: missing column note"),
        ("markers.csv", "3,5.000", "x,5.000",
         "markers.csv: line 4: invalid literal for int() with base 10: 'x'"),
        ("segments.csv", "duration_s,", "length_s,", "segments.csv: missing column duration_s"),
        ("rta_log.csv", "LZeq_1000,", "LZeq_999,", "rta_log.csv: missing column LZeq_1000"),
    ],
)  # fmt: skip
def test_invalid_csv(tmp_path, file, old, new, message):
    folder = fixture_copy(tmp_path)
    edit(folder / file, old, new)
    with pytest.raises(ValueError) as info:
        read_session(folder)
    assert str(info.value) == message


def test_row_counts_must_match(tmp_path):
    folder = fixture_copy(tmp_path)
    lines = (folder / "rta_log.csv").read_text(encoding="utf-8").splitlines(keepends=True)
    (folder / "rta_log.csv").write_text("".join(lines[:-1]), encoding="utf-8")
    with pytest.raises(ValueError, match="^rta_log.csv: 6 rows, expected 7$"):
        read_session(folder)
    folder = fixture_copy(tmp_path / "2")
    lines = (folder / "rta_segments.csv").read_text(encoding="utf-8").splitlines(keepends=True)
    (folder / "rta_segments.csv").write_text("".join(lines[:-1]), encoding="utf-8")
    with pytest.raises(ValueError, match="^rta_segments.csv: 3 rows, expected 4$"):
        read_session(folder)


def test_summary_band_levels_must_match(tmp_path):
    folder = fixture_copy(tmp_path)
    md = folder / "measurement.md"
    edit(md, "rta_LZeq_db: [-93.58, -92.58, -91.57", "rta_LZeq_db: [-93.58, null, null")
    values = read_session(folder).band_results().values["LZeq"]
    assert values.size == 27 and np.isnan(values[1:3]).all() and values[0] == -93.58
    edit(md, "rta_LZeq_db: [-93.58, ", "rta_LZeq_db: [")  # 26 values
    s = read_session(folder)  # reading does not check the summary
    with pytest.raises(ValueError, match="^summary has no rta_LZeq_db$"):
        s.band_results()
    edit(md, "rta_LZeq_db: [", "no_bands: [")
    with pytest.raises(ValueError, match="^summary has no rta_LZeq_db$"):
        read_session(folder).band_results()

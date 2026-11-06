"""Calibration records and the append-only history (D-C1, resonus-meter ADR 0006)."""

import tomllib
from datetime import UTC, datetime, timedelta, timezone

import pytest

from resonus_core.calibration import (
    SCHEMA_VERSION,
    CalibrationProposal,
    CalibrationRecord,
    CalibrationStore,
    CalibrationStoreError,
    InputProfile,
    format_time,
    parse_time,
)


def at(seconds):
    return datetime(2026, 11, 5, 18, 0, 0, tzinfo=UTC) + timedelta(seconds=seconds)


def profile(device="ALSA: UMC204HD", channel=0, name="UMC204HD 192k", rate=48000):
    return InputProfile(device, name, channel, rate)


PROPOSAL = CalibrationProposal(
    calibrator_level_db=94.0,
    measured_level_dbfs=-23.01,
    frequency_hz=1000.2,
    offset_db=117.01,
    stability_db=0.01,
    tone_to_background_db=45.0,
    peak_dbfs=-20.0,
)


def test_times():
    assert format_time(at(0)) == "2026-11-05T18:00:00Z"
    cet = timezone(timedelta(hours=1))
    assert format_time(datetime(2026, 11, 5, 19, 0, 0, 250000, tzinfo=cet)) == (
        "2026-11-05T18:00:00.250000Z"
    )
    assert parse_time("2026-11-05T18:00:00Z") == at(0)
    assert parse_time("2026-11-05T19:00:00+01:00") == at(0)
    assert parse_time("2026-11-05T18:00:00.5Z") == at(0.5)
    assert parse_time("2026-11-05T18:00:00Z").tzinfo is not None
    with pytest.raises(ValueError, match="timezone-aware"):
        format_time(datetime(2026, 11, 5))
    for bad in ("2026-11-05T18:00:00", "yesterday"):
        with pytest.raises(ValueError, match="invalid time"):
            parse_time(bad)


def test_records():
    rec = CalibrationRecord.from_proposal(profile(), PROPOSAL, at(10), note="first")
    assert rec.kind == "calibrator" and rec.saved_utc == rec.calibrated_utc == at(10)
    assert rec.offset_db == 117.01 and rec.tone_frequency_hz == 1000.2
    assert rec.tone_level_dbfs == -23.01 and rec.calibrator_level_db == 94.0
    assert rec.ambient_correction_db is None and rec.adjustment_correction_db is None
    assert rec.device_id == "ALSA: UMC204HD" and rec.channel_index == 0
    assert rec.sample_rate_hz == 48000 and rec.note == "first"
    corrected = CalibrationProposal(**{**PROPOSAL.__dict__, "ambient_correction_db": -0.1})
    rec2 = CalibrationRecord.from_proposal(
        profile(), corrected, at(11), ambient_source="pistonphone barometer, 995 hPa"
    )
    assert rec2.ambient_correction_db == -0.1
    assert rec2.ambient_source == "pistonphone barometer, 995 hPa"
    manual = CalibrationRecord.manual(profile(), 99.0, at(20))
    assert manual.kind == "manual" and manual.calibrator_level_db is None
    assert manual.offset_db == 99.0 and manual.note == ""
    again = rec.restored(at(30))
    assert again.saved_utc == at(30) and again.calibrated_utc == at(10)
    assert again.offset_db == rec.offset_db
    assert rec.belongs_to(profile(name="renamed", rate=96000))
    assert not rec.belongs_to(profile(channel=1))
    assert not rec.belongs_to(profile(device="other"))


def test_dict_round_trip_and_layout():
    rec = CalibrationRecord.from_proposal(profile(), PROPOSAL, at(10), note="n")
    d = rec.to_dict()
    assert d["saved_utc"] == "2026-11-05T18:00:10Z" and d["kind"] == "calibrator"
    assert "ambient_correction_db" not in d and "ambient_source" not in d
    assert d["note"] == "n"
    assert list(d)[:8] == [
        "saved_utc", "calibrated_utc", "device_id", "device_name", "channel_index",
        "sample_rate_hz", "kind", "offset_db",
    ]  # fmt: skip
    assert CalibrationRecord.from_dict(d) == rec
    manual = CalibrationRecord.manual(profile(), 99.0, at(20))
    dm = manual.to_dict()
    assert "calibrator_level_db" not in dm and dm["note"] == ""
    assert CalibrationRecord.from_dict(dm) == manual
    assert CalibrationRecord.from_dict({**dm, "future_key": 1}) == manual
    with pytest.raises(ValueError, match="missing offset_db"):
        CalibrationRecord.from_dict({k: v for k, v in dm.items() if k != "offset_db"})
    with pytest.raises(ValueError, match="unknown kind 'guess'"):
        CalibrationRecord.from_dict({**dm, "kind": "guess"})


def test_missing_file_is_empty_and_not_created(tmp_path):
    path = tmp_path / "calibrations.toml"
    store = CalibrationStore.open(path)
    assert store.records == [] and store.path == path
    assert store.active(profile()) is None
    assert not path.exists()
    (tmp_path / "blank.toml").write_text("  \n", encoding="utf-8")
    assert CalibrationStore.open(tmp_path / "blank.toml").records == []


def test_append_persists_and_latest_per_profile_is_active(tmp_path):
    path = tmp_path / "sub" / "calibrations.toml"
    store = CalibrationStore.open(path)
    store.append(CalibrationRecord.from_proposal(profile(), PROPOSAL, at(10), note="first"))
    store.append(CalibrationRecord.manual(profile(channel=1), 101.0, at(20)))
    store.append(CalibrationRecord.manual(profile(), 118.5, at(30)))
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == SCHEMA_VERSION == 1 and len(data["calibration"]) == 3
    assert not path.with_name(path.name + ".tmp").exists()
    reopened = CalibrationStore.open(path)
    active = reopened.active(profile())
    assert active.kind == "manual" and active.offset_db == 118.5
    history = reopened.history(profile())
    assert [r.saved_utc for r in history] == [at(30), at(10)]
    assert history[1].note == "first" and history[1].tone_frequency_hz == 1000.2
    assert reopened.active(profile(channel=1)).offset_db == 101.0
    assert reopened.active(profile(device="other")) is None


def test_restore_appends_a_copy(tmp_path):
    store = CalibrationStore.open(tmp_path / "c.toml")
    first = CalibrationRecord.from_proposal(profile(), PROPOSAL, at(10))
    store.append(first)
    store.append(CalibrationRecord.manual(profile(), 99.0, at(20)))
    store.append(first.restored(at(30)))
    active = CalibrationStore.open(tmp_path / "c.toml").active(profile())
    assert active.kind == "calibrator" and active.saved_utc == at(30)
    assert active.calibrated_utc == at(10) and len(store.records) == 3


def test_newer_version_rejected(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("version = 2\n", encoding="utf-8")
    with pytest.raises(CalibrationStoreError, match="version 2 is newer"):
        CalibrationStore.open(path)


def test_unreadable_file_is_reported_and_left_untouched(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("this is = = not toml", encoding="utf-8")
    with pytest.raises(CalibrationStoreError, match="cannot parse"):
        CalibrationStore.open(path)
    assert path.read_text(encoding="utf-8") == "this is = = not toml"
    store = CalibrationStore.unsaved()
    assert store.path is None
    store.append(CalibrationRecord.manual(profile(), 99.0, at(1)))  # memory only
    assert store.active(profile()).offset_db == 99.0
    assert path.read_text(encoding="utf-8") == "this is = = not toml"


def test_write_failure_keeps_the_record(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    store = CalibrationStore(blocker / "c.toml", [])  # parent is a file: cannot write
    with pytest.raises(CalibrationStoreError, match="cannot save"):
        store.append(CalibrationRecord.manual(profile(), 99.0, at(1)))
    assert store.active(profile()).offset_db == 99.0


METER_FILE = """version = 1

[[calibration]]
saved_utc = "2026-11-05T18:00:10Z"
calibrated_utc = "2026-11-05T18:00:10Z"
device_id = "alsa_input.usb-BEHRINGER_UMC204HD"
device_name = "UMC204HD 192k"
channel_index = 0
sample_rate_hz = 48000
kind = "calibrator"
offset_db = 117.01
calibrator_level_db = 94.0
tone_frequency_hz = 1000.2
tone_level_dbfs = -23.01
stability_db = 0.01
tone_to_background_db = 45.0
peak_dbfs = -20.0
note = "meter"

[[calibration]]
saved_utc = "2026-11-05T18:00:20.5Z"
calibrated_utc = "2026-11-05T18:00:20.5Z"
device_id = "alsa_input.usb-BEHRINGER_UMC204HD"
device_name = "UMC204HD 192k"
channel_index = 0
sample_rate_hz = 48000
kind = "manual"
offset_db = 117.2
note = ""
"""


def test_reads_a_file_written_by_resonus_meter(tmp_path):
    path = tmp_path / "calibrations.toml"
    path.write_text(METER_FILE, encoding="utf-8")
    store = CalibrationStore.open(path)
    p = profile(device="alsa_input.usb-BEHRINGER_UMC204HD")
    assert [r.kind for r in store.history(p)] == ["manual", "calibrator"]
    assert store.active(p).saved_utc == at(20.5)
    assert store.history(p)[1].peak_dbfs == -20.0 and store.history(p)[1].note == "meter"
    # what we write reads back the same (and keeps the meter's fields)
    store.append(store.records[0].restored(at(40)))
    again = CalibrationStore.open(path)
    assert again.records[:2] == store.records[:2] and len(again.records) == 3

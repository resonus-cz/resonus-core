"""Calibration proposal from a calibrator capture (D-C1, resonus-meter method)."""

import math

import numpy as np
import pytest

from resonus_core.calibration import (
    CAPTURE_DURATION,
    CLIP_LEVEL,
    FAILURE_MESSAGES,
    FREQUENCY_TOLERANCE,
    MAX_STABILITY_DB,
    MIN_LEVEL_DBFS,
    MIN_TONE_TO_BACKGROUND_DB,
    P0,
    CalibrationError,
    CalibrationProposal,
    find_tone_frequency,
    hann_power,
    lock_in_power,
    propose,
    to_pascal,
)

FS = 48000.0
# 20 lg(0.1 / sqrt(2)): mean-square level of a sine of amplitude 0.1, dB re full scale
SINE_01_DB = 20 * math.log10(0.1 / math.sqrt(2))


def sine(f, seconds, amp=0.1, fs=FS, phase=0.0):
    n = np.arange(round(seconds * fs))
    return amp * np.sin(2 * np.pi * f * n / fs + phase)


def test_constants():
    assert P0 == 20e-6  # ISO 1683:2015
    assert CAPTURE_DURATION == 8.0 and FREQUENCY_TOLERANCE == 0.03
    assert MIN_LEVEL_DBFS == -60.0 and MIN_TONE_TO_BACKGROUND_DB == 20.0
    assert MAX_STABILITY_DB == 0.1 and CLIP_LEVEL == 1.0 - 2.0**-15


def test_meter_reference_values():
    # resonus-meter calibration.rs test: 2 s of 0.1 sin at 1 kHz, 94 dB calibrator
    x = sine(1000.0, 2.0).astype(np.float32)
    p = propose(x, FS, 1000.0, 94.0)
    assert isinstance(p, CalibrationProposal)
    assert p.measured_level_dbfs == pytest.approx(-23.010_299_956_64, abs=1e-6)
    assert p.offset_db == pytest.approx(117.010_299_956_64, abs=1e-6)
    assert p.frequency_hz == pytest.approx(1000.0, abs=0.01)
    assert p.tone_to_background_db > 100.0
    assert p.calibrator_level_db == 94.0
    assert p.ambient_correction_db == 0.0 and p.adjustment_correction_db == 0.0
    assert p.peak_dbfs == pytest.approx(-20.0, abs=1e-3)
    assert p.stability_db < 1e-6


def test_refines_an_off_nominal_tone():
    # resonus-meter test: 1007.3 Hz, 8 s
    p = propose(sine(1007.3, 8.0), FS, 1000.0, 94.0)
    assert p.frequency_hz == pytest.approx(1007.3, abs=0.1)
    assert p.measured_level_dbfs == pytest.approx(-23.0103, abs=0.02)


def test_accuracy_over_the_search_range():
    # review 2026-11-05: level error below 0.005 dB within ±3 % of 1 kHz, any phase
    rng = np.random.default_rng(0)
    for f0 in np.linspace(975.0, 1025.0, 11):
        p = propose(sine(f0, 8.0, phase=rng.uniform(0, 2 * np.pi)), FS, 1000.0, 94.0)
        assert p.measured_level_dbfs == pytest.approx(SINE_01_DB, abs=0.005)
        assert p.frequency_hz == pytest.approx(f0, abs=0.05)


def test_pistonphone_with_corrections():
    # 250 Hz pistonphone at 124 dB; ambient (barometric) and adjustment corrections add
    p = propose(sine(250.0, 8.0), FS, 250.0, 124.0, ambient_correction_db=-0.12,
                adjustment_correction_db=0.05)  # fmt: skip
    assert p.offset_db == pytest.approx(124.0 - 0.12 + 0.05 - SINE_01_DB, abs=1e-4)
    assert p.ambient_correction_db == -0.12 and p.adjustment_correction_db == 0.05


def test_noise_and_hum_are_rejected_by_the_lock_in():
    # meter spec 9.1 item 7: a 94 dB tone plus noise and hum gives C within ±0.02 dB
    rng = np.random.default_rng(1)
    x = sine(1000.0, 8.0) + sine(50.0, 8.0, amp=0.005) + 0.003 * rng.standard_normal(384000)
    p = propose(x, FS, 1000.0, 94.0)
    assert p.offset_db == pytest.approx(94.0 - SINE_01_DB, abs=0.02)
    assert MIN_TONE_TO_BACKGROUND_DB < p.tone_to_background_db < 40.0


def test_other_sample_rates():
    for fs in (44100.0, 96000.0):
        p = propose(sine(1000.0, 3.0, fs=fs), fs, 1000.0, 94.0)
        assert p.measured_level_dbfs == pytest.approx(SINE_01_DB, abs=0.005)


def reason(x, fs=FS, f=1000.0):
    with pytest.raises(CalibrationError) as info:
        propose(x, fs, f, 94.0)
    assert str(info.value) == FAILURE_MESSAGES[info.value.reason]
    assert isinstance(info.value, ValueError)
    return info.value.reason


def test_failures():
    assert reason(np.zeros(0)) == "empty"
    assert reason(sine(1000.0, 2.0), f=0.0) == "invalid_frequency"
    assert reason(sine(1000.0, 2.0), f=24000.0) == "invalid_frequency"
    assert reason(np.zeros(48000)) == "too_quiet"
    assert reason(sine(1000.0, 2.0, amp=1e-4)) == "too_quiet"  # -83 dB FS
    clipped = sine(1000.0, 2.0)
    clipped[100] = -32767 / 32768  # largest negative 16-bit code but one: clipped
    assert reason(clipped) == "clipped"
    below = sine(1000.0, 2.0)
    below[100] = 32766 / 32768  # one code below: not clipped
    assert propose(below, FS, 1000.0, 94.0).peak_dbfs == pytest.approx(
        20 * math.log10(32766 / 32768)
    )
    assert reason(sine(1000.0, 0.5)) == "empty"  # less than one second
    # resonus-meter test: alternating +-0.1 swamps a 0.01 tone
    n = np.arange(48000)
    noisy = 0.01 * np.sin(2 * np.pi * n / 48.0) + np.where(n % 2 == 0, 0.1, -0.1)
    assert reason(noisy) == "noisy"
    # level changes by 1 dB after 4 s
    steps = np.concatenate([sine(1000.0, 4.0), sine(1000.0, 4.0, amp=0.1 * 10 ** (1 / 20))])
    assert reason(steps) == "unstable"
    # a 10 Hz target at 48 kHz has a single 1-Hz bin in its search range
    assert reason(sine(10.0, 2.0), f=10.0) == "no_tone"


def test_failure_order_clip_before_level():
    x = np.zeros(48000)
    x[0] = 1.0
    assert reason(x) == "clipped"


def test_lock_in_and_hann_power():
    x = sine(1000.0, 1.0, amp=0.5)
    assert lock_in_power(x, FS, 1000.0) == pytest.approx(0.125, rel=1e-9)
    assert lock_in_power(x, FS, 1100.0) < 1e-6
    assert hann_power(x, FS, 1000.0) > 1e3 * hann_power(x, FS, 1010.0)
    with pytest.raises(ValueError, match="empty signal"):
        lock_in_power([], FS, 1000.0)
    with pytest.raises(ValueError, match="at least 2 samples"):
        hann_power([1.0], FS, 1000.0)


def test_find_tone_frequency():
    assert find_tone_frequency(sine(993.6, 2.0), FS, 1000.0) == pytest.approx(993.6, abs=0.05)
    # at the edge of the range the bin itself is returned
    assert find_tone_frequency(sine(970.0, 2.0), FS, 1000.0) == 970.0


def test_to_pascal():
    p = propose(sine(1000.0, 2.0), FS, 1000.0, 94.0)
    pa = to_pascal(sine(1000.0, 2.0), p.offset_db)
    assert pa.dtype == np.float64
    level = 20 * np.log10(np.sqrt(np.mean(pa**2)) / P0)
    assert level == pytest.approx(94.0, abs=1e-6)
    np.testing.assert_allclose(to_pascal(np.ones((2, 3)), 0.0), np.full((2, 3), P0))

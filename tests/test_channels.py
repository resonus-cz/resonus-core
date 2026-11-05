import pytest

from resonus_core.channels import (
    INPUT_PREFIX,
    OUTPUT_PREFIX,
    add_row,
    device_options,
    next_channel,
    next_label,
    pick_sample_rate,
    remove_row,
)
from resonus_core.devices import DeviceInfo

DEVICES = [
    DeviceInfo(0, "UR22 (hw:0,0)", "ALSA", 2, 2, 44100.0),
    DeviceInfo(1, "HDMI", "ALSA", 0, 8, 48000.0),
    DeviceInfo(2, "Mic", "ALSA", 1, 0, 48000.0),
]


def test_prefixes():
    # naming convention s#m# (user decision D-17)
    assert (INPUT_PREFIX, OUTPUT_PREFIX) == ("m", "s")


def test_device_options():
    assert [d.name for d in device_options(DEVICES, "input")] == ["UR22 (hw:0,0)", "Mic"]
    assert [d.name for d in device_options(DEVICES, "output")] == ["UR22 (hw:0,0)", "HDMI"]
    assert device_options([], "input") == []
    with pytest.raises(ValueError, match="direction"):
        device_options(DEVICES, "both")


def test_next_channel():
    assert next_channel([], 2) == 1
    assert next_channel([1], 2) == 2
    assert next_channel([2], 4) == 1
    assert next_channel([1, 3], 4) == 2
    for used, n in (([1, 2], 2), ([], 0)):
        with pytest.raises(ValueError, match="no free channel"):
            next_channel(used, n)


@pytest.mark.parametrize(
    ("labels", "prefix", "expected"),
    [
        ([], "m", "m1"),
        (["m1"], "m", "m2"),
        (["m1", "m3R"], "m", "m4"),
        (["m1L", "m1R"], "m", "m2"),  # dummy head: both ears of position 1
        (["m10", "m2"], "m", "m11"),
        (["left", "x5"], "m", "m1"),  # no numbered label with the prefix
        (["s1", "s2"], "s", "s3"),
        (["mL"], "m", "m1"),
    ],
)
def test_next_label(labels, prefix, expected):
    assert next_label(labels, prefix) == expected


def test_add_row():
    assert add_row((1,), ("m1",), "m", 2) == ((1, 2), ("m1", "m2"))
    assert add_row([2], ["s1"], "s", 2) == ((2, 1), ("s1", "s2"))
    with pytest.raises(ValueError, match="no free channel"):
        add_row((1, 2), ("m1", "m2"), "m", 2)


def test_remove_row():
    assert remove_row((1, 2, 3), ("m1", "m2", "m3"), 1) == ((1, 3), ("m1", "m3"))
    assert remove_row([1, 2], ["m1", "m2"], 1) == ((1,), ("m1",))
    with pytest.raises(ValueError, match="first row"):
        remove_row((1, 2), ("m1", "m2"), 0)
    for index in (2, -1):
        with pytest.raises(ValueError, match="no such row"):
            remove_row((1, 2), ("m1", "m2"), index)


def test_pick_sample_rate():
    rates = (44100, 48000, 96000)
    assert pick_sample_rate(96000, 44100.0, rates) == 44100  # device default first
    assert pick_sample_rate(96000, 22050.0, rates) == 96000  # else keep the current rate
    assert pick_sample_rate(192000, 22050.0, rates) == 44100  # else the first supported
    assert pick_sample_rate(48000, 47999.9999, rates) == 48000
    assert isinstance(pick_sample_rate(48000, 48000.0, rates), int)
    with pytest.raises(ValueError, match="no supported"):
        pick_sample_rate(48000, 48000.0, ())

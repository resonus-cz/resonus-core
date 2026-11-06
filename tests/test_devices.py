import sys
import types

import numpy as np
import pytest

from resonus_core.audio import SounddeviceIO
from resonus_core.devices import (
    COMMON_SAMPLE_RATES,
    DeviceInfo,
    find_device,
    list_devices,
    supported_sample_rates,
)

FS = 48000

DEVICES = [
    # name, hostapi, inputs, outputs, default rate, supported rates (None = all)
    ("UR22 (hw:0,0)", 0, 2, 2, 44100.0, {44100, 48000}),
    ("pipewire", 0, 128, 128, 44100.0, None),
    ("UR22 Analog", 1, 2, 2, 48000.0, None),
    ("HDMI", 0, 0, 8, 44100.0, None),
]
HOSTAPIS = ({"name": "ALSA"}, {"name": "JACK Audio Connection Kit"})
DEFAULT = 1  # index of the default device


class PortAudioError(Exception):
    pass


@pytest.fixture
def fake_sd(monkeypatch, recording_streams):
    calls = {}

    def query_devices(device=None):
        rows = [
            {
                "name": n,
                "index": i,
                "hostapi": h,
                "max_input_channels": ni,
                "max_output_channels": no,
                "default_samplerate": fs,
            }
            for i, (n, h, ni, no, fs, _) in enumerate(DEVICES)
        ]
        return rows if device is None else rows[device]

    def check(kind):
        def checker(device=None, channels=None, samplerate=None, **_kw):
            i = DEFAULT if device is None else device
            _, _, ni, no, _, rates = DEVICES[i]
            limit = ni if kind == "input" else no
            if channels > limit:
                raise PortAudioError("Invalid number of channels")
            if rates is not None and samplerate not in rates:
                raise PortAudioError("Invalid sample rate")

        return checker

    mod = types.SimpleNamespace(
        **recording_streams(calls),
        query_devices=query_devices,
        query_hostapis=lambda: HOSTAPIS,
        check_input_settings=check("input"),
        check_output_settings=check("output"),
        PortAudioError=PortAudioError,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", mod)
    return calls


def test_common_sample_rates():
    # user decision D-17 (resonus-room D-17)
    assert COMMON_SAMPLE_RATES == (44100, 48000, 88200, 96000, 176400, 192000)


def test_list_devices(fake_sd):
    devs = list_devices()
    assert [d.index for d in devs] == [0, 1, 2, 3]
    jack = devs[2]
    assert isinstance(jack, DeviceInfo)
    assert jack.name == "UR22 Analog" and jack.host_api == "JACK Audio Connection Kit"
    assert (jack.max_inputs, jack.max_outputs, jack.default_sample_rate) == (2, 2, 48000.0)
    assert jack.key == "JACK Audio Connection Kit: UR22 Analog"
    assert jack.label == "UR22 Analog (JACK Audio Connection Kit, 2 in / 2 out)"
    assert devs[1].key == "ALSA: pipewire"


def test_find_device(fake_sd):
    assert find_device("JACK Audio Connection Kit: UR22 Analog").index == 2
    assert find_device("ALSA: HDMI", list_devices()).max_outputs == 8
    with pytest.raises(ValueError, match="not found"):
        find_device("ALSA: UR22 Analog")  # right name, wrong host API


def test_supported_sample_rates(fake_sd):
    assert supported_sample_rates(0, 0, 2, 2) == (44100, 48000)
    assert supported_sample_rates(1, 1, 4, 4) == COMMON_SAMPLE_RATES
    assert supported_sample_rates(0, 0, 3, 1) == ()  # channel 3 does not exist on the UR22
    assert supported_sample_rates(None, None, 1, 1) == COMMON_SAMPLE_RATES  # default device
    assert supported_sample_rates(1, 0, 1, 1) == (44100, 48000)  # both devices must accept
    assert supported_sample_rates(1, 1, 1, 1, rates=(48000, 96000)) == (48000, 96000)


def test_sounddevice_io_device_pair(fake_sd):
    io = SounddeviceIO(FS, input_device=2, output_device=0, input_channels=(1, 2))
    io.playrec(np.ones(8))
    assert fake_sd["playrec"]["device"] == (2, 0)
    io.rec(4)
    assert fake_sd["rec"]["device"] == 2
    text = io.describe()
    assert "in UR22 Analog [1, 2]" in text and "out UR22 (hw:0,0) [1]" in text
    io = SounddeviceIO(FS, input_device=2)
    io.playrec(np.ones(8))
    assert fake_sd["playrec"]["device"] == (2, None)
    assert "out default device [1]" in io.describe()


def test_sounddevice_io_single_device_unchanged(fake_sd):
    io = SounddeviceIO(FS, device=3)
    io.playrec(np.ones(8))
    assert fake_sd["playrec"]["device"] == 3
    with pytest.raises(ValueError):
        SounddeviceIO(FS, device=1, input_device=2)


def test_supported_sample_rates_checks_default_device(fake_sd, monkeypatch):
    # None means PortAudio's default device; it must be checked as well
    monkeypatch.setattr(sys.modules[__name__], "DEFAULT", 0)  # default = UR22 (44.1/48 kHz)
    assert supported_sample_rates(None, None, 1, 1) == (44100, 48000)
    assert supported_sample_rates(1, None, 1, 1) == (44100, 48000)


def test_default_device(fake_sd, monkeypatch):
    from resonus_core.devices import default_device

    mod = sys.modules["sounddevice"]
    plain = mod.query_devices

    def query_devices(device=None, kind=None):
        if kind is None:
            return plain(device)
        if kind == "input":
            return {"index": 2, "name": "UR22 Analog"}
        raise PortAudioError("no default output device")

    monkeypatch.setattr(mod, "query_devices", query_devices)
    assert default_device("input").key == "JACK Audio Connection Kit: UR22 Analog"
    assert default_device("output") is None
    assert default_device("input", list_devices()[:2]) is None  # index not in the list
    with pytest.raises(ValueError, match="direction"):
        default_device("both")

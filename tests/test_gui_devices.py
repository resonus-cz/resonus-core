"""Preferences group "Device": device menus, sample rate, drift warning."""

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gtk  # noqa: E402

if not Gtk.init_check():
    pytest.skip("no display available", allow_module_level=True)

from resonus_core.devices import DeviceInfo  # noqa: E402
from resonus_core.gui.devices import DEFAULT_DEVICE_LABEL, DeviceGroup  # noqa: E402
from resonus_core.measure import DEVICE_DRIFT_WARNING, MeasurementSettings  # noqa: E402

Adw.init()

DEVICES = [
    DeviceInfo(0, "UR22 (hw:0,0)", "ALSA", 2, 2, 44100.0),
    DeviceInfo(1, "HDMI", "ALSA", 0, 8, 48000.0),
    DeviceInfo(2, "Mic", "ALSA", 1, 0, 96000.0),
]
UR22 = "ALSA: UR22 (hw:0,0)"
RATES = {0: (44100, 48000), 1: (48000, 96000), 2: (48000, 96000), None: (44100, 48000, 96000)}


class Fixture:
    def __init__(self, settings=None, **kw):
        self.store = settings or MeasurementSettings()
        self.changes = []
        self.errors = []
        self.calls = []
        self.dg = DeviceGroup(
            self.store,
            self.on_change,
            lambda: self.store,
            devices=DEVICES,
            rates=self.rates,
            on_error=self.errors.append,
            **kw,
        )

    def on_change(self, new):
        self.store = new
        self.changes.append(new)

    def rates(self, i, o, n_in, n_out):
        self.calls.append((i, o, n_in, n_out))
        a, b = set(RATES[i]), set(RATES[o])
        return tuple(r for r in (44100, 48000, 96000) if r in a and r in b)


def items(row):
    model = row.get_model()
    return [model.get_string(k) for k in range(model.get_n_items())]


def selected(row):
    return items(row)[row.get_selected()]


def test_rows_and_menus():
    f = Fixture()
    dg = f.dg
    assert isinstance(dg.group, Adw.PreferencesGroup) and dg.group.get_title() == "Device"
    assert items(dg.input_row) == [
        DEFAULT_DEVICE_LABEL,
        "UR22 (hw:0,0) (ALSA, 2 in / 2 out)",
        "Mic (ALSA, 1 in / 0 out)",
    ]
    assert items(dg.output_row) == [
        DEFAULT_DEVICE_LABEL,
        "UR22 (hw:0,0) (ALSA, 2 in / 2 out)",
        "HDMI (ALSA, 0 in / 8 out)",
    ]
    assert dg.input_row.get_selected() == dg.output_row.get_selected() == 0
    assert items(dg.rate_row) == ["44100", "48000", "96000"] and selected(dg.rate_row) == "48000"
    assert dg.warning_row.get_title() == DEVICE_DRIFT_WARNING
    assert not dg.warning_row.get_visible()
    assert f.changes == []  # building the group changes nothing


def test_device_selection_follows_device_default_rate():
    f = Fixture()
    f.dg.input_row.set_selected(1)  # UR22, default 44.1 kHz
    assert len(f.changes) == 1
    assert f.store.input_device == UR22 and f.store.sample_rate == 44100
    assert items(f.dg.rate_row) == ["44100", "48000"] and selected(f.dg.rate_row) == "44100"
    f.dg.output_row.set_selected(1)  # UR22 as well: same device, rate kept
    assert f.store.output_device == UR22 and f.store.sample_rate == 44100
    assert not f.dg.warning_row.get_visible()
    assert len(f.changes) == 2


def test_different_devices_warn_and_rate_falls_back():
    f = Fixture()
    f.dg.input_row.set_selected(1)  # UR22 -> 44100
    f.dg.output_row.set_selected(2)  # HDMI: default 48000 accepted by both
    assert f.store.output_device == "ALSA: HDMI" and f.store.sample_rate == 48000
    assert items(f.dg.rate_row) == ["48000"]
    assert f.dg.warning_row.get_visible()
    f.dg.output_row.set_selected(0)  # default device: key "", warning gone
    assert f.store.output_device == "" and not f.dg.warning_row.get_visible()
    assert f.store.sample_rate == 48000  # default entry: current rate kept if accepted


def test_rate_selection():
    f = Fixture()
    f.dg.rate_row.set_selected(2)
    assert f.changes == [MeasurementSettings(sample_rate=96000)]


def test_rates_called_with_indices_and_highest_channels():
    s = MeasurementSettings(
        input_device=UR22,
        input_channels=(2, 1),
        input_labels=("m1", "m2"),
        reference_channel=4,
        output_channels=(3,),
    )
    f = Fixture(s)
    assert f.calls[-1] == (0, None, 4, 3)


def test_unknown_device_key_kept():
    s = MeasurementSettings(input_device="JACK: gone", sample_rate=44100)
    f = Fixture(s)
    assert selected(f.dg.input_row) == "JACK: gone (not found)"
    assert f.calls[-1][0] is None
    assert f.changes == []
    f.dg.input_row.set_selected(1)
    assert f.store.input_device == UR22
    f.dg.input_row.set_selected(3)  # back to the not-found entry: key restored
    assert f.store.input_device == "JACK: gone"


def test_rate_not_accepted_is_shown_alone():
    # building the page never changes the settings
    f = Fixture(MeasurementSettings(sample_rate=192000))
    assert items(f.dg.rate_row) == ["192000"] and f.changes == []


def test_uses_current_settings():
    f = Fixture()
    f.store = f.store.replace(repetitions=5)  # changed on another page
    f.dg.rate_row.set_selected(0)
    assert f.store.repetitions == 5 and f.store.sample_rate == 44100


def test_device_groups_wiring():
    from resonus_core.gui.devices import DeviceGroups, device_groups

    jack = DeviceInfo(3, "system", "JACK", 4, 2, 48000.0)
    devices = [*DEVICES, jack]
    store = {"s": MeasurementSettings()}
    changes, calls = [], []

    def on_change(new):
        store["s"] = new
        changes.append(new)

    def rates(i, o, n_in, n_out):
        calls.append((i, o, n_in, n_out))
        return (44100, 48000) if 3 not in (i, o) else (48000,)

    dg = device_groups(
        store["s"],
        on_change,
        lambda: store["s"],
        devices=devices,
        defaults=(DEVICES[0], DEVICES[0]),
        rates=rates,
    )
    assert isinstance(dg, DeviceGroups)
    assert [g.get_title() for g in dg.groups] == ["Device", "Inputs", "Outputs"]
    assert items(dg.inputs.rows[0].channel) == ["1", "2"]  # default device: UR22
    dg.device.input_row.set_selected(3)  # JACK system: 4 inputs, 48 kHz
    assert store["s"].input_device == "JACK: system" and store["s"].sample_rate == 48000
    assert items(dg.inputs.rows[0].channel) == ["1", "2", "3", "4"]  # channel menus refreshed
    assert items(dg.outputs.rows[0].channel) == ["1", "2"]
    dg.inputs.add_button.emit("activated")
    assert store["s"].input_channels == (1, 2)
    assert calls[-1] == (3, None, 2, 1)  # rate menu refreshed with the new channel count
    assert len(changes) == 2

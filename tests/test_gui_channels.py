"""Preferences groups "Inputs" / "Outputs": add-another channel rows."""

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gtk  # noqa: E402

if not Gtk.init_check():
    pytest.skip("no display available", allow_module_level=True)

from resonus_core.channels import channel_limit, menu_channels  # noqa: E402
from resonus_core.devices import DeviceInfo  # noqa: E402
from resonus_core.gui.channels import NO_REFERENCE_LABEL, ChannelGroup  # noqa: E402
from resonus_core.measure import MeasurementSettings  # noqa: E402

Adw.init()

UR22 = DeviceInfo(0, "UR22", "ALSA", 2, 2, 44100.0)
JACK = DeviceInfo(1, "system", "JACK", 4, 2, 48000.0)
DEVICES = [UR22, JACK]


def test_channel_limit_and_menu():
    assert channel_limit("ALSA: UR22", "input", DEVICES, None) == 2
    assert channel_limit("JACK: system", "input", DEVICES, None) == 4
    assert channel_limit("JACK: system", "output", DEVICES, UR22) == 2
    assert channel_limit("", "input", DEVICES, JACK) == 4  # default device
    assert channel_limit("", "input", DEVICES, None) == 0  # default unknown
    assert channel_limit("ALSA: gone", "output", DEVICES, UR22) == 0
    with pytest.raises(ValueError, match="direction"):
        channel_limit("", "sideways", DEVICES, UR22)
    assert menu_channels(4, (1,)) == [1, 2, 3, 4]
    assert menu_channels(2, (1, 3)) == [1, 2, 3]  # used channel kept selectable
    assert menu_channels(0, ()) == []


class Fixture:
    def __init__(self, direction="input", settings=None, default=UR22):
        self.store = settings or MeasurementSettings(input_device="ALSA: UR22")
        self.changes = []
        self.errors = []
        self.cg = ChannelGroup(
            self.store,
            self.on_change,
            lambda: self.store,
            direction,
            DEVICES,
            default,
            on_error=self.errors.append,
        )

    def on_change(self, new):
        self.store = new
        self.changes.append(new)


def items(widget):
    model = widget.get_model()
    return [model.get_string(k) for k in range(model.get_n_items())]


def state(cg):
    return [
        (r.row.get_title(), items(r.channel)[r.channel.get_selected()], r.label.get_text())
        for r in cg.rows
    ]


def test_initial_rows():
    f = Fixture()
    cg = f.cg
    assert cg.group.get_title() == "Inputs" and cg.mode_row is None
    assert state(cg) == [("Channel 1", "1", "m1")]
    assert items(cg.rows[0].channel) == ["1", "2"]
    assert cg.rows[0].remove is None  # no cross on the first row
    assert cg.add_button.get_title() == "Add input" and cg.add_button.get_sensitive()
    assert items(cg.reference_row) == [NO_REFERENCE_LABEL, "1", "2"]
    assert cg.reference_row.get_selected() == 0
    assert f.changes == []


def test_add_and_remove_rows():
    f = Fixture()
    f.cg.add_button.emit("activated")
    assert f.store.input_channels == (1, 2) and f.store.input_labels == ("m1", "m2")
    assert state(f.cg) == [("Channel 1", "1", "m1"), ("Channel 2", "2", "m2")]
    assert f.cg.rows[1].remove is not None
    assert not f.cg.add_button.get_sensitive()  # the UR22 has two inputs
    f.cg.rows[1].remove.emit("clicked")
    assert f.store.input_channels == (1,) and f.store.input_labels == ("m1",)
    assert len(f.cg.rows) == 1 and f.cg.add_button.get_sensitive()
    assert len(f.changes) == 2


def test_change_channel_and_label():
    f = Fixture()
    f.cg.rows[0].channel.set_selected(1)
    assert f.store.input_channels == (2,)
    f.cg.rows[0].label.set_text("m1L")
    assert len(f.changes) == 1  # only on Enter
    f.cg.rows[0].label.emit("activate")
    assert f.store.input_labels == ("m1L",)
    assert len(f.changes) == 2


def test_rejected_changes_revert():
    s = MeasurementSettings(
        input_device="JACK: system", input_channels=(1, 2), input_labels=("m1", "m2")
    )
    f = Fixture(settings=s)
    f.cg.rows[1].channel.set_selected(0)  # channel 1 twice
    assert f.errors and f.changes == []
    assert state(f.cg)[1][1] == "2"  # drop-down shows the settings in force
    f.cg.rows[1].label.set_text("m1")  # duplicate label
    f.cg.rows[1].label.emit("activate")
    assert len(f.errors) == 2 and f.store == s
    assert f.cg.rows[1].label.get_text() == "m2"
    f.cg.rows[0].label.set_text("")
    f.cg.rows[0].label.emit("activate")
    assert len(f.errors) == 3 and f.store == s


def test_reference_channel():
    f = Fixture()
    f.cg.reference_row.set_selected(2)
    assert f.store.reference_channel == 2
    f.cg.reference_row.set_selected(1)  # channel 1 is a measurement input
    assert f.errors and f.store.reference_channel == 2
    assert items(f.cg.reference_row)[f.cg.reference_row.get_selected()] == "2"


def test_outputs_group_and_mode():
    s = MeasurementSettings(output_device="ALSA: UR22")
    f = Fixture("output", s)
    cg = f.cg
    assert cg.group.get_title() == "Outputs" and cg.reference_row is None
    assert cg.add_button.get_title() == "Add output"
    assert state(cg) == [("Channel 1", "1", "s1")]
    assert items(cg.mode_row) == ["simultaneous", "sequential"]
    cg.add_button.emit("activated")
    assert f.store.output_labels == ("s1", "s2")
    cg.mode_row.set_selected(1)
    assert f.store.output_mode == "sequential"


def test_unknown_device_keeps_channels_and_cannot_add():
    s = MeasurementSettings(input_device="ALSA: gone", input_channels=(3,), input_labels=("m1",))
    f = Fixture(settings=s)
    assert items(f.cg.rows[0].channel) == ["1", "2", "3"] and state(f.cg)[0][1] == "3"
    assert not f.cg.add_button.get_sensitive()
    f.cg.add_button.emit("activated")
    assert f.errors and f.changes == []


def test_refresh_after_device_change():
    f = Fixture()
    f.store = f.store.replace(input_device="JACK: system")
    f.cg.refresh(f.store)
    assert items(f.cg.rows[0].channel) == ["1", "2", "3", "4"]
    assert len(f.cg.rows) == 1 and f.changes == []
    assert len([w for w in [f.cg.add_button] if w.get_parent() is not None]) == 1


def test_uses_current_settings():
    f = Fixture()
    f.store = f.store.replace(repetitions=4)  # changed on another page
    f.cg.add_button.emit("activated")
    assert f.store.repetitions == 4 and f.store.input_channels == (1, 2)

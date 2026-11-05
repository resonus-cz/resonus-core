from dataclasses import dataclass

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gtk  # noqa: E402

if not Gtk.init_check():
    pytest.skip("no display available", allow_module_level=True)

from resonus_core.gui.host import Action, StandaloneHost  # noqa: E402
from resonus_core.gui.preferences import settings_page  # noqa: E402
from resonus_core.settings import Settings, param  # noqa: E402

Adw.init()


@dataclass(frozen=True)
class Example(Settings):
    level: float = param(
        -12.0, label="Level", unit="dBFS", minimum=-60.0, maximum=0.0, group="Signal"
    )
    window: str = param("hann", label="Window", choices=("hann", "rect"), group="Analysis")
    simulate: bool = param(False, label="Simulate", group="Device")


def rows(page):
    found = []
    child = page.get_first_child()
    stack = [child]
    while stack:
        w = stack.pop()
        if w is None:
            continue
        if isinstance(w, Adw.SpinRow | Adw.ComboRow | Adw.SwitchRow):
            found.append(w)
        stack.extend([w.get_first_child(), w.get_next_sibling()])
    return found


def test_preferences_page_edits_settings():
    changes = []
    errors = []
    page = settings_page(Example(), changes.append, on_error=errors.append)
    by_type = {type(r).__name__: r for r in rows(page)}
    assert set(by_type) == {"SpinRow", "ComboRow", "SwitchRow"}
    by_type["SpinRow"].set_value(-6.0)
    by_type["ComboRow"].set_selected(1)
    by_type["SwitchRow"].set_active(True)
    assert changes[-1] == Example(level=-6.0, window="rect", simulate=True)
    assert not errors


class DummyPlugin:
    id = "dummy"
    title = "Dummy"
    icon_name = "audio-x-generic-symbolic"

    def __init__(self):
        self.ran = False

    def header_widgets(self, host):
        return [Gtk.Button(label="Measure")]

    def main_view(self, host):
        return Gtk.Label(label="content")

    def preferences(self, host):
        return [settings_page(Example(), lambda s: None)]

    def actions(self, host):
        return [Action("measure", lambda: setattr(self, "ran", True), "Measure", ["<Control>m"])]


def test_standalone_host_builds():
    app = Adw.Application(application_id="cz.resonus.Test")
    app.register(None)
    plugin = DummyPlugin()
    host = StandaloneHost(app, plugin, "0.1.0")
    assert host.window.get_title() == "Dummy"
    app.activate_action("measure", None)
    assert plugin.ran
    host.toast("hello")
    host.window.destroy()


@dataclass(frozen=True)
class WithTuple(Settings):
    rate: int = param(48000, label="Sample rate", choices=(44100, 48000), group="Device")
    channels: tuple[int, ...] = param((1,), label="Channels", minimum=1, group="Device")


def test_int_choices_combo_and_tuple_fields_skipped():
    changes = []
    page = settings_page(WithTuple(), changes.append)
    found = rows(page)
    assert [type(r).__name__ for r in found] == ["ComboRow"]  # tuple field: no generic row
    combo = found[0]
    labels = [combo.get_model().get_string(i) for i in range(combo.get_model().get_n_items())]
    assert labels == ["44100", "48000"] and combo.get_selected() == 1
    combo.set_selected(0)
    assert changes[-1] == WithTuple(rate=44100)


def group_titles(page):
    titles = []
    stack = [page.get_first_child()]
    while stack:
        w = stack.pop()
        if w is None:
            continue
        stack.append(w.get_next_sibling())
        if isinstance(w, Adw.PreferencesGroup):
            titles.append(w.get_title())
        else:
            stack.append(w.get_first_child())
    return titles


def test_settings_page_group_filter_in_given_order():
    # two pages from one settings class: Measurement / Analysis (resonus-room D-17)
    page = settings_page(Example(), lambda s: None, groups=("Device", "Signal"))
    assert group_titles(page) == ["Device", "Signal"]
    assert sorted(type(r).__name__ for r in rows(page)) == ["SpinRow", "SwitchRow"]
    assert group_titles(settings_page(Example(), lambda s: None)) == [
        "Signal",
        "Analysis",
        "Device",
    ]  # no filter: all groups in field order
    with pytest.raises(ValueError, match="Nope"):
        settings_page(Example(), lambda s: None, groups=("Signal", "Nope"))


def test_settings_pages_share_the_current_settings():
    # a change on one page must not be undone by a later change on another page
    store = {"s": Example()}

    def on_change(new):
        store["s"] = new

    def current():
        return store["s"]

    signal = settings_page(Example(), on_change, groups=("Signal",), current=current)
    analysis = settings_page(Example(), on_change, groups=("Analysis",), current=current)
    rows(signal)[0].set_value(-6.0)
    rows(analysis)[0].set_selected(1)
    assert store["s"] == Example(level=-6.0, window="rect")
    rows(signal)[0].set_value(-3.0)
    assert store["s"] == Example(level=-3.0, window="rect")

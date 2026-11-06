"""Window layout and pop-up panels (resonus-room Q-C6 and D-36).

Moved from the app template's skeleton and resonus-room (user decision 2026-11-06, "move
them"); the checks are those of the template's layout test and room's overlay tests.
"""

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gtk  # noqa: E402

if not Gtk.init_check():
    pytest.skip("no display available", allow_module_level=True)

from resonus_core.gui import layout, overlays  # noqa: E402
from resonus_core.style import TOKENS  # noqa: E402

Adw.init()


def _all(widget, kind):
    out = [widget] if isinstance(widget, kind) else []
    child = widget.get_first_child()
    while child is not None:
        out += _all(child, kind)
        child = child.get_next_sibling()
    return out


def test_css_classes():
    for cls in ("resonus-ribbon", "ribbon-caption", "resonus-brand", "resonus-panel"):
        assert f".{cls}" in layout.APP_CSS
    assert overlays.OVERLAY_CSS in layout.APP_CSS
    provider = Gtk.CssProvider()
    errors = []
    provider.connect("parsing-error", lambda *a: errors.append(a))
    provider.load_from_string(layout.APP_CSS)
    assert errors == []
    layout.install_css()
    layout.install_css()  # once per process


def test_header_brand_centres_the_title():
    brand = layout.header_brand("Room")
    assert brand.get_label() == TOKENS["brand_text"]
    assert brand.has_css_class("resonus-brand")
    header = Adw.HeaderBar()
    header.pack_start(brand)
    title = header.get_title_widget()
    assert isinstance(title, Adw.WindowTitle) and title.get_title() == "Room"


def test_ribbon_groups_line_up():
    a, b = layout.ribbon_grid(), layout.ribbon_grid()
    a.attach(layout.ribbon_button("media-record-symbolic", "Measure"), 0, 0, 1, 3)
    b.attach(layout.labelled("Length [s]", Gtk.SpinButton()), 0, 0, 1, 1)
    ribbon = layout.ribbon([("Measure", a), ("Files", b)])
    assert ribbon.has_css_class("resonus-ribbon")
    captions = [w.get_label() for w in _all(ribbon, Gtk.Label) if w.has_css_class("ribbon-caption")]
    assert captions == ["Measure", "Files"]
    assert len(_all(ribbon, Gtk.Separator)) == 1
    assert set(ribbon.size_group.get_widgets()) == {a, b}
    assert a.get_row_homogeneous()
    button = _all(a, Gtk.Button)[0]
    assert button.has_css_class("flat") and isinstance(button.get_child(), Adw.ButtonContent)
    label = _all(b, Gtk.Label)[0]
    assert label.get_label() == "Length [s]" and label.has_css_class("dim-label")


def test_tabbed_panel():
    stack = Adw.ViewStack()
    stack.add_titled(Gtk.Label(), "one", "One")
    box = layout.tabbed_panel(stack)
    switcher = _all(box, Adw.ViewSwitcher)[0]
    assert switcher.get_stack() is stack
    panel = stack.get_parent()
    assert panel.has_css_class("resonus-panel") and stack.get_vexpand()


def test_overlay_panels():
    grid, frame = overlays.band_table()
    assert not frame.get_visible() and frame.has_css_class(overlays.OVERLAY_CLASS)
    assert frame.get_halign() == Gtk.Align.END and frame.get_valign() == Gtk.Align.START
    scroller = frame.get_child()
    assert scroller.get_child().get_child() is grid  # through the viewport
    assert scroller.get_propagate_natural_width() and scroller.get_propagate_natural_height()
    assert scroller.get_max_content_height() == overlays.OVERLAY_MAX_HEIGHT == 480
    columns = ("Band [Hz]", "T20 [s]", "r²", "Flags")
    overlays.fill_table(grid, columns, [["125", "1.20", "0.99", "noise"]] * 21)
    assert grid.get_child_at(0, 0).get_label() == "Band [Hz]"
    assert grid.get_child_at(0, 0).has_css_class("heading")
    assert grid.get_child_at(1, 1).get_xalign() == 1.0  # numbers right
    assert grid.get_child_at(3, 1).get_xalign() == 0.0  # Flags left
    assert grid.get_child_at(0, 21).get_label() == "125"
    # the panel requests the table's full width (all columns, no sideways scrolling)
    width = grid.measure(Gtk.Orientation.HORIZONTAL, -1)[1]
    frame.set_visible(True)  # as its toggle does; hidden widgets measure 0
    assert frame.measure(Gtk.Orientation.HORIZONTAL, -1)[1] >= width
    overlays.fill_table(grid, (), [])
    assert grid.get_first_child() is None

    label, findings = overlays.findings_panel()
    assert label.get_wrap() and label.get_selectable()
    assert label.get_max_width_chars() == overlays.FINDINGS_CHARS == 72
    assert findings.has_css_class(overlays.OVERLAY_CLASS)


def test_exclusive_toggles():
    a, b = Gtk.ToggleButton(), Gtk.ToggleButton()
    overlays.exclusive(a, b)
    a.set_active(True)
    b.set_active(True)
    assert b.get_active() and not a.get_active()
    b.set_active(False)
    assert not a.get_active() and not b.get_active()  # both may be off

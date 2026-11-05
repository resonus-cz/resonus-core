"""MplPlotView.add_panel: more axes in one figure, shared x (resonus-room D-22)."""

from types import SimpleNamespace

import numpy as np
import pytest

from resonus_core.plot import SCROLL_ZOOM_FACTOR, MplPlotView, on_scroll, zoom_limits


def two_panels():
    view = MplPlotView()
    view.set_axes(xlabel="", ylabel="Level [dB]", xscale="log")
    panel = view.add_panel((2, 1))
    panel.set_axes(xlabel="Frequency [Hz]", ylabel="Phase [°]", xscale="log")
    return view, panel


def test_panel_shares_figure_and_x_axis():
    view, panel = two_panels()
    assert isinstance(panel, MplPlotView)
    assert panel.figure is view.figure and panel.axes is not view.axes
    assert view.figure.axes == [view.axes, panel.axes]
    assert view.axes.get_shared_x_axes().joined(view.axes, panel.axes)
    view.axes.set_xlim(100.0, 1000.0)
    np.testing.assert_allclose(panel.axes.get_xlim(), (100.0, 1000.0))
    assert view.panels() == [panel] and panel.panels() == [panel]


def test_layout_rows_and_tick_labels():
    view, panel = two_panels()
    gs = view.axes.get_subplotspec().get_gridspec()
    assert (gs.nrows, gs.ncols) == (2, 1)
    assert list(gs.get_height_ratios()) == [2.0, 1.0]
    assert panel.axes.get_subplotspec().get_gridspec() is gs
    assert view.axes.get_subplotspec().rowspan == range(0, 1)
    assert panel.axes.get_subplotspec().rowspan == range(1, 2)
    view.figure.canvas.draw()
    top = view.axes.get_position()
    bottom = panel.axes.get_position()
    assert top.y0 > bottom.y1  # view on top
    assert top.height / bottom.height == pytest.approx(2.0, rel=0.25)
    assert not any(t.label1.get_visible() for t in view.axes.xaxis.get_major_ticks())
    assert any(t.label1.get_visible() for t in panel.axes.xaxis.get_major_ticks())


def test_third_panel_from_a_panel():
    view, panel = two_panels()
    third = panel.add_panel((2, 1, 1))  # passed on to the top view
    assert view.panels() == [panel, third]
    gs = third.axes.get_subplotspec().get_gridspec()
    assert list(gs.get_height_ratios()) == [2.0, 1.0, 1.0]
    assert [v.axes.get_subplotspec().rowspan for v in (view, panel, third)] == [
        range(0, 1),
        range(1, 2),
        range(2, 3),
    ]
    view.figure.canvas.draw()
    assert not any(t.label1.get_visible() for t in panel.axes.xaxis.get_major_ticks())


def test_traces_and_legend_per_panel():
    view, panel = two_panels()
    view.set_trace("a", [100.0, 1000.0], [0.0, 1.0], label="a")
    panel.set_trace("p", [100.0, 1000.0], [10.0, 20.0], label="p")
    assert view.trace_ids() == ["a"] and panel.trace_ids() == ["p"]
    assert [ln.get_label() for ln in panel.axes.get_lines()] == ["p"]
    assert [t.get_text() for t in panel.axes.get_legend().get_texts()] == ["p"]
    panel.remove_trace("p")
    assert panel.trace_ids() == [] and panel.axes.get_legend() is None
    assert view.trace_ids() == ["a"]
    panel.refresh()  # no canvas yet: nothing to do, no error


def test_save_has_both_panels(tmp_path):
    view, panel = two_panels()
    view.save(str(tmp_path / "both.svg"))
    svg = (tmp_path / "both.svg").read_text()
    assert "Phase" in svg and "Level" in svg
    assert panel._repr_png_() == view._repr_png_()  # same figure


def test_scroll_on_panel_zooms_shared_x():
    view, panel = two_panels()
    view.axes.set_xlim(100.0, 1000.0)
    panel.axes.set_ylim(-180.0, 180.0)
    on_scroll(panel, SimpleNamespace(inaxes=panel.axes, xdata=200.0, ydata=0.0, button="up"))
    f = 1 / SCROLL_ZOOM_FACTOR
    np.testing.assert_allclose(view.axes.get_xlim(), zoom_limits(100.0, 1000.0, 200.0, f, True))
    np.testing.assert_allclose(panel.axes.get_ylim(), zoom_limits(-180.0, 180.0, 0.0, f, False))


@pytest.mark.parametrize(
    ("ratios", "match"),
    [((1,), "needs 2 values"), ((2, 1, 1), "needs 2 values"), ((2, 0), "> 0"), ((1, -1), "> 0")],
)
def test_add_panel_errors(ratios, match):
    view = MplPlotView()
    with pytest.raises(ValueError, match=match):
        view.add_panel(ratios)
    assert view.panels() == [] and len(view.figure.axes) == 1


def test_navigable_with_panel_one_toolbar_scrolls_panels():
    gi = pytest.importorskip("gi")
    gi.require_version("Gtk", "4.0")
    from gi.repository import Gtk

    if not Gtk.init_check():
        pytest.skip("no display available")
    from matplotlib.backend_bases import MouseEvent
    from matplotlib.backends.backend_gtk4 import NavigationToolbar2GTK4

    from resonus_core.plot import navigable

    view, panel = two_panels()
    box = navigable(view)
    assert isinstance(box.get_first_child(), NavigationToolbar2GTK4)
    assert box.get_last_child() is view.widget() is panel.widget()
    third = view.add_panel((2, 1, 1))  # added after navigable: scrolls too
    canvas = view.widget()
    view.axes.set_xlim(100.0, 1000.0)
    third.axes.set_ylim(0.0, 10.0)
    view.figure.draw_without_rendering()
    x, y = third.axes.transData.transform((300.0, 5.0))
    event = MouseEvent("scroll_event", canvas, x, y, button="up", step=1)
    assert event.inaxes is third.axes
    canvas.callbacks.process("scroll_event", event)
    assert third.axes.get_ylim()[1] - third.axes.get_ylim()[0] < 10.0

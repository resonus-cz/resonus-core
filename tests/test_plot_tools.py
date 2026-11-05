from types import SimpleNamespace

import numpy as np
import pytest

from resonus_core.plot import (
    SCROLL_ZOOM_FACTOR,
    MplPlotView,
    on_scroll,
    set_trace_visible,
    style_points,
    zoom_limits,
)


def test_zoom_limits_linear_and_log():
    assert zoom_limits(0.0, 10.0, 5.0, 0.5, False) == pytest.approx((2.5, 7.5))
    assert zoom_limits(0.0, 10.0, 2.0, 2.0, False) == pytest.approx((-2.0, 18.0))
    lo, hi = zoom_limits(100.0, 10000.0, 1000.0, 0.5, True)
    assert lo == pytest.approx(10**2.5) and hi == pytest.approx(10**3.5)
    for args in ((0.0, 1.0, 0.5, 0.0, False), (0.0, 10.0, 1.0, 0.5, True)):
        with pytest.raises(ValueError):
            zoom_limits(*args)


def view_with_lines():
    v = MplPlotView()
    v.set_axes(xlabel="f", ylabel="T", xscale="log")
    for name in ("EDT", "T20", "T30"):
        v.set_trace(name, [125.0, 250.0, 500.0], [1.0, 0.9, 0.8], label=name)
    return v


def test_scroll_zooms_around_cursor():
    v = view_with_lines()
    v.axes.set_xlim(100.0, 1000.0)
    v.axes.set_ylim(0.0, 2.0)
    on_scroll(v, SimpleNamespace(inaxes=v.axes, xdata=200.0, ydata=1.0, button="up"))
    f = 1 / SCROLL_ZOOM_FACTOR
    exp_x = zoom_limits(100.0, 1000.0, 200.0, f, True)
    np.testing.assert_allclose(v.axes.get_xlim(), exp_x)
    np.testing.assert_allclose(v.axes.get_ylim(), zoom_limits(0.0, 2.0, 1.0, f, False))
    on_scroll(v, SimpleNamespace(inaxes=v.axes, xdata=200.0, ydata=1.0, button="down"))
    np.testing.assert_allclose(v.axes.get_xlim(), (100.0, 1000.0))
    # outside the axes: nothing happens
    on_scroll(v, SimpleNamespace(inaxes=None, xdata=None, ydata=None, button="up"))
    np.testing.assert_allclose(v.axes.get_xlim(), (100.0, 1000.0))


def test_visibility_and_legend():
    v = view_with_lines()
    set_trace_visible(v, "T20", False)
    vis = {ln.get_label(): ln.get_visible() for ln in v.axes.get_lines()}
    assert vis == {"EDT": True, "T20": False, "T30": True}
    assert [t.get_text() for t in v.axes.get_legend().get_texts()] == ["EDT", "T30"]
    assert v.trace_ids() == ["EDT", "T20", "T30"]  # hidden, not removed
    for name in ("EDT", "T30"):
        set_trace_visible(v, name, False)
    assert v.axes.get_legend() is None
    set_trace_visible(v, "T20", True)
    assert [t.get_text() for t in v.axes.get_legend().get_texts()] == ["T20"]


def test_style_points():
    v = view_with_lines()
    style_points(v)
    assert all(ln.get_marker() == "o" for ln in v.axes.get_lines())
    assert v.axes.xaxis.get_gridlines()[0].get_visible()


def test_hidden_trace_ignored_by_autoscale():
    view = MplPlotView()
    view.set_trace("a", [1.0, 2.0], [1.0, 2.0], label="a")
    view.set_trace("b", [1.0, 2.0], [10.0, 20.0], label="b")
    set_trace_visible(view, "b", False)
    assert view.axes.get_ylim()[1] < 10.0


def test_navigable_plot_has_toolbar():
    gi = pytest.importorskip("gi")
    gi.require_version("Gtk", "4.0")
    from gi.repository import Gtk

    if not Gtk.init_check():
        pytest.skip("no display available")
    from matplotlib.backends.backend_gtk4 import NavigationToolbar2GTK4

    from resonus_core.plot import navigable

    view = MplPlotView()
    box = navigable(view)
    assert isinstance(box, Gtk.Box)
    assert isinstance(box.get_first_child(), NavigationToolbar2GTK4)
    assert box.get_last_child() is view.widget()

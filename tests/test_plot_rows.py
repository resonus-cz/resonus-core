"""MplPlotView.add_row: panels side by side in one figure (resonus-room D-29)."""

from types import SimpleNamespace

import numpy as np
import pytest

from resonus_core.plot import MplPlotView, on_scroll


def clarity_layout():
    view = MplPlotView()
    lower = view.add_row((2, 1), 2)
    return view, lower


def test_row_of_two_below_a_wide_view():
    view, lower = clarity_layout()
    assert len(lower) == 2 and all(isinstance(p, MplPlotView) for p in lower)
    assert view.panels() == lower and lower[1].panels() == lower
    assert view.figure.axes == [view.axes, lower[0].axes, lower[1].axes]
    for p in lower:
        assert p.figure is view.figure
        assert view.axes.get_shared_x_axes().joined(view.axes, p.axes)
        assert p.widget is not None
    gs = view.axes.get_subplotspec().get_gridspec()
    assert (gs.nrows, gs.ncols) == (2, 1) and list(gs.get_height_ratios()) == [2.0, 1.0]
    view.figure.canvas.draw()
    top = view.axes.get_position()
    left, right = (p.axes.get_position() for p in lower)
    # the wide view spans both lower panels; the lower ones sit side by side, below it
    assert top.x0 <= left.x0 + 1e-9 and top.x1 >= right.x1 - 1e-9
    assert left.x1 < right.x0
    assert max(left.y1, right.y1) < top.y0
    assert left.height == pytest.approx(right.height)
    assert left.width == pytest.approx(right.width, rel=1e-6)
    # only the last row labels the shared x axis
    assert not view.axes.xaxis.get_tick_params()["labelbottom"]
    for p in lower:
        assert p.axes.xaxis.get_tick_params().get("labelbottom", True)


def test_width_ratios():
    view = MplPlotView()
    lower = view.add_row((1, 1), 2, width_ratios=(3, 1))
    view.figure.canvas.draw()
    a, b = (p.axes.get_position().width for p in lower)
    assert a / b == pytest.approx(3.0, rel=0.15)


def test_add_panel_is_a_one_column_row():
    view = MplPlotView()
    panel = view.add_panel((2, 1))
    gs = view.axes.get_subplotspec().get_gridspec()
    assert panel.axes.get_subplotspec().get_gridspec() is gs
    assert panel.axes.get_subplotspec().rowspan == range(1, 2)
    assert view._rows == [1]


def test_rows_keep_their_columns_when_more_are_added():
    view = MplPlotView()
    pair = view.add_row((2, 1), 2)
    last = view.add_panel((2, 1, 1))  # a third row, one wide panel
    assert view.panels() == pair + [last]
    assert view._rows == [2, 1]
    view.figure.canvas.draw()
    top = view.axes.get_position()
    a, b = (p.axes.get_position() for p in pair)
    bottom = last.axes.get_position()
    assert a.x1 < b.x0  # still side by side
    assert bottom.y1 < min(a.y0, b.y0) and bottom.x1 >= b.x1 - 1e-9
    assert top.height > a.height
    # tick labels: only the last row
    for p in pair:
        assert not p.axes.xaxis.get_tick_params()["labelbottom"]
    assert last.axes.xaxis.get_tick_params().get("labelbottom", True)
    # called on a panel, the row goes to the top view
    more = pair[0].add_row((2, 1, 1, 1), 3)
    assert len(more) == 3 and view.panels()[-3:] == more and view._rows == [2, 1, 3]


def test_panels_have_own_traces_and_scroll_zoom():
    view, (d50, ts) = clarity_layout()
    for v in (view, d50, ts):
        v.set_axes(xlabel="f", ylabel="y", xscale="log")
    d50.set_trace("a", [100, 1000], [0.2, 0.8], label="A")
    ts.set_trace("b", [100, 1000], [80, 120], label="B")
    assert d50.trace_ids() == ["a"] and ts.trace_ids() == ["b"] and view.trace_ids() == []
    view.axes.set_xlim(100, 1000)
    ts.axes.set_ylim(0, 200)
    event = SimpleNamespace(inaxes=ts.axes, xdata=300.0, ydata=100.0, button="up", step=1)
    on_scroll(ts, event)
    lo, hi = ts.axes.get_ylim()
    assert hi - lo < 200  # zoomed in the panel that got the event
    np.testing.assert_allclose(d50.axes.get_xlim(), view.axes.get_xlim())


@pytest.mark.parametrize(
    ("args", "kwargs", "message"),
    [
        (((2, 1, 1), 2), {}, "height_ratios needs 2 values, one per row"),
        (((2, 0), 2), {}, "height_ratios must be > 0"),
        (((2, float("nan")), 2), {}, "height_ratios must be > 0"),
        (((2, 1), 0), {}, "columns must be >= 1"),
        (((2, 1), 2), {"width_ratios": (1, 1, 1)}, "width_ratios needs 2 values"),
        (((2, 1), 2), {"width_ratios": (1, -1)}, "width_ratios must be > 0"),
        (((2, 1), 2), {"width_ratios": (1, float("nan"))}, "width_ratios must be > 0"),
    ],
)
def test_errors_change_nothing(args, kwargs, message):
    view = MplPlotView()
    with pytest.raises(ValueError, match=message):
        view.add_row(*args, **kwargs)
    assert view.panels() == [] and view._rows == [] and view.figure.axes == [view.axes]


def test_add_panel_message():
    view = MplPlotView()
    with pytest.raises(ValueError, match="height_ratios needs 2 values, one per row"):
        view.add_panel((1,))

"""Plot interface used by all apps, and its matplotlib renderer.

Apps describe *what* to show through :class:`PlotView`; renderers decide *how*.
:class:`MplPlotView` serves notebooks, CLI exports and (for now) the GUI; the GPU
widget (decision D6) will implement the same interface.

Interactive helpers for the GUI (navigation toolbar, scroll zoom, trace visibility,
point markers) operate on :class:`MplPlotView`; GTK is imported only by
:func:`navigable` (decision D-P1).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal, Protocol, cast

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from numpy.typing import ArrayLike

from resonus_core import style

Scale = Literal["linear", "log"]


class PlotView(Protocol):
    """Renderer-independent plot: named traces on one pair of axes."""

    def set_axes(
        self,
        *,
        xlabel: str,
        ylabel: str,
        xscale: Scale = "linear",
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        title: str = "",
    ) -> None:
        """Configure axes labels (with units), scale and limits."""
        ...

    def set_trace(self, trace_id: str, x: ArrayLike, y: ArrayLike, *, label: str) -> None:
        """Add a trace, or replace the data of an existing one."""
        ...

    def remove_trace(self, trace_id: str) -> None:
        """Remove a trace; unknown ids are ignored."""
        ...

    def trace_ids(self) -> list[str]:
        """Return the ids of the shown traces, in insertion order."""
        ...


def _freq_label(value: float, _pos: int) -> str:
    """Frequency tick label: 31.5, 500, 1k, 2.5k, ..."""
    if value >= 1000:
        return f"{value / 1000:g}k"
    return f"{value:g}"


class MplPlotView:
    """:class:`PlotView` rendered with matplotlib, in the unified style.

    Parameters
    ----------
    figsize
        Figure size in inches.
    brand
        Place the brand mark in the bottom-right corner.
    brand_text
        Text of the mark (:func:`resonus_core.style.add_brand`); None for "Re:sonus",
        empty for no visible mark. Change it later with :meth:`set_brand`.

    A view can get further axes below it in the same figure, sharing its x axis
    (:meth:`add_panel`): one canvas, one navigation toolbar, one exported image.
    """

    def __init__(
        self,
        figsize: tuple[float, float] = (8.0, 4.5),
        brand: bool = True,
        brand_text: str | None = None,
    ) -> None:
        style.register_fonts()
        import matplotlib as mpl

        with mpl.rc_context(cast(Any, style.mpl_rc())):
            self.figure = Figure(figsize=figsize, layout="constrained")
            self.axes = self.figure.add_subplot()
        self._lines: dict[str, Any] = {}
        self._canvas: Any = None
        self._parent: MplPlotView | None = None
        self._panels: list[MplPlotView] = []
        self._brand: Any = style.add_brand(self.figure, brand_text) if brand else None

    def add_panel(self, height_ratios: Sequence[float]) -> MplPlotView:
        """Add a view with its own axes below this view's, in the same figure.

        The figure is laid out again as one column of ``n = len(self._panels) + 2`` rows
        (this view on top, earlier panels, the new panel at the bottom): ``gs =
        self.figure.add_gridspec(n, 1, height_ratios=list(height_ratios))``; the
        existing axes get ``ax.set_subplotspec(gs[k])`` and ``ax.tick_params(
        labelbottom=False)`` (only the lowest axes label the shared x axis); the new axes
        are ``self.figure.add_subplot(gs[n - 1], sharex=self.axes)`` created inside the
        style's ``rc_context``. The returned view (built with :meth:`_panel`) has the same
        ``figure``, its own traces and legend; its :meth:`widget` is this view's canvas
        and its :meth:`refresh` redraws it. On a panel, the call is passed on to the view
        it was added to. Ends with :meth:`refresh`.

        Parameters
        ----------
        height_ratios
            Relative heights of all rows, top down, one per view after adding (``n``
            values), each > 0. Example: ``view.add_panel((2, 1))`` gives a panel half
            the height of the view above it.

        Raises
        ------
        ValueError
            ``f"height_ratios needs {n} values, one per panel"``;
            ``"height_ratios must be > 0"``.
        """
        if self._parent is not None:
            return self._parent.add_panel(height_ratios)
        ratios = [float(r) for r in height_ratios]
        if len(ratios) != len(self._panels) + 2:
            raise ValueError(f"height_ratios needs {len(self._panels) + 2} values, one per panel")
        if any(not r > 0 for r in ratios):  # also rejects NaN
            raise ValueError("height_ratios must be > 0")
        import matplotlib as mpl

        with mpl.rc_context(cast(Any, style.mpl_rc())):
            n = len(self._panels) + 2
            gs = self.figure.add_gridspec(n, 1, height_ratios=ratios)
            # reposition existing axes
            for k, ax in enumerate([self.axes] + [p.axes for p in self._panels]):
                ax.set_subplotspec(gs[k])
                if k < n - 1:
                    ax.tick_params(labelbottom=False)
            # new panel
            new_ax = self.figure.add_subplot(gs[n - 1], sharex=self.axes)
            panel = self._panel(self, new_ax)
            self._panels.append(panel)
            self.refresh()
            return panel

    @classmethod
    def _panel(cls, parent: MplPlotView, axes: Any) -> MplPlotView:
        """Return a view of ``axes`` in ``parent``'s figure (no new figure, no brand).

        ``view = cls.__new__(cls)`` (skips ``__init__``), then set ``figure =
        parent.figure``, ``axes``, ``_lines = {}``, ``_canvas = None``, ``_parent =
        parent``, ``_panels = []``, ``_brand = None``.
        """
        view = cls.__new__(cls)
        view.figure = parent.figure
        view.axes = axes
        view._lines = {}
        view._canvas = None
        view._parent = parent
        view._panels = []
        view._brand = None
        return view

    def set_brand(self, text: str | None) -> None:
        """Change the text of the brand mark; None restores "Re:sonus", empty hides it.

        ``label = style.TOKENS["brand_text"] if text is None else text``; on the mark
        (``self._brand``): ``set_text(label)``, ``set_visible(bool(label))``, then
        :meth:`refresh`. On a panel the call goes to ``self._parent``. A view created
        with ``brand=False`` has no mark: nothing happens.
        """
        if self._parent is not None:
            self._parent.set_brand(text)
            return
        if self._brand is None:
            return
        label = style.TOKENS["brand_text"] if text is None else text
        self._brand.set_text(label)
        self._brand.set_visible(bool(label))
        self.refresh()

    def panels(self) -> list[MplPlotView]:
        """Return the views added below this one by :meth:`add_panel`, top down.

        A copy of ``self._panels``; on a panel, ``self._parent.panels()``.
        """
        if self._parent is not None:
            return self._parent.panels()
        return list(self._panels)

    def set_axes(
        self,
        *,
        xlabel: str,
        ylabel: str,
        xscale: Scale = "linear",
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        title: str = "",
    ) -> None:
        """Configure axes labels (with units), scale and limits."""
        ax = self.axes
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xscale(xscale)
        if xscale == "log":
            ax.xaxis.set_major_formatter(FuncFormatter(_freq_label))
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        self.refresh()

    def set_trace(self, trace_id: str, x: ArrayLike, y: ArrayLike, *, label: str) -> None:
        """Add a trace, or replace the data of an existing one."""
        xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if xa.shape != ya.shape or xa.ndim != 1:
            raise ValueError("x and y must be 1-D arrays of equal length")
        if trace_id in self._lines:
            line = self._lines[trace_id]
            line.set_data(xa, ya)
            line.set_label(label)
        else:
            color = style.TRACE_COLORS[len(self._lines) % len(style.TRACE_COLORS)]
            (self._lines[trace_id],) = self.axes.plot(
                xa, ya, label=label, color=color, lw=style.TOKENS["line_width"]
            )
        self.axes.relim()
        self.axes.autoscale_view()
        self._legend()
        self.refresh()

    def remove_trace(self, trace_id: str) -> None:
        """Remove a trace; unknown ids are ignored."""
        line = self._lines.pop(trace_id, None)
        if line is not None:
            line.remove()
            self._legend()
            self.refresh()

    def trace_ids(self) -> list[str]:
        """Return the ids of the shown traces, in insertion order."""
        return list(self._lines)

    def widget(self) -> Any:
        """Return a GTK 4 widget showing the figure (created on first call).

        A panel (:meth:`add_panel`, ``self._parent`` not None) returns
        ``self._parent.widget()``.
        """
        if self._parent is not None:
            return self._parent.widget()
        if self._canvas is None:
            from matplotlib.backends.backend_gtk4agg import FigureCanvasGTK4Agg

            self._canvas = FigureCanvasGTK4Agg(self.figure)
            self._canvas.set_hexpand(True)
            self._canvas.set_vexpand(True)
        return self._canvas

    def refresh(self) -> None:
        """Redraw the GTK widget, if one exists (a panel: ``self._parent.refresh()``)."""
        if self._parent is not None:
            self._parent.refresh()
            return
        if self._canvas is not None:
            self._canvas.draw_idle()

    def _repr_png_(self) -> bytes:
        """Return the figure as PNG, so a view displays itself in Jupyter."""
        import io

        buf = io.BytesIO()
        self.figure.savefig(buf, format="png")
        return buf.getvalue()

    def save(self, path: str) -> None:
        """Export the figure (format from the file extension: png, pdf, svg)."""
        self.figure.savefig(path)

    def _legend(self) -> None:
        if self._lines:
            self.axes.legend(loc="best")
            return
        legend = self.axes.get_legend()
        if legend is not None:
            legend.remove()


# Interactive navigation, zoom and trace visibility (moved from resonus-room, user decision
# 2026-11-04, resonus-room D-15).
# Zoom step of one scroll-wheel click (design choice): scrolling up shows 1/1.2 of the
# current range around the cursor, scrolling down 1.2 times the range.
SCROLL_ZOOM_FACTOR = 1.2
MARKER_SIZE = 4.0


def zoom_limits(
    lower: float, upper: float, center: float, factor: float, log: bool
) -> tuple[float, float]:
    """Return new axis limits after zooming by ``factor`` around ``center``.

    Linear axis: ``center - (center - lower) * factor`` and ``center + (upper - center) *
    factor``. Log axis: the same applied to ``log10`` of all three values, the result
    converted back with ``10**``. ``factor < 1`` zooms in, ``> 1`` out.

    Raises
    ------
    ValueError
        If ``factor <= 0``, or on a log axis if any value is ``<= 0``.
    """
    if factor <= 0:
        raise ValueError("factor must be > 0")
    if log:
        if min(lower, upper, center) <= 0:
            raise ValueError("log axis limits must be > 0")
        log_lower = math.log10(lower)
        log_upper = math.log10(upper)
        log_center = math.log10(center)
        new_lower = 10 ** (log_center - (log_center - log_lower) * factor)
        new_upper = 10 ** (log_center + (log_upper - log_center) * factor)
        return new_lower, new_upper
    else:
        new_lower = center - (center - lower) * factor
        new_upper = center + (upper - center) * factor
        return new_lower, new_upper


def on_scroll(view: MplPlotView, event: Any) -> None:
    """Zoom both axes of ``view`` around the mouse position for a matplotlib scroll event.

    Ignore events with ``event.inaxes is not view.axes`` or without ``xdata``/``ydata``.
    ``factor = 1 / SCROLL_ZOOM_FACTOR`` for ``event.button == "up"``, else
    ``SCROLL_ZOOM_FACTOR``. New x limits from :func:`zoom_limits` with ``log =
    view.axes.get_xscale() == "log"``, y limits likewise; set them, then
    ``view.refresh()``.
    """
    if event.inaxes is not view.axes or event.xdata is None or event.ydata is None:
        return
    f = 1 / SCROLL_ZOOM_FACTOR if event.button == "up" else SCROLL_ZOOM_FACTOR
    ax = view.axes
    xlim = zoom_limits(ax.get_xlim()[0], ax.get_xlim()[1], event.xdata, f, ax.get_xscale() == "log")
    ylim = zoom_limits(ax.get_ylim()[0], ax.get_ylim()[1], event.ydata, f, ax.get_yscale() == "log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    view.refresh()


def navigable(view: MplPlotView) -> Any:
    """Return a GTK box with matplotlib's navigation toolbar above the plot.

    ``canvas = view.widget()``; ``toolbar = NavigationToolbar2GTK4(canvas)`` (from
    ``matplotlib.backends.backend_gtk4``); ``box = Gtk.Box(orientation=VERTICAL)``;
    append toolbar, then canvas; ``canvas.mpl_connect("scroll_event", lambda e:
    on_scroll(view, e))``. The toolbar gives pan, zoom-to-rectangle, home/back/forward
    and save.

    With panels (:meth:`MplPlotView.add_panel`): call it once, on the top view; the
    toolbar then acts on all axes of the figure, and the scroll handler calls
    :func:`on_scroll` for the view and each of ``view.panels()`` (each ignores events
    outside its own axes; the panels are looked up at each event, so panels added later
    zoom too).
    """
    import gi

    gi.require_version("Gtk", "4.0")
    from gi.repository import Gtk
    from matplotlib.backends.backend_gtk4 import NavigationToolbar2GTK4

    canvas = view.widget()
    toolbar = NavigationToolbar2GTK4(canvas)
    box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    box.append(toolbar)
    box.append(canvas)

    def scroll(event: Any) -> None:
        for v in (view, *view.panels()):
            on_scroll(v, event)

    canvas.mpl_connect("scroll_event", scroll)
    return box


def set_trace_visible(view: MplPlotView, label: str, visible: bool) -> None:
    """Show or hide the lines labelled ``label``; the legend lists visible lines only.

    For each ``line in view.axes.get_lines()`` with ``line.get_label() == label``:
    ``line.set_visible(visible)``. Then ``shown = [ln for ln in view.axes.get_lines() if
    ln.get_visible()]``; if ``shown``: ``view.axes.legend(handles=shown, loc="best")``,
    else remove an existing legend. Rescale to the visible lines
    (``view.axes.relim(visible_only=True)``, ``view.axes.autoscale_view()``), then
    ``view.refresh()``.
    """
    for ln in view.axes.get_lines():
        if ln.get_label() == label:
            ln.set_visible(visible)
    shown = [ln for ln in view.axes.get_lines() if ln.get_visible()]
    if shown:
        view.axes.legend(handles=shown, loc="best")
    else:
        legend = view.axes.get_legend()
        if legend is not None:
            legend.remove()
    view.axes.relim(visible_only=True)
    view.axes.autoscale_view()
    view.refresh()


def style_points(view: MplPlotView) -> None:
    """Mark the data points of all lines and add a grid (band plots, ISO 3382-2 9.1).

    For each line: ``line.set_marker("o")``, ``line.set_markersize(MARKER_SIZE)``;
    ``view.axes.grid(True, which="both", alpha=0.3)``; ``view.refresh()``.
    """
    for ln in view.axes.get_lines():
        ln.set_marker("o")
        ln.set_markersize(MARKER_SIZE)
    view.axes.grid(True, which="both", alpha=0.3)
    view.refresh()

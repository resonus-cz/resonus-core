"""Pop-up panels over a plot: band tables and findings behind toggle buttons.

A panel is a ``Gtk.Frame`` with the CSS class :data:`OVERLAY_CLASS` (a nearly opaque
view background, accent border; ``OVERLAY_CSS``), placed at the top right of a
``Gtk.Overlay`` whose child is the plot, hidden until its toggle is active, so the plot
keeps the full height (resonus-room D-36, user 2026-11-06). The scroller takes the
natural size of its content (height capped), so a table shows all its columns when the
window allows.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from gi.repository import Gtk

OVERLAY_CLASS: Final = "resonus-overlay"
# Background 95 % view colour: the plot behind barely shows through (user, 2026-11-06:
# "make the background less transparent"); layout choice, not normative.
OVERLAY_CSS: Final = """
.resonus-overlay {
  background-color: color-mix(in srgb, var(--view-bg-color) 95%, transparent);
  border: 1px solid color-mix(in srgb, var(--accent-bg-color) 45%, transparent);
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
}
"""
# Height cap of a panel's content, px (layout choice: 21 one-third-octave rows at a
# 850 px window). No width cap: the panel takes the table's natural width, and the
# overlay limits it to the plot's width (scrolling only in a narrower window).
OVERLAY_MAX_HEIGHT: Final = 480
# Findings text wraps at this many characters (layout choice).
FINDINGS_CHARS: Final = 72


def overlay_panel(child: Gtk.Widget) -> Gtk.Frame:
    """Wrap ``child`` in a hidden pop-up panel for the top right of a plot overlay.

    Parameters
    ----------
    child
        The content (a table grid or a label).

    Returns
    -------
    Gtk.Frame
        Invisible, ``halign`` END, ``valign`` START, 12 px top/end margins, CSS class
        :data:`OVERLAY_CLASS`; its child is a ``Gtk.ScrolledWindow`` propagating the
        natural width and height of ``child`` (height up to :data:`OVERLAY_MAX_HEIGHT`),
        with 6 px padding margins.
    """
    scroller = Gtk.ScrolledWindow(
        propagate_natural_width=True,
        propagate_natural_height=True,
        max_content_height=OVERLAY_MAX_HEIGHT,
        margin_top=6,
        margin_bottom=6,
    )
    scroller.set_child(child)
    frame = Gtk.Frame(
        visible=False,
        halign=Gtk.Align.END,
        valign=Gtk.Align.START,
        margin_top=12,
        margin_end=12,
    )
    frame.add_css_class(OVERLAY_CLASS)
    frame.set_child(scroller)
    return frame


def band_table() -> tuple[Gtk.Grid, Gtk.Frame]:
    """Return a band table grid in a hidden pop-up panel (:func:`overlay_panel`)."""
    grid = Gtk.Grid(column_spacing=18, row_spacing=4, margin_start=12, margin_end=12)
    return grid, overlay_panel(grid)


def findings_panel() -> tuple[Gtk.Label, Gtk.Frame]:
    """Return a wrapped, selectable findings label in a hidden pop-up panel."""
    label = Gtk.Label(
        wrap=True,
        selectable=True,
        xalign=0.0,
        yalign=0.0,
        max_width_chars=FINDINGS_CHARS,
        width_chars=FINDINGS_CHARS,
        margin_start=12,
        margin_end=12,
    )
    return label, overlay_panel(label)


def exclusive(*toggles: Gtk.ToggleButton) -> None:
    """Activating one of ``toggles`` deactivates the others (panels share a corner)."""

    def on_toggled(button: Gtk.ToggleButton) -> None:
        if button.get_active():
            for other in toggles:
                if other is not button:
                    other.set_active(False)

    for t in toggles:
        t.connect("toggled", on_toggled)


def fill_table(grid: Gtk.Grid, columns: Sequence[str], rows: list[list[str]]) -> None:
    """Replace the cells of ``grid``: a heading row ``columns``, then ``rows``.

    Numbers right-aligned; the first column (band) and a last "Flags" column left-aligned.
    """
    child = grid.get_first_child()
    while child:
        next_child = child.get_next_sibling()
        grid.remove(child)
        child = next_child
    last_left = bool(columns) and columns[-1] == "Flags"
    for col, title in enumerate(columns):
        head = Gtk.Label(label=title, xalign=1.0 if col else 0.0)
        head.add_css_class("heading")
        grid.attach(head, col, 0, 1, 1)
    for r, row in enumerate(rows, start=1):
        for col, text in enumerate(row):
            right = 0 < col and not (last_left and col == len(row) - 1)
            cell = Gtk.Label(label=text, xalign=1.0 if right else 0.0)
            cell.add_css_class("numeric")
            grid.attach(cell, col, r, 1, 1)

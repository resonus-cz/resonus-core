"""Window layout of a Re:sonus app: header brand and title, ribbon, tabs in a panel.

The layout of resonus-room (its decisions D-20 ribbon, D-22, D-23 header brand and
title), which every Re:sonus app follows (app architecture section 2, D8); moved here
from the app template and resonus-room (resonus-room Q-C6, user decision 2026-11-06):

- header bar: the brand "Re:sonus" left (Lexend SemiBold, the weight of the plot mark),
  the app name without "Resonus" centred, the main menu right; no other controls;
- below it a ribbon of titled groups (e.g. Measure, Device, Files, Results), each a grid
  of three rows with its caption underneath, separated by vertical lines;
- below the ribbon a tab row (``Adw.ViewSwitcher``) and the tab content in a frame with
  a slight accent border (CSS class ``resonus-panel``).

Pop-up panels over plots (tables, findings) are in :mod:`resonus_core.gui.overlays`;
their CSS is part of :data:`APP_CSS`.

A deliberate departure from the GNOME HIG (header-bar controls), decided by the user
for resonus-room on 2026-11-05.
"""

from __future__ import annotations

from collections.abc import Sequence

from gi.repository import Adw, Gdk, GLib, Gtk, PangoCairo

from resonus_core.gui.overlays import OVERLAY_CSS
from resonus_core.style import TOKENS, font_files

# Ribbon, captions, brand and the accent frame around the tab content (design choices of
# resonus-room D-20...D-23). The brand's 5px margin lines its text up with the ribbon's
# first button (header 6px + 1px side bearing + 5px = ribbon padding 12px).
APP_CSS = (
    """
.resonus-ribbon { padding: 6px 12px 3px 12px; }
.resonus-ribbon .ribbon-caption { font-size: smaller; opacity: 0.6; }
.resonus-ribbon button.destructive-action {
  background-color: var(--destructive-bg-color);
  color: var(--destructive-fg-color);
}
.resonus-brand { font-family: Lexend; font-weight: 600; font-size: 1.2em; margin-left: 5px; }
.resonus-panel {
  border: 1px solid color-mix(in srgb, var(--accent-bg-color) 45%, transparent);
  border-radius: 8px;
}
"""
    + OVERLAY_CSS
)

_fonts_installed = False
_css_installed = False


def install_fonts() -> None:
    """Make the bundled Lexend fonts (resonus-core) available to GTK, once per process.

    ``PangoCairo.FontMap.get_default().add_font_file(path)`` for each of
    ``resonus_core.style.font_files()`` (Pango >= 1.56); without that call, or if a file
    cannot be added, GTK falls back to the default font.
    """
    global _fonts_installed
    if _fonts_installed:
        return
    font_map = PangoCairo.FontMap.get_default()
    if hasattr(font_map, "add_font_file"):
        for path in font_files():
            try:
                font_map.add_font_file(str(path))
            except GLib.Error:
                pass
    _fonts_installed = True


def install_css() -> None:
    """Load ``APP_CSS`` for the default display, once per process."""
    global _css_installed
    display = Gdk.Display.get_default()
    if _css_installed or display is None:
        return
    provider = Gtk.CssProvider()
    provider.load_from_string(APP_CSS)
    Gtk.StyleContext.add_provider_for_display(
        display, provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
    )
    _css_installed = True


def header_brand(title: str) -> Gtk.Label:
    """Return the header-bar brand label; once packed, it centres ``title`` in that bar.

    Return it from ``AppPlugin.header_widgets``: the host packs it at the start of its
    ``Adw.HeaderBar``; the label then replaces the bar's title widget by
    ``Adw.WindowTitle(title=title)``. The window title itself (window list, About dialog)
    stays the plugin's ``title``.
    """
    install_fonts()
    install_css()
    brand = Gtk.Label(label=str(TOKENS["brand_text"]))
    brand.add_css_class("resonus-brand")

    def centre(widget: Gtk.Widget, _pspec: object) -> None:
        header = widget.get_ancestor(Adw.HeaderBar)
        if isinstance(header, Adw.HeaderBar):
            header.set_title_widget(Adw.WindowTitle(title=title))

    brand.connect("notify::parent", centre)
    return brand


def ribbon_button(icon_name: str, label: str) -> Gtk.Button:
    """Return a flat ribbon button with icon and label (``Adw.ButtonContent``)."""
    button = Gtk.Button(child=Adw.ButtonContent(icon_name=icon_name, label=label))
    button.add_css_class("flat")
    return button


def labelled(text: str, widget: Gtk.Widget) -> Gtk.Widget:
    """Return a box with a dimmed label before ``widget`` (e.g. ribbon spin buttons)."""
    box = Gtk.Box(spacing=6, valign=Gtk.Align.CENTER)
    label = Gtk.Label(label=text)
    label.add_css_class("dim-label")
    box.append(label)
    box.append(widget)
    return box


def ribbon_grid() -> Gtk.Grid:
    """Return the grid of one ribbon group (rows of equal height; resonus-room uses three)."""
    return Gtk.Grid(row_spacing=3, column_spacing=6, row_homogeneous=True)


def ribbon_group(title: str, content: Gtk.Widget) -> Gtk.Widget:
    """Return one ribbon group: ``content`` with the caption ``title`` below."""
    caption = Gtk.Label(label=title)
    caption.add_css_class("ribbon-caption")
    group = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
    group.append(content)
    group.append(caption)
    return group


def ribbon(groups: Sequence[tuple[str, Gtk.Widget]]) -> Gtk.Widget:
    """Return the ribbon: the ``(title, content)`` groups left to right, with separators.

    The contents share one vertical ``Gtk.SizeGroup``: all get the height of the tallest,
    so the captions line up whatever the groups hold (kept on the box as ``size_group``).
    """
    install_css()
    box = Gtk.Box(spacing=12)
    box.add_css_class("resonus-ribbon")
    sizes = Gtk.SizeGroup(mode=Gtk.SizeGroupMode.VERTICAL)
    box.size_group = sizes  # keep the group alive
    for k, (title, content) in enumerate(groups):
        sizes.add_widget(content)
        if k:
            box.append(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL))
        box.append(ribbon_group(title, content))
    return box


def tabbed_panel(stack: Adw.ViewStack) -> Gtk.Widget:
    """Return the tab row (``Adw.ViewSwitcher``, wide, left-aligned) over ``stack`` in a panel.

    The panel is a box with the CSS class ``resonus-panel`` and 12 px margins.
    """
    install_css()
    switcher = Adw.ViewSwitcher(stack=stack, policy=Adw.ViewSwitcherPolicy.WIDE)
    switcher.set_halign(Gtk.Align.START)
    tabs = Gtk.Box(margin_start=12, margin_end=12, margin_top=3, margin_bottom=6)
    tabs.append(switcher)
    stack.set_vexpand(True)
    panel = Gtk.Box(
        orientation=Gtk.Orientation.VERTICAL,
        vexpand=True,
        margin_start=12,
        margin_end=12,
        margin_bottom=12,
        overflow=Gtk.Overflow.HIDDEN,
    )
    panel.add_css_class("resonus-panel")
    panel.append(stack)
    box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    box.append(tabs)
    box.append(panel)
    return box

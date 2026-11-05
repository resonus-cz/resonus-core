"""Plugin protocol and the standalone host window (GNOME HIG, libadwaita).

An app implements :class:`AppPlugin` once. :func:`run_standalone` hosts it in its own
window; the viewer bundle will host many plugins through the same protocol, discovered
via the entry-point group :data:`ENTRY_POINT_GROUP`.
"""

from __future__ import annotations

import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from gi.repository import Adw, Gio, GLib, Gtk

ENTRY_POINT_GROUP = "resonus.apps"
T = TypeVar("T")


@dataclass
class Action:
    """An app action for menus and keyboard shortcuts.

    Parameters
    ----------
    name
        Action name, unique within the app (used as ``app.<name>``).
    callback
        Called when the action is activated.
    label
        Menu label; None keeps it out of the primary menu.
    accels
        Keyboard shortcuts, e.g. ``["<Control>m"]``.
    """

    name: str
    callback: Callable[[], None]
    label: str | None = None
    accels: list[str] = field(default_factory=list)


class Host(Protocol):
    """Services a host window offers to plugins."""

    @property
    def window(self) -> Gtk.Window:
        """The host window (parent for dialogs)."""
        ...

    def toast(self, message: str) -> None:
        """Show a short notification."""
        ...

    def run_in_thread(
        self,
        work: Callable[[], T],
        on_done: Callable[[T], None],
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        """Run ``work`` off the main loop; callbacks run back on the main loop."""
        ...


class AppPlugin(Protocol):
    """What every app provides to a host (standalone window or viewer bundle)."""

    id: str
    title: str
    icon_name: str

    def header_widgets(self, host: Host) -> list[Gtk.Widget]:
        """Return the top-panel controls holding the app's key functionality."""
        ...

    def main_view(self, host: Host) -> Gtk.Widget:
        """Return the main content (graphs, results)."""
        ...

    def preferences(self, host: Host) -> list[Adw.PreferencesPage]:
        """Return the pages for the separate preferences window."""
        ...

    def actions(self, host: Host) -> list[Action]:
        """Return the actions for the primary menu and keyboard shortcuts."""
        ...


class StandaloneHost:
    """Single-plugin window: header bar (top panel), toasts, preferences, about.

    Parameters
    ----------
    app
        The running ``Adw.Application``.
    plugin
        The app to host.
    version
        App version shown in the About dialog.
    """

    def __init__(self, app: Adw.Application, plugin: AppPlugin, version: str) -> None:
        self.plugin = plugin
        self.version = version
        self._window = Adw.ApplicationWindow(application=app, title=plugin.title)
        self._window.set_default_size(1100, 700)
        self._toasts = Adw.ToastOverlay()

        header = Adw.HeaderBar()
        for widget in plugin.header_widgets(self):
            header.pack_start(widget)

        menu = Gio.Menu()
        app_section = Gio.Menu()
        for action in plugin.actions(self):
            self._add_action(app, action)
            if action.label:
                app_section.append(action.label, f"app.{action.name}")
        if app_section.get_n_items():
            menu.append_section(None, app_section)
        std = Gio.Menu()
        std.append("Preferences", "app.preferences")
        std.append(f"About {plugin.title}", "app.about")
        std.append("Quit", "app.quit")
        menu.append_section(None, std)
        self._add_action(
            app, Action("preferences", self.show_preferences, accels=["<Control>comma"])
        )
        self._add_action(app, Action("about", self.show_about))
        self._add_action(app, Action("quit", app.quit, accels=["<Control>q"]))
        menu_button = Gtk.MenuButton(icon_name="open-menu-symbolic", menu_model=menu)
        menu_button.set_tooltip_text("Main Menu")
        menu_button.set_primary(True)
        header.pack_end(menu_button)

        self._toasts.set_child(plugin.main_view(self))
        view = Adw.ToolbarView()
        view.add_top_bar(header)
        view.set_content(self._toasts)
        self._window.set_content(view)

    @property
    def window(self) -> Gtk.Window:
        """The host window."""
        return self._window

    def present(self) -> None:
        """Show the window."""
        self._window.present()

    def toast(self, message: str) -> None:
        """Show a short notification."""
        self._toasts.add_toast(Adw.Toast(title=GLib.markup_escape_text(message)))

    def run_in_thread(
        self,
        work: Callable[[], T],
        on_done: Callable[[T], None],
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        """Run ``work`` in a daemon thread; deliver the result on the GTK main loop."""

        def target() -> None:
            try:
                result = work()
            except BaseException as exc:  # noqa: BLE001 - reported to the UI, never lost
                traceback.print_exc()
                handler = on_error or (lambda e: self.toast(f"Error: {e}"))
                GLib.idle_add(_once(handler, exc))
                return
            GLib.idle_add(_once(on_done, result))

        threading.Thread(target=target, daemon=True).start()

    def show_preferences(self) -> None:
        """Open the separate preferences window."""
        dialog = Adw.PreferencesDialog()
        for page in self.plugin.preferences(self):
            dialog.add(page)
        dialog.present(self._window)

    def show_about(self) -> None:
        """Open the About dialog."""
        about = Adw.AboutDialog(application_name=self.plugin.title, version=self.version)
        about.set_application_icon(self.plugin.icon_name)
        about.set_developer_name("Re:sonus")
        about.present(self._window)

    def _add_action(self, app: Adw.Application, action: Action) -> None:
        simple = Gio.SimpleAction.new(action.name, None)
        simple.connect("activate", lambda *_: action.callback())
        app.add_action(simple)
        if action.accels:
            app.set_accels_for_action(f"app.{action.name}", action.accels)


def _once(fn: Callable[[Any], None], arg: Any) -> Callable[[], bool]:
    def call() -> bool:
        fn(arg)
        return GLib.SOURCE_REMOVE

    return call


def run_standalone(
    plugin_factory: Callable[[], AppPlugin],
    app_id: str,
    version: str,
    argv: list[str] | None = None,
) -> int:
    """Run one plugin as a standalone GNOME application.

    Parameters
    ----------
    plugin_factory
        Creates the plugin (called once the application is active).
    app_id
        Reverse-DNS application id, e.g. ``"cz.resonus.Room"``.
    version
        App version for the About dialog.
    argv
        Command-line arguments passed to GTK.

    Returns
    -------
    int
        Exit status.
    """
    app = Adw.Application(application_id=app_id, flags=Gio.ApplicationFlags.NON_UNIQUE)

    def on_activate(application: Adw.Application) -> None:
        StandaloneHost(application, plugin_factory(), version).present()

    app.connect("activate", on_activate)
    return int(app.run(argv or []))

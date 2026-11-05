"""Preferences groups "Inputs" / "Outputs": "add another" channel rows (GTK).

One row per channel: a channel drop-down, a label entry (``m#`` / ``s#``) and, from the
second row on, a remove button (cross); below the rows an "Add input" / "Add output"
button. The inputs group also holds the loopback reference channel (voluntary), the
outputs group the output mode (simultaneous / sequential). User decisions and requests
2026-11-04, resonus-room D-17. The rules are in
:mod:`resonus_core.channels`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from gi.repository import Adw, Gtk

from resonus_core.channels import (
    INPUT_PREFIX,
    OUTPUT_PREFIX,
    add_row,
    channel_limit,
    menu_channels,
    remove_row,
)
from resonus_core.devices import DeviceInfo
from resonus_core.measure import OUTPUT_MODES, MeasurementSettings

# Menu entry of the reference channel for "no loopback reference" (reference_channel 0).
NO_REFERENCE_LABEL = "None"


@dataclass
class ChannelRow:
    """Widgets of one channel row.

    Attributes
    ----------
    row : Adw.ActionRow
        The row, title ``f"Channel {k + 1}"`` (row ``k``, 0-based).
    channel : Gtk.DropDown
        Channel numbers (strings) of :func:`menu_channels`.
    label : Gtk.Entry
        The channel's label.
    remove : Gtk.Button or None
        Remove button (``window-close-symbolic``); None on the first row.
    """

    row: Adw.ActionRow
    channel: Gtk.DropDown
    label: Gtk.Entry
    remove: Gtk.Button | None


class ChannelGroup[S: MeasurementSettings]:
    """Channel rows of one direction of a :class:`MeasurementSettings` object.

    ``direction`` ``"input"``: title "Inputs", fields ``input_channels`` /
    ``input_labels``, prefix ``INPUT_PREFIX``, button "Add input", extra row
    ``reference_row``; ``"output"``: title "Outputs", ``output_channels`` /
    ``output_labels``, ``OUTPUT_PREFIX``, "Add output", extra row ``mode_row``.

    Layout (rebuilt by :meth:`refresh`): the channel rows (``self.rows``), then
    ``add_button`` (Adw.ButtonRow, ``start_icon_name="list-add-symbolic"``), then the
    extra row.

    - Channel row ``k``: ``Adw.ActionRow(title=f"Channel {k + 1}")``; prefix: a
      ``Gtk.DropDown`` of ``menu_channels(limit, channels)`` as strings, the row's
      channel selected; suffix: ``Gtk.Entry(text=label, width_chars=6)``; for ``k >= 1``
      a second suffix ``Gtk.Button(icon_name="window-close-symbolic")`` with tooltip
      "Remove".
    - ``limit = channel_limit(<direction>_device of the settings, direction, devices,
      default)``. ``add_button`` is sensitive iff a free channel exists
      (``len(channels) < limit``).
    - ``reference_row``: ``Adw.ComboRow(title="Reference channel")``, entries
      ``[NO_REFERENCE_LABEL] + [str(c) for c in menu_channels(limit,
      (*input_channels, reference_channel))]``; entry 0 = 0, else the number.
    - ``mode_row``: ``Adw.ComboRow(title="Several outputs")``, entries ``OUTPUT_MODES``.

    Changes (each accepted change calls ``on_change`` once; ``base = current()``):

    - Channel drop-down of row ``k``: channels with item ``k`` replaced.
    - Label entry of row ``k``, on ``"activate"`` (Enter): labels with item ``k``
      replaced by the entry text (no change if equal).
    - Remove button of row ``k``: ``remove_row(channels, labels, k)``; then
      :meth:`refresh`.
    - ``add_button`` ``"activated"``: ``add_row(channels, labels, prefix, limit)``; then
      :meth:`refresh`.
    - ``reference_row``: ``reference_channel``; ``mode_row``: ``output_mode``.

    The change is ``base.replace(<direction>_channels=..., <direction>_labels=...)`` (or
    the single field). A ``ValueError`` (from ``replace``, ``add_row`` or
    ``remove_row``) is passed to ``on_error`` (if given), nothing is emitted, and the
    widgets are rebuilt from ``current()`` (:meth:`refresh`), so they show the settings
    in force.

    Parameters
    ----------
    settings
        Settings when the page is built.
    on_change
        Called with the new settings after every accepted change.
    current
        Returns the latest settings (shared by all preference pages).
    direction
        ``"input"`` or ``"output"``.
    devices
        Device list (for the channel count of the selected device).
    default
        PortAudio's default device for this direction, or None if unknown.
    on_error
        Called with the message of a rejected change.

    Attributes
    ----------
    group : Adw.PreferencesGroup
    rows : list of ChannelRow
    add_button : Adw.ButtonRow
    reference_row : Adw.ComboRow or None
        Inputs only.
    mode_row : Adw.ComboRow or None
        Outputs only.

    Raises
    ------
    ValueError
        For an invalid ``direction``.
    """

    def __init__(
        self,
        settings: S,
        on_change: Callable[[S], None],
        current: Callable[[], S],
        direction: str,
        devices: Sequence[DeviceInfo],
        default: DeviceInfo | None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        if direction not in ("input", "output"):
            raise ValueError(f"direction must be 'input' or 'output', got {direction!r}")
        self.direction = direction
        self._on_change = on_change
        self._current = current
        self._devices = devices
        self._default = default
        self._on_error = on_error
        self._updating = False
        self._widgets: list[Gtk.Widget] = []
        self._retired: list[Gtk.Widget] = []
        self._reference_values: list[int] = []
        is_input = direction == "input"
        self._prefix = INPUT_PREFIX if is_input else OUTPUT_PREFIX
        self.group = Adw.PreferencesGroup(title="Inputs" if is_input else "Outputs")
        self.rows: list[ChannelRow] = []
        self.add_button = Adw.ButtonRow(
            title="Add input" if is_input else "Add output", start_icon_name="list-add-symbolic"
        )
        self.add_button.connect("activated", lambda _b: self._on_add())
        self.reference_row: Adw.ComboRow | None = None
        self.mode_row: Adw.ComboRow | None = None
        if is_input:
            self.reference_row = Adw.ComboRow(title="Reference channel")
            self.reference_row.connect("notify::selected", lambda *_a: self._on_reference())
        else:
            self.mode_row = Adw.ComboRow(title="Several outputs")
            self.mode_row.set_model(Gtk.StringList.new(list(OUTPUT_MODES)))
            self.mode_row.connect("notify::selected", lambda *_a: self._on_mode())
        self.refresh(settings)

    # public --------------------------------------------------------------------------

    def refresh(self, settings: S) -> None:
        """Rebuild all widgets from ``settings`` without emitting a change.

        Also called by the page after a device change (the channel count may differ).
        """
        self._updating = True
        try:
            for w in self._widgets:
                self.group.remove(w)
            # keep the previous rows alive: refresh may run inside a signal handler of
            # one of their widgets (a rejected change), which must not be freed there
            self._retired = self._widgets
            self._widgets = []
            channels, labels = self._get(settings)
            limit = self._limit(settings)
            menu = menu_channels(limit, channels)
            self.rows = [
                self._make_row(k, ch, lab, menu)
                for k, (ch, lab) in enumerate(zip(channels, labels, strict=True))
            ]
            self.add_button.set_sensitive(len(channels) < limit)
            extra: Gtk.Widget
            if self.reference_row is not None:
                ref = settings.reference_channel
                values = [0, *menu_channels(limit, (*settings.input_channels, ref))]
                if values != self._reference_values:
                    # a new model only when the entries change: replacing the model inside
                    # the row's own notify::selected handler crashes GTK
                    self._reference_values = values
                    self.reference_row.set_model(
                        Gtk.StringList.new([NO_REFERENCE_LABEL] + [str(v) for v in values[1:]])
                    )
                self.reference_row.set_selected(values.index(ref))
                extra = self.reference_row
            else:
                assert self.mode_row is not None
                self.mode_row.set_selected(OUTPUT_MODES.index(settings.output_mode))
                extra = self.mode_row
            for w in [*(r.row for r in self.rows), self.add_button, extra]:
                self.group.add(w)
                self._widgets.append(w)
        finally:
            self._updating = False

    # internals -----------------------------------------------------------------------

    def _get(self, settings: S) -> tuple[tuple[int, ...], tuple[str, ...]]:
        d = self.direction
        return getattr(settings, f"{d}_channels"), getattr(settings, f"{d}_labels")

    def _limit(self, settings: S) -> int:
        key = getattr(settings, f"{self.direction}_device")
        return channel_limit(key, self.direction, self._devices, self._default)

    def _make_row(self, k: int, channel: int, label: str, menu: list[int]) -> ChannelRow:
        row = Adw.ActionRow(title=f"Channel {k + 1}")
        drop = Gtk.DropDown.new_from_strings([str(c) for c in menu])
        drop.set_valign(Gtk.Align.CENTER)
        drop.set_selected(menu.index(channel))
        drop.connect("notify::selected", lambda d, _p: self._on_channel(k, menu[d.get_selected()]))
        row.add_prefix(drop)
        entry = Gtk.Entry(text=label, width_chars=6, valign=Gtk.Align.CENTER)
        entry.connect("activate", lambda e: self._on_label(k, e.get_text()))
        row.add_suffix(entry)
        remove = None
        if k >= 1:
            remove = Gtk.Button(icon_name="window-close-symbolic", valign=Gtk.Align.CENTER)
            remove.set_tooltip_text("Remove")
            remove.add_css_class("flat")
            remove.connect("clicked", lambda _b: self._on_remove(k))
            row.add_suffix(remove)
        return ChannelRow(row, drop, entry, remove)

    def _commit(self, rebuild: bool, **changes: Any) -> None:
        try:
            new = self._current().replace(**changes)
        except ValueError as exc:
            self._fail(exc)
            return
        self._on_change(new)
        if rebuild:
            self.refresh(new)

    def _fail(self, exc: ValueError) -> None:
        if self._on_error is not None:
            self._on_error(str(exc))
        self.refresh(self._current())

    def _set_pair(self, channels: tuple[int, ...], labels: tuple[str, ...], rebuild: bool) -> None:
        d = self.direction
        self._commit(rebuild, **{f"{d}_channels": channels, f"{d}_labels": labels})

    def _on_channel(self, k: int, value: int) -> None:
        if self._updating:
            return
        channels, labels = self._get(self._current())
        if channels[k] == value:
            return
        self._set_pair(tuple(value if i == k else c for i, c in enumerate(channels)), labels, False)

    def _on_label(self, k: int, text: str) -> None:
        if self._updating:
            return
        channels, labels = self._get(self._current())
        if labels[k] == text:
            return
        self._set_pair(channels, tuple(text if i == k else x for i, x in enumerate(labels)), False)

    def _on_remove(self, k: int) -> None:
        channels, labels = self._get(self._current())
        try:
            new_channels, new_labels = remove_row(channels, labels, k)
        except ValueError as exc:
            self._fail(exc)
            return
        self._set_pair(new_channels, new_labels, True)

    def _on_add(self) -> None:
        base = self._current()
        channels, labels = self._get(base)
        try:
            new_channels, new_labels = add_row(channels, labels, self._prefix, self._limit(base))
        except ValueError as exc:
            self._fail(exc)
            return
        self._set_pair(new_channels, new_labels, True)

    def _on_reference(self) -> None:
        if self._updating or self.reference_row is None:
            return
        value = self._reference_values[self.reference_row.get_selected()]
        if value != self._current().reference_channel:
            self._commit(False, reference_channel=value)

    def _on_mode(self) -> None:
        if self._updating or self.mode_row is None:
            return
        mode = OUTPUT_MODES[self.mode_row.get_selected()]
        if mode != self._current().output_mode:
            self._commit(False, output_mode=mode)

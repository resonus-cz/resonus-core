"""Preferences pages generated from a :class:`~resonus_core.settings.Settings` dataclass.

One source of truth: the same dataclass drives the CLI (TOML) and this GUI.
Widgets follow libadwaita's preferences rows (GNOME HIG).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields
from typing import Any

from gi.repository import Adw, Gtk

from resonus_core.settings import Settings


def settings_page[S: Settings](
    settings: S,
    on_change: Callable[[S], None],
    *,
    title: str = "General",
    icon_name: str = "preferences-system-symbolic",
    on_error: Callable[[str], None] | None = None,
    groups: Sequence[str] | None = None,
    current: Callable[[], S] | None = None,
) -> Adw.PreferencesPage:
    """Build a preferences page with one row per settings field, grouped by ``group``.

    Several pages can show different groups of one settings object (e.g. "Measurement"
    and "Analysis"): pass ``groups`` to select them and ``current`` so that every page
    starts each change from the latest settings, not from its own copy.

    Parameters
    ----------
    settings
        Current settings; the page edits a copy.
    on_change
        Called with the new, validated settings after every accepted change.
    title, icon_name
        Page title and icon.
    on_error
        Called with a message when a value is rejected (e.g. to show a toast).
    groups
        Groups to show, in this order; None: all groups in field order.
    current
        Returns the latest settings; each change is applied to ``current()``. None: the
        page keeps its own copy (fine for a single page).

    Raises
    ------
    ValueError
        ``f"unknown settings group: {name!r}"`` if a name in ``groups`` is not the group of
        any field shown on a page (tuple fields are not shown).

    Returns
    -------
    Adw.PreferencesPage
        The page, ready for an ``Adw.PreferencesDialog``.
    """
    # Determine which fields are shown
    shown = [
        f for f in fields(settings) if "kind" in f.metadata and f.metadata["kind"] is not tuple
    ]
    # Known groups in field order
    known = list(dict.fromkeys(f.metadata["group"] for f in shown))
    # Order to display
    order = known if groups is None else list(groups)
    # Validate group names
    for name in order:
        if name not in known:
            raise ValueError(f"unknown settings group: {name!r}")

    page = Adw.PreferencesPage(title=title, icon_name=icon_name)
    boxes: dict[str, Adw.PreferencesGroup] = {}
    state: dict[str, S] = {"settings": settings}

    # Create groups in the requested order
    for name in order:
        group = Adw.PreferencesGroup(title=name)
        boxes[name] = group
        page.add(group)

    def commit(name: str, value: Any, revert: Callable[[], None]) -> None:
        base = current() if current is not None else state["settings"]
        try:
            new = base.replace(**{name: value})
        except ValueError as exc:
            revert()
            if on_error is not None:
                on_error(str(exc))
            return
        if current is None:
            state["settings"] = new
        on_change(new)

    for f in shown:
        meta = f.metadata
        if meta["group"] in boxes:
            group = boxes[meta["group"]]
            group.add(_row(f.name, meta, getattr(settings, f.name), commit))
    return page


def _subtitle(meta: Mapping[str, Any]) -> str:
    parts = [p for p in (meta["help"], f"[{meta['unit']}]" if meta["unit"] else "") if p]
    return "  ".join(parts)


def _row(
    name: str,
    meta: Mapping[str, Any],
    value: Any,
    commit: Callable[[str, Any, Callable[[], None]], None],
) -> Gtk.Widget:
    kind = meta["kind"]
    if kind is bool:
        sw = Adw.SwitchRow(title=meta["label"], subtitle=_subtitle(meta), active=value)
        sw.connect("notify::active", lambda r, _p: commit(name, r.get_active(), lambda: None))
        return sw
    if kind is str and meta["choices"]:
        choices = list(meta["choices"])
        combo = Adw.ComboRow(title=meta["label"], subtitle=_subtitle(meta))
        combo.set_model(Gtk.StringList.new([str(c) for c in choices]))
        combo.set_selected(choices.index(value))
        combo.connect(
            "notify::selected", lambda r, _p: commit(name, choices[r.get_selected()], lambda: None)
        )
        return combo
    if kind in (int, float) and meta["choices"] is None:
        lo = meta["minimum"] if meta["minimum"] is not None else -1e12
        hi = meta["maximum"] if meta["maximum"] is not None else 1e12
        step = meta["step"] or (1 if kind is int else 0.1)
        adj = Gtk.Adjustment(
            value=value, lower=lo, upper=hi, step_increment=step, page_increment=10 * step
        )
        spin = Adw.SpinRow(title=meta["label"], subtitle=_subtitle(meta), adjustment=adj)
        spin.set_digits(0 if kind is int else meta["digits"])
        last = {"value": value}

        def on_value(r: Adw.SpinRow, _p: Any) -> None:
            v = int(round(r.get_value())) if kind is int else r.get_value()
            prev = last["value"]

            def revert() -> None:
                r.set_value(prev)

            last["value"] = v
            commit(name, v, revert)

        spin.connect("notify::value", on_value)
        return spin
    if kind in (int, float) and meta["choices"] is not None:
        choices = list(meta["choices"])
        combo = Adw.ComboRow(title=meta["label"], subtitle=_subtitle(meta))
        combo.set_model(Gtk.StringList.new([str(c) for c in choices]))
        combo.set_selected(choices.index(value))
        combo.connect(
            "notify::selected", lambda r, _p: commit(name, choices[r.get_selected()], lambda: None)
        )
        return combo
    entry = Adw.EntryRow(title=meta["label"], text=value, show_apply_button=True)
    entry.connect("apply", lambda r: commit(name, r.get_text(), lambda: None))
    return entry

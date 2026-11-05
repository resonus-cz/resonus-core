"""Typed, validated app settings shared by GUI, CLI and notebooks.

An app declares its settings once as a frozen dataclass whose fields are created with
:func:`param`. The same class is saved/loaded as TOML (CLI ``--settings``) and turned
into a preferences window by :mod:`resonus_core.gui.preferences`.
"""

from __future__ import annotations

import dataclasses
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Self

import tomli_w

_KINDS = (bool, int, float, str)
_ITEM_KINDS = (int, float, str)


def param(
    default: Any,
    *,
    label: str,
    unit: str = "",
    minimum: float | None = None,
    maximum: float | None = None,
    step: float | None = None,
    digits: int = 2,
    item: type | None = None,
    min_items: int = 0,
    unique: bool = False,
    choices: tuple[Any, ...] | None = None,
    group: str = "General",
    help: str = "",
) -> Any:
    """Declare a settings field with its presentation and validation metadata.

    Parameters
    ----------
    default
        Default value; its type (bool, int, float, str, or tuple) is the field's type.
        A tuple default declares a list-valued field (e.g. channel numbers); all its items
        have the item kind (int, float or str). Lists are not accepted as defaults
        (fields must be immutable).
    label
        Human-readable name shown in the preferences window.
    unit
        Unit shown next to the value, e.g. ``"Hz"``.
    minimum, maximum
        Inclusive numeric limits (for tuple fields: of every item).
    step
        Increment for spin buttons.
    digits
        Decimal places shown for float fields.
    choices
        Allowed values (any kind; shown as a drop-down; for tuple fields: of every item).
    group
        Preferences group the field is shown in.
    help
        One-line explanation shown as subtitle / CLI help.
    item
        Item kind of a tuple field (``int``, ``float`` or ``str``); inferred from the
        default's items if None. Required for an empty default.
    min_items
        Minimum number of items of a tuple field.
    unique
        Tuple field: items must be distinct.

    Returns
    -------
    dataclasses.Field
        Field to assign in a :class:`Settings` subclass. Metadata keys: ``label``,
        ``unit``, ``minimum``, ``maximum``, ``step``, ``digits``, ``choices``, ``group``,
        ``help``, ``kind`` (type of the default), ``item`` (item kind for tuples, else
        None), ``min_items``, ``unique``.

    Raises
    ------
    TypeError
        On an unsupported default type; for a tuple default: items of more than one kind
        (int items are allowed in a float tuple and converted), an item kind other than
        int/float/str, items not of ``item``, or an empty default without ``item``.
    """
    if type(default) is tuple:
        kinds = {type(x) for x in default}
        if item is None:
            if len(kinds) != 1:
                raise TypeError("tuple default must have items of a single kind")
            item = kinds.pop()
        if item not in _ITEM_KINDS:
            raise TypeError(f"unsupported tuple item type: {item.__name__}")
        allowed = {float, int} if item is float else {item}
        if not kinds.issubset(allowed):
            raise TypeError("tuple items must be of the declared item kind")
        if item is float:
            default = tuple(float(x) for x in default)
    elif type(default) not in _KINDS:
        raise TypeError(f"unsupported settings type: {type(default).__name__}")
    meta = {
        "label": label,
        "unit": unit,
        "minimum": minimum,
        "maximum": maximum,
        "step": step,
        "digits": digits,
        "choices": choices,
        "group": group,
        "help": help,
        "kind": type(default),
        "item": item,
        "min_items": min_items,
        "unique": unique,
    }
    return field(default=default, metadata=meta)


@dataclass(frozen=True)
class Settings:
    """Base class for app settings: validation, TOML persistence, immutable updates."""

    def __post_init__(self) -> None:
        """Validate every field against its declared type and limits.

        Tuple fields: a list value is converted to a tuple (TOML arrays); int items of a
        float tuple are converted to float; then every item is checked for its kind
        (``bool`` is not an ``int``), ``minimum``, ``maximum`` and ``choices``, and the
        tuple for ``min_items`` and ``unique``. Messages: ``"{name}: expected tuple, got
        {type}"``, ``"{name}: expected {item}, got {type}"``, ``"{name}: at least
        {min_items} items required"``, ``"{name}: duplicate items"``; limit and choice
        messages as for scalar fields.
        """
        for f in fields(self):
            meta = f.metadata
            if "kind" not in meta:
                continue
            value = getattr(self, f.name)
            kind = meta["kind"]
            if kind is tuple:
                if isinstance(value, list):
                    value = tuple(value)
                if not isinstance(value, tuple):
                    raise ValueError(f"{f.name}: expected tuple, got {type(value).__name__}")
                item = meta["item"]
                # convert int items to float if needed
                if item is float:
                    value = tuple(float(x) if type(x) is int else x for x in value)
                # check each item
                for x in value:
                    if type(x) is not item:
                        raise ValueError(
                            f"{f.name}: expected {item.__name__}, got {type(x).__name__}"
                        )
                    self._check(f.name, meta, x)
                if len(value) < meta["min_items"]:
                    raise ValueError(f"{f.name}: at least {meta['min_items']} items required")
                if meta["unique"] and len(set(value)) != len(value):
                    raise ValueError(f"{f.name}: duplicate items")
                object.__setattr__(self, f.name, value)
                continue
            if kind is float and type(value) is int:
                object.__setattr__(self, f.name, float(value))
                value = float(value)
            if type(value) is not kind:
                raise ValueError(f"{f.name}: expected {kind.__name__}, got {type(value).__name__}")
            self._check(f.name, meta, value)

    @staticmethod
    def _check(name: str, meta: Mapping[str, Any], value: Any) -> None:
        if meta["minimum"] is not None and value < meta["minimum"]:
            raise ValueError(f"{name}: {value} is below the minimum {meta['minimum']}")
        if meta["maximum"] is not None and value > meta["maximum"]:
            raise ValueError(f"{name}: {value} is above the maximum {meta['maximum']}")
        if meta["choices"] is not None and value not in meta["choices"]:
            raise ValueError(f"{name}: {value!r} is not one of {meta['choices']}")

    def replace(self, **changes: Any) -> Self:
        """Return a validated copy with some fields changed.

        Raises
        ------
        ValueError
            If a changed value violates its field's type or limits.
        """
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Return the settings as a plain dict."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create settings from a dict; missing keys keep their defaults.

        Raises
        ------
        ValueError
            On unknown keys or invalid values.
        """
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown settings: {', '.join(sorted(unknown))}")
        return cls(**data)

    def save(self, path: str | Path) -> None:
        """Write the settings to a TOML file."""
        Path(path).write_text(tomli_w.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Read settings from a TOML file.

        Raises
        ------
        ValueError
            On unknown keys or invalid values.
        """
        with Path(path).open("rb") as fh:
            return cls.from_dict(tomllib.load(fh))

"""Unified plot style: one set of tokens for every renderer (matplotlib now, GL widget later).

The tokens live in the data file ``assets/style_tokens.toml`` (D-T1), which the Rust apps
share; this module reads it at import. Fonts are the bundled static Lexend files (SIL OFL
1.1, see ``assets/fonts/lexend``); nothing has to be installed system-wide.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Final, Literal, cast

FONT_DIR = Path(str(files("resonus_core") / "assets" / "fonts" / "lexend"))
# The style tokens data file shared with the Rust apps (D-T1).
STYLE_FILE: Final = Path(str(files("resonus_core") / "assets" / "style_tokens.toml"))
STYLE_SCHEMA_VERSION: Final = 1

# Required keys of the [tokens] table and their kind: "size" (a TOML float or integer,
# returned as float, > 0), "color" ("#rrggbb", case-insensitive, returned lower-case) or
# "text" (a string).
TOKEN_KINDS: Final[dict[str, str]] = {
    "font_size": "size",
    "title_size": "size",
    "line_width": "size",
    "grid_color": "color",
    "grid_width": "size",
    "axes_color": "color",
    "background": "color",
    "brand_text": "text",
    "brand_weight": "text",
    "brand_size": "size",
    "brand_color": "color",
    "neutral_color": "color",
}
COLOR_PATTERN: Final = re.compile(r"#[0-9a-fA-F]{6}")


@dataclass(frozen=True)
class StyleTokens:
    """Contents of a style tokens file.

    Parameters
    ----------
    font_family
        Font family name (``"Lexend"``).
    trace_colors
        Trace colours in order, ``"#rrggbb"`` lower-case (at least one).
    tokens
        The ``[tokens]`` table: exactly the keys of ``TOKEN_KINDS``, in that order.
    """

    font_family: str
    trace_colors: tuple[str, ...]
    tokens: dict[str, Any]


def load_tokens(
    path: str | Path = STYLE_FILE, variant: Literal["light", "dark"] = "light"
) -> StyleTokens:
    """Read and check a style tokens file (D-T1), light or dark variant.

    1. ``tomllib`` parse of the UTF-8 text (errors pass through: ``OSError``,
       ``tomllib.TOMLDecodeError``).
    2. ``version`` (required, an int): ``> STYLE_SCHEMA_VERSION`` raises.
    3. ``font_family``: a non-empty string. ``trace_colors``: a non-empty list of colours.
    4. ``[tokens]``: every key of ``TOKEN_KINDS`` with a value of its kind; unknown keys,
       in the table and at the top level, are ignored.
    5. ``variant="dark"``: the optional ``[dark]`` table (missing: the light result).
       ``dark.trace_colors``, if present, replaces ``trace_colors`` (same checks, key
       ``"dark.trace_colors[i]"``, empty list or not a list: ``"style tokens:
       dark.trace_colors must be a non-empty list"``). Each key of ``[dark.tokens]`` that is
       in ``TOKEN_KINDS`` replaces the light value (same checks, key
       ``"dark.tokens.<key>"``); other keys are ignored. The light file is fully checked
       first, also for the dark variant. Any other ``variant``: ``ValueError(f"style
       tokens: unknown variant {variant!r}")``, before reading the file.

    A colour is a string that ``COLOR_PATTERN.fullmatch`` accepts; returned lower-case.
    A size is an ``int`` or ``float`` but not a ``bool``, ``> 0``; returned as ``float``.

    Raises
    ------
    ValueError
        Messages (``key`` is the dotted name, e.g. ``"tokens.font_size"``,
        ``"trace_colors[2]"``): ``f"style tokens: missing {key}"``; ``f"style tokens:
        version {v} is newer than {STYLE_SCHEMA_VERSION}"``; ``f"style tokens: {key} must
        be a colour '#rrggbb'"``; ``f"style tokens: {key} must be a positive number"``;
        ``f"style tokens: {key} must be a string"`` (also for an empty ``font_family`` and
        a non-int ``version``: ``"must be an integer"``); ``"style tokens: trace_colors must
        be a non-empty list"``.
    """
    if variant not in ("light", "dark"):
        raise ValueError(f"style tokens: unknown variant {variant!r}")
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))

    # 1. version
    if "version" not in data:
        raise ValueError("style tokens: missing version")
    version = data["version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("style tokens: version must be an integer")
    if version > STYLE_SCHEMA_VERSION:
        raise ValueError(f"style tokens: version {version} is newer than {STYLE_SCHEMA_VERSION}")

    # 2. font_family
    if "font_family" not in data:
        raise ValueError("style tokens: missing font_family")
    font_family = data["font_family"]
    if not isinstance(font_family, str) or not font_family:
        raise ValueError("style tokens: font_family must be a string")

    # 3. trace_colors
    if "trace_colors" not in data:
        raise ValueError("style tokens: missing trace_colors")
    trace_colors = _check_colors(data["trace_colors"], "trace_colors")

    # 4. tokens
    table = data.get("tokens", {})
    if not isinstance(table, dict):
        table = {}
    tokens: dict[str, Any] = {}
    for key, kind in TOKEN_KINDS.items():
        if key not in table:
            raise ValueError(f"style tokens: missing tokens.{key}")
        tokens[key] = _check_token(table[key], kind, f"tokens.{key}")
    if variant == "light":
        return StyleTokens(font_family, trace_colors, tokens)

    # 5. dark overrides
    dark = data.get("dark", {})
    if not isinstance(dark, dict):
        dark = {}
    if "trace_colors" in dark:
        trace_colors = _check_colors(dark["trace_colors"], "dark.trace_colors")
    dark_table = dark.get("tokens", {})
    if isinstance(dark_table, dict):
        for key, kind in TOKEN_KINDS.items():
            if key in dark_table:
                tokens[key] = _check_token(dark_table[key], kind, f"dark.tokens.{key}")
    return StyleTokens(font_family, trace_colors, tokens)


def _check_colors(raw: Any, name: str) -> tuple[str, ...]:
    """Return the checked, lower-case colours of a non-empty list (``name`` for messages)."""
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"style tokens: {name} must be a non-empty list")
    return tuple(_check_token(c, "color", f"{name}[{i}]") for i, c in enumerate(raw))


def _check_token(value: Any, kind: str, name: str) -> Any:
    """Return a checked token value of ``kind`` (see ``TOKEN_KINDS``)."""
    if kind == "size":
        if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
            raise ValueError(f"style tokens: {name} must be a positive number")
        return float(value)
    if kind == "color":
        if not isinstance(value, str) or not COLOR_PATTERN.fullmatch(value):
            raise ValueError(f"style tokens: {name} must be a colour '#rrggbb'")
        return value.lower()
    if not isinstance(value, str):
        raise ValueError(f"style tokens: {name} must be a string")
    return value


# Module tokens, from STYLE_FILE. FONT_FAMILY, TRACE_COLORS and TOKENS: the light style.
LIGHT = load_tokens()
DARK = load_tokens(variant="dark")
FONT_FAMILY = LIGHT.font_family
TRACE_COLORS = LIGHT.trace_colors
TOKENS: dict[str, Any] = LIGHT.tokens

_registered = False


def font_files() -> list[Path]:
    """Return the bundled Lexend font files."""
    return sorted(FONT_DIR.glob("Lexend-*.ttf"))


def register_fonts() -> None:
    """Make the bundled Lexend fonts available to matplotlib (idempotent)."""
    global _registered
    if _registered:
        return
    from matplotlib import font_manager

    for path in font_files():
        font_manager.fontManager.addfont(str(path))
    _registered = True


def mpl_rc(dark: bool = False) -> dict[str, Any]:
    """Return matplotlib rcParams implementing the style tokens.

    Parameters
    ----------
    dark
        Use the dark variant (``DARK``: its tokens and trace colours) instead of ``LIGHT``.
    """
    from cycler import cycler

    s = DARK if dark else LIGHT
    t = s.tokens
    return {
        "font.family": s.font_family,
        "font.size": t["font_size"],
        "axes.titlesize": t["title_size"],
        "axes.titleweight": "medium",
        "axes.labelsize": t["font_size"],
        "axes.edgecolor": t["axes_color"],
        "axes.labelcolor": t["axes_color"],
        "axes.titlecolor": t["axes_color"],
        "text.color": t["axes_color"],
        "legend.labelcolor": t["axes_color"],
        "axes.prop_cycle": cycler(color=list(s.trace_colors)),
        "axes.grid": True,
        "axes.grid.which": "both",
        "grid.color": t["grid_color"],
        "grid.linewidth": t["grid_width"],
        "xtick.color": t["axes_color"],
        "ytick.color": t["axes_color"],
        "lines.linewidth": t["line_width"],
        "figure.facecolor": t["background"],
        "axes.facecolor": t["background"],
        "savefig.dpi": 200,
        "legend.frameon": False,
    }


def apply_mpl_style() -> None:
    """Register the fonts and apply the style globally to matplotlib."""
    import matplotlib as mpl

    register_fonts()
    mpl.rcParams.update(cast(Any, mpl_rc()))


def add_brand(fig: Any, text: str | None = None, dark: bool = False) -> Any:
    """Place the brand mark in the bottom-right corner of a matplotlib figure.

    ``text=None`` uses ``TOKENS["brand_text"]`` ("Re:sonus"); any other text replaces it
    (user request 2026-11-05, resonus-room D-24: the mark is user-replaceable). An empty
    text adds an invisible mark (``set_visible(False)``), so it can be set later.

    Parameters
    ----------
    fig
        matplotlib Figure.
    text
        Text of the mark; None for the default.
    dark
        Use the dark style's ``brand_color``.

    Returns
    -------
    matplotlib.text.Text
        The added text artist (gid ``"resonus-brand"``).
    """
    register_fonts()
    t = (DARK if dark else LIGHT).tokens
    label = t["brand_text"] if text is None else text
    artist = fig.text(
        0.995,
        0.005,
        label,
        ha="right",
        va="bottom",
        fontfamily=FONT_FAMILY,
        fontweight=t["brand_weight"],
        fontsize=t["brand_size"],
        color=t["brand_color"],
        gid="resonus-brand",
    )
    artist.set_visible(bool(label))
    return artist

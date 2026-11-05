"""Unified plot style: one set of tokens for every renderer (matplotlib now, GL widget later).

Fonts are the bundled static Lexend files (SIL OFL 1.1, see ``assets/fonts/lexend``);
nothing has to be installed system-wide.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any, cast

FONT_DIR = Path(str(files("resonus_core") / "assets" / "fonts" / "lexend"))
FONT_FAMILY = "Lexend"

# Trace colors: GNOME HIG palette (https://developer.gnome.org/hig/reference/palette.html):
# Blue 3, Red 3, Green 5, Orange 3, Purple 3, Brown 3, Yellow 5, Dark 3.
# Values transcribed from memory, to be verified against the palette page.
TRACE_COLORS = (
    "#3584e4",
    "#e01b24",
    "#26a269",
    "#ff7800",
    "#9141ac",
    "#986a44",
    "#e5a50a",
    "#3d3846",
)

TOKENS: dict[str, Any] = {
    "font_size": 10.0,
    "title_size": 12.0,
    "line_width": 1.4,
    "grid_color": "#d0d0d0",
    "grid_width": 0.6,
    "axes_color": "#3d3846",
    "background": "#ffffff",
    "brand_text": "Re:sonus",
    "brand_weight": "semibold",  # Lexend SemiBold (600), decision D1
    "brand_size": 9.0,
    "brand_color": "#5e5c64",
}

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


def mpl_rc() -> dict[str, Any]:
    """Return matplotlib rcParams implementing the style tokens."""
    from cycler import cycler

    t = TOKENS
    return {
        "font.family": FONT_FAMILY,
        "font.size": t["font_size"],
        "axes.titlesize": t["title_size"],
        "axes.titleweight": "medium",
        "axes.labelsize": t["font_size"],
        "axes.edgecolor": t["axes_color"],
        "axes.labelcolor": t["axes_color"],
        "axes.prop_cycle": cycler(color=list(TRACE_COLORS)),
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


def add_brand(fig: Any, text: str | None = None) -> Any:
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

    Returns
    -------
    matplotlib.text.Text
        The added text artist (gid ``"resonus-brand"``).
    """
    register_fonts()
    t = TOKENS
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

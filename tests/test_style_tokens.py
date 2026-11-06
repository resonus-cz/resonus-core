"""Style tokens data file shared with the Rust apps (D-T1)."""

import tomllib

import pytest

from resonus_core import style
from resonus_core.style import (
    STYLE_FILE,
    STYLE_SCHEMA_VERSION,
    TOKEN_KINDS,
    StyleTokens,
    load_tokens,
)

# Light: the 0.9.0 values, except Orange 3 -> Orange 4, Dark 3 out of the traces into
# neutral_color, Yellow 5 last (user, 2026-11-05, D-T1).
EXPECTED_TOKENS = {
    "font_size": 10.0,
    "title_size": 12.0,
    "line_width": 1.4,
    "grid_color": "#d0d0d0",
    "grid_width": 0.6,
    "axes_color": "#3d3846",
    "background": "#ffffff",
    "brand_text": "Re:sonus",
    "brand_weight": "semibold",
    "brand_size": 9.0,
    "brand_color": "#5e5c64",
    "neutral_color": "#3d3846",
}
EXPECTED_COLORS = (
    "#3584e4", "#e01b24", "#26a269", "#e66100", "#9141ac", "#986a44", "#e5a50a",
)  # fmt: skip
# Dark (user, 2026-11-05, D-T1): Purple 2 for Purple 3; greys and background.
EXPECTED_DARK_TOKENS = {
    **EXPECTED_TOKENS,
    "grid_color": "#3d3846",
    "axes_color": "#deddda",
    "background": "#1d1d20",
    "brand_color": "#9a9996",
    "neutral_color": "#deddda",
}
EXPECTED_DARK_COLORS = (
    "#3584e4", "#e01b24", "#26a269", "#e66100", "#c061cb", "#986a44", "#e5a50a",
)  # fmt: skip


def srgb_luminance(color):
    # WCAG 2.1 relative luminance (sRGB)
    c = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def contrast(a, b):
    hi, lo = sorted([srgb_luminance(a), srgb_luminance(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


MINIMAL = """version = 1
font_family = "Lexend"
trace_colors = ["#3584E4", "#e01b24"]

[tokens]
font_size = 10
title_size = 12.0
line_width = 1.4
grid_color = "#D0D0D0"
grid_width = 0.6
axes_color = "#3d3846"
background = "#ffffff"
brand_text = "Re:sonus"
brand_weight = "semibold"
brand_size = 9.0
brand_color = "#5e5c64"
neutral_color = "#3d3846"
"""


def write(tmp_path, text):
    path = tmp_path / "tokens.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_bundled_file():
    assert STYLE_FILE.name == "style_tokens.toml" and STYLE_FILE.is_file()
    data = tomllib.loads(STYLE_FILE.read_text(encoding="utf-8"))
    assert data["version"] == STYLE_SCHEMA_VERSION == 1
    t = load_tokens()
    assert isinstance(t, StyleTokens)
    assert t == load_tokens(STYLE_FILE) == load_tokens(str(STYLE_FILE))
    assert t.font_family == "Lexend"
    assert t.trace_colors == EXPECTED_COLORS
    assert t.tokens == EXPECTED_TOKENS
    assert list(t.tokens) == list(TOKEN_KINDS)


def test_bundled_dark_variant():
    d = load_tokens(variant="dark")
    assert d.font_family == "Lexend"
    assert d.trace_colors == EXPECTED_DARK_COLORS
    assert d.tokens == EXPECTED_DARK_TOKENS and list(d.tokens) == list(TOKEN_KINDS)
    assert load_tokens(STYLE_FILE, "light") == load_tokens()


def test_bundled_contrast_rule():
    # D-T1: every trace >= 3:1 on its background (WCAG 2.1 SC 1.4.11), except the last
    # light trace (Yellow 5); the neutral colour too
    for t in (load_tokens(), load_tokens(variant="dark")):
        bg = t.tokens["background"]
        ratios = [contrast(c, bg) for c in t.trace_colors + (t.tokens["neutral_color"],)]
        assert all(r >= 3.0 for r in ratios[:-2] + ratios[-1:])
    assert contrast("#e5a50a", "#ffffff") < 3.0  # why yellow is last


def test_module_tokens_come_from_the_file():
    t = load_tokens()
    assert style.LIGHT == t and style.DARK == load_tokens(variant="dark")
    assert style.FONT_FAMILY == t.font_family
    assert style.TRACE_COLORS == t.trace_colors and isinstance(style.TRACE_COLORS, tuple)
    assert style.TOKENS == t.tokens
    rc = style.mpl_rc()
    assert rc == style.mpl_rc(dark=False)
    assert rc["font.size"] == 10.0 and rc["grid.color"] == "#d0d0d0"
    assert rc["axes.prop_cycle"].by_key()["color"] == list(EXPECTED_COLORS)
    dark = style.mpl_rc(dark=True)
    assert dark["figure.facecolor"] == dark["axes.facecolor"] == "#1d1d20"
    assert dark["axes.edgecolor"] == dark["xtick.color"] == "#deddda"
    assert dark["grid.color"] == "#3d3846" and dark["font.size"] == 10.0
    assert dark["axes.prop_cycle"].by_key()["color"] == list(EXPECTED_DARK_COLORS)
    assert set(dark) == set(rc)
    for key in ("axes.titlecolor", "text.color", "legend.labelcolor", "axes.labelcolor"):
        assert rc[key] == "#3d3846" and dark[key] == "#deddda"


def test_brand_colour_per_style():
    import matplotlib.pyplot as plt

    fig = plt.figure()
    try:
        assert style.add_brand(fig).get_color() == "#5e5c64"
        assert style.add_brand(fig, dark=True).get_color() == "#9a9996"
    finally:
        plt.close(fig)


def test_dark_variant_overrides(tmp_path):
    path = write(tmp_path, MINIMAL)
    assert load_tokens(path, "dark") == load_tokens(path)  # no [dark] table
    text = MINIMAL + '\n[dark]\ntrace_colors = ["#C061CB"]\nfuture = 1\n'
    text += '[dark.tokens]\nbackground = "#1D1D20"\nline_width = 2\nnew_token = "x"\n'
    d = load_tokens(write(tmp_path, text), "dark")
    assert d.trace_colors == ("#c061cb",)
    assert d.tokens["background"] == "#1d1d20" and d.tokens["line_width"] == 2.0
    assert d.tokens["grid_color"] == "#d0d0d0" and list(d.tokens) == list(TOKEN_KINDS)
    light = load_tokens(write(tmp_path, text))
    assert light.tokens["background"] == "#ffffff"
    only_tokens = MINIMAL + '\n[dark.tokens]\naxes_color = "#deddda"\n'
    d = load_tokens(write(tmp_path, only_tokens), "dark")
    assert d.trace_colors == ("#3584e4", "#e01b24") and d.tokens["axes_color"] == "#deddda"


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ('[dark]\ntrace_colors = []\n', "style tokens: dark.trace_colors must be a non-empty list"),
        ('[dark]\ntrace_colors = ["#ffffff", "x"]\n',
         "style tokens: dark.trace_colors[1] must be a colour '#rrggbb'"),
        ('[dark.tokens]\nbackground = "black"\n',
         "style tokens: dark.tokens.background must be a colour '#rrggbb'"),
        ('[dark.tokens]\nfont_size = -1\n',
         "style tokens: dark.tokens.font_size must be a positive number"),
        ('[dark.tokens]\nbrand_text = 3\n',
         "style tokens: dark.tokens.brand_text must be a string"),
    ],
)  # fmt: skip
def test_dark_invalid(tmp_path, extra, message):
    path = write(tmp_path, MINIMAL + "\n" + extra)
    load_tokens(path)  # the light variant does not look at [dark]
    with pytest.raises(ValueError) as info:
        load_tokens(path, "dark")
    assert str(info.value) == message


def test_dark_checks_the_light_file_first(tmp_path):
    text = MINIMAL.replace("font_size = 10\n", "") + "\n[dark.tokens]\nfont_size = 10\n"
    with pytest.raises(ValueError, match="style tokens: missing tokens.font_size"):
        load_tokens(write(tmp_path, text), "dark")


def test_unknown_variant(tmp_path):
    with pytest.raises(ValueError, match="style tokens: unknown variant 'sepia'"):
        load_tokens(tmp_path / "missing.toml", "sepia")


def test_minimal_file_normalised(tmp_path):
    t = load_tokens(write(tmp_path, MINIMAL))
    assert t.trace_colors == ("#3584e4", "#e01b24")
    assert t.tokens["grid_color"] == "#d0d0d0"
    assert t.tokens["font_size"] == 10.0 and isinstance(t.tokens["font_size"], float)
    assert list(t.tokens) == list(TOKEN_KINDS)


def test_unknown_keys_ignored(tmp_path):
    text = MINIMAL.replace("[tokens]\n", 'future = 1\n[tokens]\nnew_token = "x"\n')
    text += '\n[sepia]\nbackground = "#704214"\n'
    t = load_tokens(write(tmp_path, text))
    assert "new_token" not in t.tokens and list(t.tokens) == list(TOKEN_KINDS)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("version = 1\n", "version = 2\n", "style tokens: version 2 is newer than 1"),
        ("version = 1\n", "", "style tokens: missing version"),
        ("version = 1\n", 'version = "1"\n', "style tokens: version must be an integer"),
        ("version = 1\n", "version = true\n", "style tokens: version must be an integer"),
        ('font_family = "Lexend"\n', "", "style tokens: missing font_family"),
        ('font_family = "Lexend"\n', 'font_family = ""\n',
         "style tokens: font_family must be a string"),
        ('trace_colors = ["#3584E4", "#e01b24"]\n', "", "style tokens: missing trace_colors"),
        ('trace_colors = ["#3584E4", "#e01b24"]\n', "trace_colors = []\n",
         "style tokens: trace_colors must be a non-empty list"),
        ('trace_colors = ["#3584E4", "#e01b24"]\n', 'trace_colors = "#3584e4"\n',
         "style tokens: trace_colors must be a non-empty list"),
        ('"#e01b24"]', '"red"]', "style tokens: trace_colors[1] must be a colour '#rrggbb'"),
        ("font_size = 10\n", "", "style tokens: missing tokens.font_size"),
        ("font_size = 10\n", "font_size = 0\n",
         "style tokens: tokens.font_size must be a positive number"),
        ("font_size = 10\n", "font_size = true\n",
         "style tokens: tokens.font_size must be a positive number"),
        ("font_size = 10\n", 'font_size = "10"\n',
         "style tokens: tokens.font_size must be a positive number"),
        ('grid_color = "#D0D0D0"\n', 'grid_color = "#d0d0d"\n',
         "style tokens: tokens.grid_color must be a colour '#rrggbb'"),
        ('grid_color = "#D0D0D0"\n', "grid_color = 3\n",
         "style tokens: tokens.grid_color must be a colour '#rrggbb'"),
        ('brand_text = "Re:sonus"\n', "brand_text = 1\n",
         "style tokens: tokens.brand_text must be a string"),
    ],
)  # fmt: skip
def test_invalid(tmp_path, old, new, message):
    assert old in MINIMAL
    with pytest.raises(ValueError) as info:
        load_tokens(write(tmp_path, MINIMAL.replace(old, new, 1)))
    assert str(info.value) == message


def test_missing_tokens_table(tmp_path):
    text = MINIMAL[: MINIMAL.index("[tokens]")]
    with pytest.raises(ValueError, match="style tokens: missing tokens.font_size"):
        load_tokens(write(tmp_path, text))


def test_parse_and_read_errors_pass_through(tmp_path):
    with pytest.raises(tomllib.TOMLDecodeError):
        load_tokens(write(tmp_path, "version = = 1"))
    with pytest.raises(FileNotFoundError):
        load_tokens(tmp_path / "missing.toml")

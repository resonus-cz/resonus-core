import numpy as np
import pytest
from matplotlib import font_manager

from resonus_core import style
from resonus_core.plot import MplPlotView


def test_fonts_bundled_and_registered():
    names = {p.name for p in style.font_files()}
    assert {"Lexend-Regular.ttf", "Lexend-SemiBold.ttf"} <= names
    style.register_fonts()
    path = font_manager.findfont(
        font_manager.FontProperties(family="Lexend", weight="semibold"), fallback_to_default=False
    )
    assert path.endswith("Lexend-SemiBold.ttf")


def test_brand_mark():
    view = MplPlotView()
    texts = [t for t in view.figure.texts if t.get_gid() == "resonus-brand"]
    assert len(texts) == 1
    t = texts[0]
    assert t.get_text() == "Re:sonus"
    assert t.get_fontweight() == "semibold"
    assert t.get_position() == (0.995, 0.005) and t.get_ha() == "right"


def test_traces_and_export(tmp_path):
    view = MplPlotView()
    view.set_axes(xlabel="Frequency [Hz]", ylabel="Level [dB]", xscale="log", xlim=(20, 20000))
    f = np.logspace(np.log10(20), np.log10(20000), 50)
    view.set_trace("a", f, np.zeros(50), label="A")
    view.set_trace("b", f, np.ones(50), label="B")
    view.set_trace("a", f, 2 * np.ones(50), label="A2")
    assert view.trace_ids() == ["a", "b"]
    view.remove_trace("b")
    view.remove_trace("unknown")
    assert view.trace_ids() == ["a"]
    assert view.axes.get_xscale() == "log"
    view.save(str(tmp_path / "p.png"))
    assert (tmp_path / "p.png").stat().st_size > 1000
    with pytest.raises(ValueError):
        view.set_trace("c", [1, 2], [1], label="bad")


def test_jupyter_png_repr():
    png = MplPlotView()._repr_png_()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"

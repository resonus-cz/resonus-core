"""User-replaceable brand mark (resonus-room D-24)."""

import matplotlib.pyplot as plt
import pytest

from resonus_core.plot import MplPlotView
from resonus_core.style import TOKENS, add_brand


def marks(view):
    return [t for t in view.figure.texts if t.get_gid() == "resonus-brand"]


def test_add_brand_default_and_custom():
    fig = plt.figure()
    a = add_brand(fig)
    b = add_brand(fig, "Acme Acoustics")
    c = add_brand(fig, "")
    assert a.get_text() == TOKENS["brand_text"] == "Re:sonus" and a.get_visible()
    assert b.get_text() == "Acme Acoustics" and b.get_visible()
    assert b.get_fontweight() == "semibold" and b.get_position() == (0.995, 0.005)
    assert c.get_text() == "" and not c.get_visible()
    assert all(t.get_gid() == "resonus-brand" for t in (a, b, c))
    plt.close(fig)


@pytest.mark.parametrize(
    ("text", "shown", "visible"),
    [(None, "Re:sonus", True), ("Ing. Novák", "Ing. Novák", True), ("", "", False)],
)
def test_view_brand_text(text, shown, visible):
    view = MplPlotView(brand_text=text)
    (mark,) = marks(view)
    assert mark.get_text() == shown and mark.get_visible() is visible


def test_set_brand():
    view = MplPlotView()
    view.set_brand("Studio X")
    (mark,) = marks(view)
    assert mark.get_text() == "Studio X" and mark.get_visible()
    view.set_brand("")
    assert len(marks(view)) == 1 and not mark.get_visible()
    view.set_brand(None)
    assert mark.get_text() == "Re:sonus" and mark.get_visible()


def test_set_brand_on_panel_and_without_mark():
    view = MplPlotView()
    panel = view.add_panel((2, 1))
    panel.set_brand("Panel call")  # goes to the figure's mark
    assert [t.get_text() for t in marks(view)] == ["Panel call"]
    plain = MplPlotView(brand=False)
    plain.set_brand("ignored")
    assert marks(plain) == []
    assert marks(MplPlotView(brand=False, brand_text="x")) == []


def test_exported_image_has_the_text(tmp_path):
    view = MplPlotView(brand_text="Custom Mark")
    view.save(str(tmp_path / "a.svg"))
    assert "Custom Mark" in (tmp_path / "a.svg").read_text()
    view.set_brand("")
    view.save(str(tmp_path / "b.svg"))
    assert "Custom Mark" not in (tmp_path / "b.svg").read_text()

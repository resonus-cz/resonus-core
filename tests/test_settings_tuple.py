from dataclasses import dataclass, fields

import pytest

from resonus_core.settings import Settings, param


@dataclass(frozen=True)
class Multi(Settings):
    channels: tuple[int, ...] = param(
        (1,), label="Channels", minimum=1, maximum=64, min_items=1, unique=True
    )
    labels: tuple[str, ...] = param(("m1",), label="Labels")
    gains: tuple[float, ...] = param((), label="Gains", item=float, minimum=-20.0)
    rate: int = param(48000, label="Sample rate", choices=(44100, 48000, 96000))


def test_defaults_and_metadata():
    s = Multi()
    assert s.channels == (1,) and s.labels == ("m1",) and s.gains == ()
    meta = {f.name: f.metadata for f in fields(Multi)}
    assert meta["channels"]["kind"] is tuple and meta["channels"]["item"] is int
    assert meta["labels"]["item"] is str and meta["gains"]["item"] is float
    assert meta["channels"]["min_items"] == 1 and meta["channels"]["unique"] is True
    assert meta["rate"]["kind"] is int


def test_list_becomes_tuple():
    assert Multi(channels=[1, 2]).channels == (1, 2)
    assert Multi().replace(channels=[3]).channels == (3,)
    assert isinstance(Multi(labels=["a", "b"]).labels, tuple)


def test_float_items_accept_int():
    g = Multi(gains=(1, 2.5)).gains
    assert g == (1.0, 2.5) and all(type(x) is float for x in g)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"channels": (0,)}, "minimum"),
        ({"channels": (65,)}, "maximum"),
        ({"channels": ()}, "at least 1"),
        ({"channels": (1, 1)}, "duplicate"),
        ({"channels": ("1",)}, "expected int"),
        ({"channels": (True,)}, "expected int"),
        ({"channels": 1}, "expected tuple"),
        ({"gains": (-30.0,)}, "minimum"),
        ({"labels": (1,)}, "expected str"),
        ({"rate": 22050}, "one of"),
    ],
)
def test_validation(changes, message):
    with pytest.raises(ValueError, match=message):
        Multi(**changes)


def test_int_choices():
    assert Multi(rate=96000).rate == 96000


def test_toml_roundtrip(tmp_path):
    s = Multi(channels=(1, 3), labels=("m1L", "m1R"), gains=(0.5,), rate=96000)
    s.save(tmp_path / "s.toml")
    assert Multi.load(tmp_path / "s.toml") == s
    (tmp_path / "t.toml").write_text('channels = [2, 4]\nlabels = ["a", "b"]\n')
    t = Multi.load(tmp_path / "t.toml")
    assert t.channels == (2, 4) and t.labels == ("a", "b")


def test_param_type_errors():
    with pytest.raises(TypeError):
        param((), label="no item kind")  # empty tuple needs item=
    with pytest.raises(TypeError):
        param(((1,),), label="nested")
    with pytest.raises(TypeError):
        param((1, "a"), label="mixed")
    with pytest.raises(TypeError):
        param((1,), label="wrong item", item=str)
    with pytest.raises(TypeError):
        param([1], label="list default")  # defaults must be immutable tuples

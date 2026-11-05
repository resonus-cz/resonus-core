from dataclasses import dataclass

import pytest

from resonus_core.settings import Settings, param


@dataclass(frozen=True)
class Example(Settings):
    rate: int = param(
        48000, label="Sample rate", unit="Hz", choices=None, minimum=8000, maximum=192000
    )
    level: float = param(-12.0, label="Level", unit="dBFS", minimum=-60.0, maximum=0.0)
    window: str = param("hann", label="Window", choices=("hann", "rect"))
    simulate: bool = param(False, label="Simulate")


def test_defaults_and_validation():
    s = Example()
    assert s.rate == 48000 and s.window == "hann"
    with pytest.raises(ValueError, match="maximum"):
        Example(level=3.0)
    with pytest.raises(ValueError, match="one of"):
        s.replace(window="kaiser")
    with pytest.raises(ValueError, match="expected int"):
        Example(rate=44100.5)


def test_int_accepted_for_float():
    assert Example(level=-6).level == -6.0
    assert isinstance(Example(level=-6).level, float)


def test_toml_roundtrip(tmp_path):
    s = Example(rate=96000, level=-6.5, window="rect", simulate=True)
    s.save(tmp_path / "s.toml")
    assert Example.load(tmp_path / "s.toml") == s


def test_unknown_key_rejected(tmp_path):
    (tmp_path / "s.toml").write_text("rate = 48000\nbogus = 1\n")
    with pytest.raises(ValueError, match="bogus"):
        Example.load(tmp_path / "s.toml")


def test_unsupported_type():
    with pytest.raises(TypeError):
        param([1, 2], label="x")

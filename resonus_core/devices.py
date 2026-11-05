"""Audio device listing for device menus: host API, channel counts, supported sample rates.

The same card is usually listed several times, once per host API (on Linux e.g. ALSA
``hw:`` devices, the ``pipewire``/``default`` ALSA devices and JACK ports via
pipewire-jack). A device is therefore identified by its *key* ``"<host API>: <name>"``,
which is stored in settings; indices change between sessions and are only used at run
time.

Input and output devices of one full-duplex stream must belong to the same host API
(PortAudio); different devices may drift against each other (resonus-room
resonus-room D-17).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# Sample rates offered in device menus, Hz (design choice; user decision,
# resonus-room D-17). Only those accepted by the selected devices are shown.
COMMON_SAMPLE_RATES = (44100, 48000, 88200, 96000, 176400, 192000)


@dataclass(frozen=True)
class DeviceInfo:
    """One PortAudio device.

    Parameters
    ----------
    index
        PortAudio device index (valid for the current session only).
    name
        Device name as reported by PortAudio.
    host_api
        Name of the device's host API (``query_hostapis()[hostapi]["name"]``).
    max_inputs, max_outputs
        Maximum numbers of input and output channels.
    default_sample_rate
        Default sample rate in Hz.
    """

    index: int
    name: str
    host_api: str
    max_inputs: int
    max_outputs: int
    default_sample_rate: float

    @property
    def key(self) -> str:
        """Identifier stored in settings: ``f"{host_api}: {name}"``."""
        return f"{self.host_api}: {self.name}"

    @property
    def label(self) -> str:
        """Menu text: ``f"{name} ({host_api}, {max_inputs} in / {max_outputs} out)"``."""
        return f"{self.name} ({self.host_api}, {self.max_inputs} in / {self.max_outputs} out)"


def list_devices() -> list[DeviceInfo]:
    """Return all PortAudio devices in index order.

    From ``sounddevice.query_devices()`` (keys ``index``, ``name``, ``hostapi``,
    ``max_input_channels``, ``max_output_channels``, ``default_samplerate``) and
    ``sounddevice.query_hostapis()`` (host API names). ``sounddevice`` is imported inside
    the function.
    """
    import sounddevice

    apis = sounddevice.query_hostapis()
    return [
        DeviceInfo(
            int(d["index"]),
            str(d["name"]),
            str(apis[d["hostapi"]]["name"]),
            int(d["max_input_channels"]),
            int(d["max_output_channels"]),
            float(d["default_samplerate"]),
        )
        for d in sounddevice.query_devices()
    ]


def find_device(key: str, devices: Sequence[DeviceInfo] | None = None) -> DeviceInfo:
    """Return the device whose :attr:`DeviceInfo.key` equals ``key``.

    Parameters
    ----------
    key
        ``"<host API>: <name>"``.
    devices
        Devices to search; :func:`list_devices` if None.

    Raises
    ------
    ValueError
        ``f"audio device not found: {key!r}"`` if no device has this key.
    """
    devices = list_devices() if devices is None else devices
    for d in devices:
        if d.key == key:
            return d
    raise ValueError(f"audio device not found: {key!r}")


def supported_sample_rates(
    input_index: int | None,
    output_index: int | None,
    input_channels: int,
    output_channels: int,
    rates: Sequence[int] = COMMON_SAMPLE_RATES,
) -> tuple[int, ...]:
    """Return the rates of ``rates`` that both devices accept, in the given order.

    A rate is accepted if ``sounddevice.check_input_settings(device=input_index,
    channels=input_channels, samplerate=rate)`` and
    ``sounddevice.check_output_settings(device=output_index, channels=output_channels,
    samplerate=rate)`` both succeed; ``sounddevice.PortAudioError`` and ``ValueError``
    mean "not accepted". None selects PortAudio's default device.

    Parameters
    ----------
    input_index, output_index
        Device indices, or None for the default devices.
    input_channels, output_channels
        Number of channels to open: the highest channel number used (channel maps are
        1-based, so channel 3 needs 3 channels).
    rates
        Candidate rates in Hz.
    """
    import sounddevice

    accepted = []
    for rate in rates:
        try:
            sounddevice.check_input_settings(
                device=input_index, channels=input_channels, samplerate=rate
            )
            sounddevice.check_output_settings(
                device=output_index, channels=output_channels, samplerate=rate
            )
            accepted.append(rate)
        except (sounddevice.PortAudioError, ValueError):
            continue
    return tuple(accepted)


def default_device(
    direction: str, devices: Sequence[DeviceInfo] | None = None
) -> DeviceInfo | None:
    """Return PortAudio's default device for ``"input"`` or ``"output"``, or None.

    ``sounddevice.query_devices(kind=direction)["index"]`` looked up in ``devices``
    (:func:`list_devices` if None) by :attr:`DeviceInfo.index`. None if there is no
    default device (``sounddevice.PortAudioError`` or ``ValueError``) or the index is
    not in ``devices``.

    Raises
    ------
    ValueError
        ``f"direction must be 'input' or 'output', got {direction!r}"``.
    """
    if direction not in ("input", "output"):
        raise ValueError(f"direction must be 'input' or 'output', got {direction!r}")
    import sounddevice

    try:
        index = int(sounddevice.query_devices(kind=direction)["index"])
    except (sounddevice.PortAudioError, ValueError):
        return None
    devices = list_devices() if devices is None else devices
    return next((d for d in devices if d.index == index), None)

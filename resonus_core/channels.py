"""Channel-row logic of the device preferences (no GTK): device menus, rows, sample rate.

The preferences show one row per input and per output channel ("add another" rows,
removable from the second row on), device drop-downs for input and output and a
sample-rate drop-down (resonus-room D-17; user requests
2026-11-04). This module holds the rules, so that they are tested without a display;
``resonus_core.gui`` builds the widgets on top of it.

Labels follow the user's naming convention ``s#`` (source) / ``m#`` (microphone),
user decision, resonus-room D-17.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from resonus_core.devices import DeviceInfo

# Label prefixes of the naming convention s#m# (user decision, resonus-room D-17).
INPUT_PREFIX = "m"
OUTPUT_PREFIX = "s"


def device_options(devices: Sequence[DeviceInfo], direction: str) -> list[DeviceInfo]:
    """Devices offered in the input or output drop-down, in the given order.

    ``direction == "input"``: devices with ``max_inputs > 0``; ``"output"``: devices with
    ``max_outputs > 0``.

    Raises
    ------
    ValueError
        ``f"direction must be 'input' or 'output', got {direction!r}"``.
    """
    if direction not in ("input", "output"):
        raise ValueError(f"direction must be 'input' or 'output', got {direction!r}")
    if direction == "input":
        return [d for d in devices if d.max_inputs > 0]
    return [d for d in devices if d.max_outputs > 0]


def next_channel(used: Sequence[int], max_channels: int) -> int:
    """Smallest channel number ``1..max_channels`` not in ``used``.

    Raises
    ------
    ValueError
        ``"no free channel"`` if all are used (or ``max_channels < 1``).
    """
    if max_channels < 1:
        raise ValueError("no free channel")
    used_set = set(used)
    for ch in range(1, max_channels + 1):
        if ch not in used_set:
            return ch
    raise ValueError("no free channel")


def next_label(labels: Sequence[str], prefix: str) -> str:
    """Label for a new row: ``prefix`` + (highest number used + 1).

    The number of a label is the run of digits right after ``prefix`` at its start
    (``"m2"`` -> 2, ``"m1L"`` -> 1, ``"m10"`` -> 10); labels without it are ignored.
    No numbered label: ``prefix + "1"``. Example: ``(["m1", "m3R"], "m") -> "m4"``.
    """
    max_num = 0
    for lbl in labels:
        if lbl.startswith(prefix):
            m = re.match(r"\d+", lbl[len(prefix) :])
            if m:
                num = int(m.group(0))
                if num > max_num:
                    max_num = num
    return f"{prefix}{max_num + 1}"


def add_row(
    channels: Sequence[int], labels: Sequence[str], prefix: str, max_channels: int
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Append a row: ``next_channel(channels, max_channels)`` and ``next_label(labels, prefix)``.

    Returns
    -------
    channels, labels
        New tuples, one item longer.

    Raises
    ------
    ValueError
        From :func:`next_channel` (no free channel).
    """
    new_ch = next_channel(channels, max_channels)
    new_lbl = next_label(labels, prefix)
    return tuple(channels) + (new_ch,), tuple(labels) + (new_lbl,)


def remove_row(
    channels: Sequence[int], labels: Sequence[str], index: int
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Remove row ``index`` from both tuples.

    Raises
    ------
    ValueError
        ``"the first row cannot be removed"`` for ``index == 0`` (the GUI shows no cross
        on the first row; at least one channel is needed), ``"no such row"`` if
        ``index`` is outside ``1..len(channels)-1``.
    """
    if index == 0:
        raise ValueError("the first row cannot be removed")
    if index < 0 or index >= len(channels):
        raise ValueError("no such row")
    new_ch = tuple(ch for i, ch in enumerate(channels) if i != index)
    new_lbl = tuple(lb for i, lb in enumerate(labels) if i != index)
    return new_ch, new_lbl


def pick_sample_rate(current: int, device_default: float, supported: Sequence[int]) -> int:
    """Sample rate after a device selection (user decision, resonus-room D-17).

    The device's default rate ``round(device_default)`` if it is in ``supported``; else
    ``current`` if it is in ``supported``; else the first of ``supported``.

    Raises
    ------
    ValueError
        ``"no supported sample rate"`` if ``supported`` is empty.
    """
    if not supported:
        raise ValueError("no supported sample rate")
    supported_list = list(supported)
    dev_def = int(round(device_default))
    if dev_def in supported_list:
        return dev_def
    if current in supported_list:
        return current
    return supported_list[0]


def channel_limit(
    key: str, direction: str, devices: Sequence[DeviceInfo], default: DeviceInfo | None
) -> int:
    """Return the number of channels of the selected device in one direction.

    The device whose ``key`` matches (``""``: ``default``, PortAudio's default device
    for this direction): its ``max_inputs`` for ``"input"``, ``max_outputs`` for
    ``"output"``. 0 if the key is not found or ``default`` is None (unknown device).

    Raises
    ------
    ValueError
        As :func:`device_options` for an invalid ``direction``.
    """
    options = device_options(devices, direction)
    dev = default if key == "" else next((d for d in options if d.key == key), None)
    if dev is None:
        return 0
    return dev.max_inputs if direction == "input" else dev.max_outputs


def menu_channels(limit: int, used: Sequence[int]) -> list[int]:
    """Return the channel numbers of a channel menu: ``1..max(limit, max(used))``.

    Channels already used stay selectable even when the device has fewer (unknown or
    disconnected device), so building the page never changes the settings.
    """
    return list(range(1, max(limit, max(used, default=0)) + 1))

"""Preferences group "Device": input and output device menus, sample rate, drift warning.

Built on :mod:`resonus_core.channels` (rules) and :mod:`resonus_core.devices` (listing).
Devices are stored in the settings by their key ``"<host API>: <name>"``; empty = the
default device. After a device selection the sample rate menu lists the rates both devices
accept and the rate follows the device default (:func:`resonus_core.channels.pick_sample_rate`).
Different input and output devices show a warning row. User decisions, resonus-room D-17.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from gi.repository import Adw, Gtk

from resonus_core.channels import device_options, pick_sample_rate
from resonus_core.devices import DeviceInfo, default_device, supported_sample_rates
from resonus_core.gui.channels import ChannelGroup
from resonus_core.measure import DEVICE_DRIFT_WARNING, MeasurementSettings

# Menu entry for an empty device key (PortAudio's default device).
DEFAULT_DEVICE_LABEL = "Default device"

RatesFunction = Callable[[int | None, int | None, int, int], tuple[int, ...]]


class DeviceGroup[S: MeasurementSettings]:
    """The "Device" preferences group of a :class:`MeasurementSettings` object.

    Rows, in this order: ``input_row`` (Adw.ComboRow "Input device"), ``output_row``
    (Adw.ComboRow "Output device"), ``rate_row`` (Adw.ComboRow "Sample rate"),
    ``warning_row`` (Adw.ActionRow, title ``DEVICE_DRIFT_WARNING``, visible only while
    both device keys are non-empty and different).

    Device menus: entry 0 is ``DEFAULT_DEVICE_LABEL`` (key ``""``), then the
    ``DeviceInfo.label`` of each device of ``device_options(devices, "input")`` (resp.
    ``"output"``). A stored key that is not in the list gets an extra last entry
    ``f"{key} (not found)"``, which is selected; nothing is changed until the user picks
    another entry.

    Rate menu: ``self.rates()`` (see below), as strings, the current rate selected; if
    the current rate is not in that list, it is the menu ``[current]`` alone, so the
    settings are never changed by building the page.

    Changes (each calls ``on_change`` exactly once with the new settings):

    - Device selected: ``key`` of the entry (``""`` for the default device; the
      "(not found)" entry keeps the stored key). ``rates = self.rates(...)`` for the new
      pair; ``sample_rate = pick_sample_rate(base.sample_rate, device default rate,
      rates)`` where the device default rate is the selected device's
      ``default_sample_rate``, or ``base.sample_rate`` for the default/not-found entry;
      if ``rates`` is empty the rate stays ``base.sample_rate``. One
      ``base.replace(input_device=..., sample_rate=...)`` (resp. ``output_device``),
      then the rate menu is rebuilt (without emitting a change) and the warning row
      updated.
    - Rate selected: ``base.replace(sample_rate=int(entry))``.

    ``base`` is ``current()``. A ``ValueError`` from ``replace`` is passed to
    ``on_error`` (if given) and nothing is emitted.

    Parameters
    ----------
    settings
        Settings when the page is built.
    on_change
        Called with the new settings after every accepted change.
    current
        Returns the latest settings (shared by all preference pages).
    devices
        Device list; None: :func:`resonus_core.devices.list_devices`.
    rates
        ``rates(input_index, output_index, input_channels, output_channels)`` -> accepted
        rates; default :func:`resonus_core.devices.supported_sample_rates`. Called with
        the device indices (None for the default device or a key not found) and the
        highest input channel (``max(input_channels + (reference_channel,))``) and the
        highest output channel.
    on_error
        Called with the message of a rejected change.

    Attributes
    ----------
    group : Adw.PreferencesGroup
        Title "Device"; holds the four rows.
    """

    def __init__(
        self,
        settings: S,
        on_change: Callable[[S], None],
        current: Callable[[], S],
        devices: Sequence[DeviceInfo] | None = None,
        rates: RatesFunction = supported_sample_rates,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        if devices is None:
            from resonus_core.devices import list_devices

            devices = list_devices()
        self._on_change = on_change
        self._current = current
        self._rates_fn = rates
        self._on_error = on_error
        self._updating = False
        self._inputs = device_options(devices, "input")
        self._outputs = device_options(devices, "output")
        self.group = Adw.PreferencesGroup(title="Device")
        self.input_row, self._input_keys = self._device_row(
            "Input device", self._inputs, settings.input_device
        )
        self.output_row, self._output_keys = self._device_row(
            "Output device", self._outputs, settings.output_device
        )
        self.rate_row = Adw.ComboRow(title="Sample rate", subtitle="[Hz]")
        self.warning_row = Adw.ActionRow(title=DEVICE_DRIFT_WARNING)
        self.warning_row.add_prefix(Gtk.Image(icon_name="dialog-warning-symbolic"))
        for row in (self.input_row, self.output_row, self.rate_row, self.warning_row):
            self.group.add(row)
        self._set_rate_menu(settings)
        self._update_warning(settings)
        self.input_row.connect("notify::selected", lambda *_a: self._on_device("input"))
        self.output_row.connect("notify::selected", lambda *_a: self._on_device("output"))
        self.rate_row.connect("notify::selected", lambda *_a: self._on_rate())

    def rates(self, settings: S) -> tuple[int, ...]:
        """Rates accepted by the devices of ``settings`` (see the ``rates`` parameter).

        ``n_in = max((*settings.input_channels, settings.reference_channel))``, ``n_out =
        max(settings.output_channels)``; devices found with :meth:`_find` in the input /
        output options; ``self._rates_fn(input index or None, output index or None, n_in,
        n_out)``.
        """
        n_in = max((*settings.input_channels, settings.reference_channel))
        n_out = max(settings.output_channels)
        input_dev = self._find(self._inputs, settings.input_device)
        output_dev = self._find(self._outputs, settings.output_device)
        input_index = input_dev.index if input_dev is not None else None
        output_index = output_dev.index if output_dev is not None else None
        return self._rates_fn(input_index, output_index, n_in, n_out)

    def refresh(self, settings: S) -> None:
        """Rebuild the rate menu and the warning for ``settings`` without emitting a change.

        Called after a channel change elsewhere (the highest channel changes the
        accepted rates).
        """
        self._set_rate_menu(settings)
        self._update_warning(settings)

    # internals -----------------------------------------------------------------------

    @staticmethod
    def _find(devices: Sequence[DeviceInfo], key: str) -> DeviceInfo | None:
        """Return the device of ``devices`` whose ``key`` equals ``key``, else None."""
        for d in devices:
            if d.key == key:
                return d
        return None

    def _device_row(
        self, title: str, devices: Sequence[DeviceInfo], key: str
    ) -> tuple[Adw.ComboRow, list[str]]:
        """Combo row and its keys: ``[""] + device keys`` (+ ``key`` if not in them).

        Labels ``[DEFAULT_DEVICE_LABEL] + device labels`` (+ ``f"{key} (not found)"``);
        model ``Gtk.StringList.new(labels)``; ``row.set_selected(keys.index(key))``.
        """
        keys = [""]
        labels = [DEFAULT_DEVICE_LABEL]
        for d in devices:
            keys.append(d.key)
            labels.append(d.label)
        if key not in keys:
            keys.append(key)
            labels.append(f"{key} (not found)")
        row = Adw.ComboRow(title=title)
        row.set_model(Gtk.StringList.new(labels))
        row.set_selected(keys.index(key))
        return row, keys

    def _set_rate_menu(self, settings: S) -> None:
        """Rebuild the rate menu for ``settings`` without emitting a change.

        ``rates = list(self.rates(settings))``, or ``[settings.sample_rate]`` if the rate
        is not in it; store it in ``self._rate_values``; set ``self._updating = True``
        while setting the model and the selection (``try``/``finally``).
        """
        rates = list(self.rates(settings))
        if not rates or settings.sample_rate not in rates:
            rates = [settings.sample_rate]
            selected = 0
        else:
            selected = rates.index(settings.sample_rate)
        self._rate_values = rates
        self._updating = True
        try:
            self.rate_row.set_model(Gtk.StringList.new([str(r) for r in rates]))
            self.rate_row.set_selected(selected)
        finally:
            self._updating = False

    def _update_warning(self, settings: S) -> None:
        """Show ``warning_row`` iff both device keys are non-empty and different."""
        if (
            settings.input_device
            and settings.output_device
            and settings.input_device != settings.output_device
        ):
            self.warning_row.set_visible(True)
        else:
            self.warning_row.set_visible(False)

    def _commit(self, **changes: object) -> S | None:
        """Apply ``changes`` to ``self._current()`` and emit the result.

        ``new = self._current().replace(**changes)``; on ValueError call ``on_error`` (if
        given) and return None, else call ``on_change(new)`` and return ``new``.
        """
        try:
            new = self._current().replace(**changes)
        except ValueError as e:
            if self._on_error:
                self._on_error(str(e))
            return None
        self._on_change(new)
        return new

    def _on_device(self, direction: str) -> None:
        """Handle a device selection (class docstring, "Device selected").

        Return at once if ``self._updating``. Key from the row's keys; ``base =
        self._current()``; ``rates = self.rates(base.replace(<direction>_device=key))``;
        rate as in the class docstring; ``new = self._commit(<direction>_device=key,
        sample_rate=rate)``; if not None: ``_set_rate_menu(new)``, ``_update_warning(new)``.
        """
        if self._updating:
            return
        if direction == "input":
            row = self.input_row
            keys = self._input_keys
            devices = self._inputs
        else:
            row = self.output_row
            keys = self._output_keys
            devices = self._outputs
        key = keys[row.get_selected()]
        base = self._current()
        new_base = base.replace(**{f"{direction}_device": key})
        rates = self.rates(new_base)
        dev = self._find(devices, key)
        default = dev.default_sample_rate if dev is not None else float(base.sample_rate)
        if not rates:
            rate = base.sample_rate
        else:
            rate = pick_sample_rate(base.sample_rate, default, rates)
        new = self._commit(**{f"{direction}_device": key, "sample_rate": rate})
        if new is not None:
            self._set_rate_menu(new)
            self._update_warning(new)

    def _on_rate(self) -> None:
        """Return if ``self._updating``; else ``_commit(sample_rate=int(selected value))``."""
        if self._updating:
            return
        selected = self.rate_row.get_selected()
        rate = self._rate_values[selected]
        self._commit(sample_rate=rate)


@dataclass
class DeviceGroups[S: MeasurementSettings]:
    """The three device preference groups of one settings object, wired together.

    Attributes
    ----------
    device : DeviceGroup
    inputs, outputs : ChannelGroup
    """

    device: DeviceGroup[S]
    inputs: ChannelGroup[S]
    outputs: ChannelGroup[S]

    @property
    def groups(self) -> list[Adw.PreferencesGroup]:
        """``[device.group, inputs.group, outputs.group]``, to add to a page."""
        return [self.device.group, self.inputs.group, self.outputs.group]


def device_groups[S: MeasurementSettings](
    settings: S,
    on_change: Callable[[S], None],
    current: Callable[[], S],
    devices: Sequence[DeviceInfo] | None = None,
    defaults: tuple[DeviceInfo | None, DeviceInfo | None] | None = None,
    rates: RatesFunction = supported_sample_rates,
    on_error: Callable[[str], None] | None = None,
) -> DeviceGroups[S]:
    """Build the preference groups "Device", "Inputs" and "Outputs", wired together.

    They replace the generic rows of the settings groups "Device" and "Channels" in an
    app's preferences page.

    ``devices``: :func:`resonus_core.devices.list_devices` if None; ``defaults`` (input,
    output default device): ``(default_device("input", devices), default_device("output",
    devices))`` if None. Wiring: each group gets ``notify`` as its ``on_change``, which
    calls ``on_change(new)`` and then refreshes the *other* groups with ``new``: after a
    change in "Device" the two channel groups (``ChannelGroup.refresh``), after a change
    in "Inputs" or "Outputs" the device group (``DeviceGroup.refresh``).
    """
    if devices is None:
        from resonus_core.devices import list_devices

        devices = list_devices()
    if defaults is None:
        defaults = (default_device("input", devices), default_device("output", devices))
    holder: dict[str, Any] = {}

    def notify(source: str) -> Callable[[S], None]:
        def handler(new: S) -> None:
            on_change(new)
            if source == "device":
                holder["inputs"].refresh(new)
                holder["outputs"].refresh(new)
            else:
                holder["device"].refresh(new)

        return handler

    holder["device"] = DeviceGroup(settings, notify("device"), current, devices, rates, on_error)
    holder["inputs"] = ChannelGroup(
        settings, notify("inputs"), current, "input", devices, defaults[0], on_error
    )
    holder["outputs"] = ChannelGroup(
        settings, notify("outputs"), current, "output", devices, defaults[1], on_error
    )
    return DeviceGroups(holder["device"], holder["inputs"], holder["outputs"])

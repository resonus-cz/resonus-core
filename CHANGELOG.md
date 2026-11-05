# Changelog

Semantic versioning: patch = fixes, minor = additions, major = breaking API changes.
Apps depend on a pinned tag; upgrading is a deliberate change in the app.

## 0.8.1 (2026-11-05)
First public release (no API change).

- License AGPL-3.0-only (`LICENSE`); README, CONTRIBUTING, pull-request template.
- Design decisions published in `docs/DECISIONS.md` (D-N1, D-F1 added from the planning
  notes); references in docstrings point to public documents only.
- Tests: the fake `sounddevice.Stream` fixture lives in `tests/conftest.py`.

## 0.8.0 (2026-11-05)
User-replaceable brand mark (user request 2026-11-05, resonus-room D-24).

- `style.add_brand(fig, text=None)`: the text of the mark, `None` for "Re:sonus"
  (`TOKENS["brand_text"]`), empty for an invisible mark.
- `MplPlotView(..., brand_text=None)` and `MplPlotView.set_brand(text)` (on a panel: the
  figure's mark; no effect on views created with `brand=False`).

## 0.7.0 (2026-11-05)
Stop button and shared magnitude/phase toolbar (user decisions 2026-11-05, resonus-room
D-22).

- `audio`: `AudioIO.abort()` and `MeasurementAborted`: call `abort()` from any thread to
  stop the running play/record call; it and all later calls raise `MeasurementAborted`
  (an aborted object is used up). `SounddeviceIO` sets a flag and calls
  `sounddevice.stop()` (blocking `playrec`/`rec`), the `playrec_blocks` stream callback
  aborts; `SimulatedIO` checks before and after each call and between blocks. `measure`
  passes the exception on. Addition to the protocol: other `AudioIO`
  implementations need the method.
- `plot`: `MplPlotView.add_panel(height_ratios)` adds a view with its own axes below,
  in the same figure, sharing the x axis (one canvas, one toolbar, one exported image);
  `panels()`; `navigable` scroll-zooms every panel.
- tests: the fake `sounddevice.Stream` fixture moved to `tests/conftest.py`.

## 0.6.0 (2026-11-05)
Streaming play/record (user request 2026-11-05, resonus-room D-21: live input level
meter of the level test).

- `audio`: `AudioIO.playrec_blocks(signal, on_block, block_size=STREAM_BLOCK_SIZE)`:
  plays and records like `playrec` and passes each recorded block to `on_block` as it
  arrives; `SounddeviceIO` through a `sounddevice.Stream` callback (float32, channel
  mapping done in the callback), `SimulatedIO` block by block after the simulation. Addition to the protocol: other `AudioIO` implementations need the method.

## 0.5.0 (2026-11-04)
Multichannel measurement (resonus-room M4, D-17; core changes by user decision Q-M7).
Contains breaking changes (marked below); before 1.0 they bump the minor version.

- `docs/FILE_FORMAT.md`: documentation of the HDF5 result file layout (all kinds,
  metadata); no code change.
- `settings`: tuple-valued parameters (`item`, `min_items`, `unique`); choices of any
  type.
- `devices`: device listing by host API (`DeviceInfo`, key `"<host API>: <name>"`),
  `find_device`, `supported_sample_rates`, `COMMON_SAMPLE_RATES`; `SounddeviceIO` takes
  separate `input_device` / `output_device`.
- `measure`: channel settings (`input_channels`/`labels`, `output_channels`/`labels`,
  `output_mode` simultaneous/sequential, `input_device`/`output_device`), relation checks
  incl. periodic IR length <= period (Mueller & Massarani 2001, 2.1),
  `measurement_warnings`, `open_io`. Multichannel `measure()`: one IR per
  input (simultaneous) or per input and output (sequential, output-major),
  `metadata.extra["channel"]` with the s#m# label, `MeasurementResult.recordings` /
  `deconvolutions` per run.
- `gui.preferences.settings_page`: `groups` (select and order the groups shown) and
  `current` (each change starts from the latest settings), so one settings object can
  be split over several pages.
- `channels`: rules of the device/channel preferences without GTK: device menus per
  direction, next free channel, next `s#`/`m#` label, add/remove rows, sample rate after
  a device selection.
- `gui.devices.DeviceGroup`: preferences group "Device": input/output device menus by
  key, sample-rate menu limited to the rates both devices accept and following the
  device default, drift warning for different devices (`measure.DEVICE_DRIFT_WARNING`).
- `gui.channels.ChannelGroup`: preferences groups "Inputs" / "Outputs" with "add another"
  channel rows (channel drop-down, label entry, remove cross from the second row on),
  "Add input/output", reference channel (inputs), output mode (outputs); rejected
  changes revert to the settings in force. `channels.channel_limit`, `menu_channels`.
- `gui.devices.device_groups` / `DeviceGroups`: the groups Device, Inputs and Outputs
  wired together (a device change refreshes the channel menus, a channel change the
  rate menu); `DeviceGroup.refresh`; `devices.default_device`.
- **Breaking:** the io must record `input_channels` followed by the reference channel;
  the reference is the last io input (it was io input `reference_channel`), and
  `measure()` checks the io channel counts against the settings. `sample_rate` is now a
  choice of `COMMON_SAMPLE_RATES`.

## 0.4.0 (2026-11-04)
Moved from resonus-room by user decision (docs/DECISIONS.md D-P1).

- `plot`: interactive helpers for `MplPlotView`: `navigable` (GTK 4 box with matplotlib's
  navigation toolbar: pan, zoom box, home/back/forward, save; scroll-wheel zoom around the
  cursor via `on_scroll` / `zoom_limits`), `set_trace_visible` (hide/show by label, legend
  and autoscale follow the visible lines), `style_points` (point markers + grid for
  per-band plots). GTK is imported only inside `navigable`.

## 0.3.0 (2026-11-03)
Moved from resonus-room by user decision (docs/DECISIONS.md D-B1, D-A1, D-D1).

- `bands`: base-10 fractional-octave band set (`band_set`, `BandSet`, exact centres and
  edges, nominal labels; formulas ported from python-acoustics, THIRD_PARTY.md) and a
  Butterworth filter bank (`filter_bands`, forward or time-reversed). IEC 61260-1
  compliance of the filters not verified (open item).
- `air`: ISO 9613-1:1993 atmospheric absorption coefficient, power attenuation
  coefficient m, speed of sound (ISO 354:2003 Eq. 6); checked against ISO 9613-1 Table 1.
- `data.BandResults`: scalar results per band (values and units per quantity, labels,
  flags, metadata), stored by `data.save`/`load` as kind `"BandResults"`.

## 0.2.0 (2026-11-03)
Impulse-response measurement backend.

**Breaking:** `AudioIO` is multichannel: `playrec`/`rec` return `(n, n_inputs)`;
`SounddeviceIO(input_channels=..., output_channels=...)`; `SimulatedIO` takes a
`(n_inputs, n_outputs, length)` matrix of impulse responses.

- `signals`: exponential sweep (Farina, scipy chirp; optional synchronized sweep per
  Novák 2015 Eqs. 31-33), linear sweep, white/pink noise (random or periodic), MLS.
- `deconv`: Farina inverse filter, Kirkeby regularized division (Farina 2007 Eqs. 3-5),
  Novák analytic inverse (Eq. 43); common `Deconvolved` layout (harmonics at t < 0).
- `harmonics`: harmonic IR separation with fractional-delay alignment, harmonic
  responses referenced to arrival, THD vs fundamental frequency.
- `noise`: synchronous averaging, periodic deconvolution, H1 + coherence.
- `continuous`: continuous IR measurement with looped periodic noise
  (`ContinuousAnalyzer`, linear/exponential averaging).
- `window`: Hilbert envelope, ISO 3382-1:2009 A.3.4 onset (backward search from the
  peak, warning + compromise at low dynamic range), align, REW-style windows, normalize.
- `spectral`: band window, Butterworth band filters forward / zero-phase, regularized
  pre-equalization (generalized from vyhyb/imptube `filter_sweep`).
- `measure`: `MeasurementSettings`, `make_excitation`, `measure()` with repetitions,
  loopback reference channel, causal IRs and pre-ringing-aware frequency responses.
- `examples/ir_measurement.ipynb`, executed by the tests.

## 0.1.3 (2026-11-03)
**Use this or later. Tags v0.1.1 and v0.1.2 are defective:** they contain a stale
`build/` directory from 0.1.0, which setuptools may install instead of the tagged code
(timestamp-dependent). Code is otherwise unchanged from 0.1.2.
- Application ids use the `io.github.vyhyb` prefix (docs, tests).
- Build artifacts removed from the repository and ignored.

## 0.1.2 (2026-11-03)
- `settings_page` is generic in the settings type: apps pass callbacks taking their own
  settings subclass (found by type-checking the app skeleton).

## 0.1.1 (2026-11-03)
- `py.typed` marker: apps type-check their use of resonus-core (PEP 561).
- `MplPlotView._repr_png_`: a plot view displays itself in Jupyter without pyplot.

## 0.1.0 (2026-11-03)
- `settings`: typed, validated settings dataclasses (`param`), TOML load/save.
- `data`: common result types `ImpulseResponse`, `FrequencyResponse`, `Metadata`;
  HDF5 file format version 1 (`save`/`load`).
- `audio`: `AudioIO` protocol, `SounddeviceIO` (PortAudio), `SimulatedIO` (loopback
  through a known impulse response, latency, noise).
- `style`: unified plot style tokens, bundled Lexend fonts (OFL), Re:sonus mark.
- `plot`: `PlotView` interface and `MplPlotView` renderer (GTK widget, exports).
- `gui`: `AppPlugin` protocol, `StandaloneHost` (libadwaita header bar, primary menu,
  toasts, preferences, about), preferences pages generated from settings.
- Not yet verified: GNOME palette color values (transcribed from memory).

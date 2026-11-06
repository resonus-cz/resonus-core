# resonus result files (HDF5), format version 1

Written by `resonus_core.data.save`, read by `resonus_core.data.load`. One result per
file. The container is HDF5 (The HDF Group), so any HDF5 tool can read the files:
h5py, MATLAB `h5read` / `h5readatt`, HDFView, `h5dump`. The layout inside is the
convention described here. It is not an acoustics standard; for exchange with other
acoustics software, export WAV or CSV.

## Common root attributes (all kinds)

| Attribute | Type | Content |
|---|---|---|
| `format` | string | always `"resonus"`; files without it are rejected |
| `format_version` | integer | `1`; readers reject files with a newer version |
| `kind` | string | `"ImpulseResponse"`, `"FrequencyResponse"`, `"BandResults"`, `"LevelHistory"` or `"TimeSignal"` |
| `metadata` | string (JSON) | provenance, see below |

### `metadata` (JSON object, `resonus_core.data.Metadata`)

| Key | Type | Content |
|---|---|---|
| `software` | string | app that produced the result, e.g. `"resonus-room"` |
| `software_version` | string | version of that app |
| `created` | string | ISO 8601 time stamp in UTC, e.g. `"2026-11-04T13:25:36+00:00"` |
| `description` | string | free text (source file name, position, ...); may be empty |
| `device` | string | audio device / measurement chain as described by the audio backend |
| `standards` | list of strings | standards (edition, clause) the result was computed to |
| `extra` | object | app-specific keys (JSON values); documented by each app |

Results measured with `resonus_core.measure.measure` carry in `extra`:
`settings` (all `MeasurementSettings` fields of the app's settings class, as
`Settings.to_dict()`, so app-specific fields are included), `excitation` (the
excitation kind, e.g. `"exp_sweep"`) and `channel`:

| Key | Type | Content |
|---|---|---|
| `input` | integer | input channel number of the device (1-based) |
| `input_label` | string | its label, e.g. `"m1"` |
| `outputs` | list of integers | output channels that played: all in simultaneous mode, one in sequential mode |
| `output_labels` | list of strings | their labels, e.g. `["s1"]` |
| `label` | string | `"+".join(output_labels) + input_label`, e.g. `"s1m1"`, `"s1+s2m1"` |

## Kind `ImpulseResponse`

| Item | HDF5 object | Type | Content |
|---|---|---|---|
| `unit` | attribute | string | unit of the samples; `"1"` = uncalibrated (relative) |
| `sample_rate` | attribute | number | sample rate in Hz |
| `data` | dataset | float64, 1-D | samples; sample `n` is at time `n / sample_rate` |

Measured impulse responses are causal: `data[0]` is t = 0 of the deconvolution
(`measure` module docstring), length `ir_length` seconds.

## Kind `FrequencyResponse`

| Item | HDF5 object | Type | Content |
|---|---|---|---|
| `unit` | attribute | string | unit of the values; `"1"` = uncalibrated |
| `frequency` | dataset | float64, 1-D | frequencies in Hz, increasing, >= 0 |
| `values` | dataset | complex128, 1-D | complex response, same length as `frequency` |

## Kind `BandResults`

| Item | HDF5 object | Type | Content |
|---|---|---|---|
| `fraction` | attribute | integer | 1 = octave, 3 = one-third octave bands |
| `labels` | attribute | string (JSON list) | nominal band labels, e.g. `["125", "250"]` |
| `quantities` | attribute | string (JSON list) | names of the quantities, in order |
| `units` | attribute | string (JSON object) | quantity name -> unit (`"s"`, `"dB"`, `"1"`, ...) |
| `flags` | attribute | string (JSON list of lists) | per band: warning flags (strings); may be empty |
| `center` | dataset | float64, 1-D | exact band centre frequencies in Hz |
| `values/<name>` | dataset in group `values` | float64, 1-D | one value per band; NaN = not available |

## Kind `LevelHistory`

Levels over time, e.g. the log of a sound level meter (`resonus_core.data.LevelHistory`;
resonus-meter bundles convert with `meter_session.read_session(...).level_history()`).

| Item | HDF5 object | Type | Content |
|---|---|---|---|
| `quantities` | attribute | string (JSON list) | names of the level columns, in order, e.g. `["LAF", "LCF", "LZF"]` |
| `units` | attribute | string (JSON object) | quantity name -> unit, e.g. `"dB"` |
| `markers` | attribute | string (JSON list) | `[time_s, label]` pairs; may be empty |
| `start_utc` | attribute | string | ISO 8601 UTC time of `time = 0`; empty if unknown |
| `time` | dataset | float64, 1-D | times in s from `start_utc`, strictly increasing |
| `levels/<name>` | dataset in group `levels` | float64, 1-D | one value per time; NaN = missing |

## Kind `TimeSignal`

A recorded signal, e.g. background noise in Pa (`resonus_core.data.TimeSignal`, since
0.11.0; resonus-room Q-C9). Earlier apps stored recordings as `ImpulseResponse` with a
role in `metadata.extra`.

| Item | HDF5 object | Type | Content |
|---|---|---|---|
| `unit` | attribute | string | unit of the samples, e.g. `"Pa"`, `"FS"` (digital full scale) |
| `sample_rate` | attribute | number | sample rate in Hz |
| `start_utc` | attribute | string | ISO 8601 UTC time of the first sample; empty if unknown |
| `data` | dataset | float64, 1-D | samples; sample `n` is at time `n / sample_rate` |

## Reading without resonus-core

Python (h5py):

```python
import json, h5py

with h5py.File("room.ir.h5") as f:
    assert f.attrs["format"] == "resonus" and f.attrs["kind"] == "ImpulseResponse"
    fs = float(f.attrs["sample_rate"])
    ir = f["data"][()]
    meta = json.loads(f.attrs["metadata"])
```

MATLAB:

```matlab
ir   = h5read("room.ir.h5", "/data");
fs   = h5readatt("room.ir.h5", "/", "sample_rate");
meta = jsondecode(h5readatt("room.ir.h5", "/", "metadata"));
```

## Compatibility

Additions that old readers can ignore (new `extra` keys, new kinds) keep version 1; a
new kind is rejected by older readers with "unknown result kind". Changes to existing
items increase `format_version`.

# resonus-core

Shared building blocks of the Re:sonus acoustics apps (for example
[Re:sonus Room](https://github.com/resonus-cz/resonus-room)). Each module that
implements a standard cites edition and clause and is tested against reference values.

- `measure`, `signals`, `deconv`, `harmonics`, `noise`, `continuous`, `window`: impulse
  and transfer-function measurement (exponential and synchronized sweeps, noise, MLS,
  loopback reference, synchronous averaging, harmonic distortion, IR onset per
  ISO 3382-1:2009 A.3.4), single and multichannel.
- `audio`, `devices`, `channels`: audio I/O through PortAudio (`sounddevice`) or a
  simulated device, streaming play/record, stopping a running measurement; device and
  channel rules.
- `bands`, `spectral`: fractional-octave bands and filter banks (IEC 61260-1 class
  conformance not verified yet), spectra.
- `air`: atmospheric absorption per ISO 9613-1:1993.
- `data`: result data model and HDF5 file format
  ([docs/FILE_FORMAT.md](docs/FILE_FORMAT.md)): impulse and frequency responses, per-band
  results.
- `settings`: typed settings with TOML files and generated preference pages.
- `plot`, `style`: one plot interface for GUI, exports and notebooks, the Re:sonus plot
  style (Lexend font, replaceable plot mark), navigation helpers.
- `gui`: GTK 4 / libadwaita host for app plugins, preference groups for devices and
  channels.

> Status: 0.x, the API may still change between minor versions (see
> [CHANGELOG.md](CHANGELOG.md)). Apps pin a tag.

## Installation
Python ≥ 3.12; GTK 4, libadwaita, PortAudio and the GObject introspection headers
(PyGObject is built by pip). On Fedora:

    sudo dnf install gtk4 libadwaita portaudio gobject-introspection-devel \
        cairo-gobject-devel python3-devel gcc pkgconf

Use from an app (pinned tag):

    dependencies = ["resonus-core @ git+https://github.com/resonus-cz/resonus-core@v0.11.1"]

Example: [examples/ir_measurement.ipynb](examples/ir_measurement.ipynb) measures an
impulse response on the simulated device.

## Development

    python -m venv .venv && .venv/bin/pip install -e ".[dev]"
    scripts/check.sh        # pytest, ruff (lint + format), mypy

Exact dependency versions of the tested environment: `requirements.lock`. Design
decisions: [docs/DECISIONS.md](docs/DECISIONS.md).

## License
[AGPL-3.0-only](LICENSE). Copyright © 2026 David Jun (Re:sonus, resonus.cz).
The bundled Lexend fonts are under the SIL Open Font License 1.1
(`resonus_core/assets/fonts/lexend/OFL.txt`). Contributions: see
[CONTRIBUTING.md](CONTRIBUTING.md). Third-party notices: [THIRD_PARTY.md](THIRD_PARTY.md).

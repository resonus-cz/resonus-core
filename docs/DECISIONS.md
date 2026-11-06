# Decision log (resonus-core)

One entry per decision; never rewrite old entries, supersede them. D-N1 and D-F1 were
first recorded in the planning notes (2026-11-02) and copied here unchanged in substance.

## D-N1: Noise generation (module `signals`)
- scipy has no white or pink noise generator (only `max_len_seq` for MLS). White noise:
  numpy `Generator`. Pink noise: shaped in the frequency domain over the whole length
  (periodic, exactly 1/f power); an IIR pink filter for non-periodic, continuous streams.

## D-F1: Zero-phase vs forward filtering (module `spectral`)
- `sosfiltfilt` (zero-phase) applies the filter twice: the magnitude response is squared
  (a Butterworth's -3 dB edges become -6 dB) and the effective order doubles; time
  reversal also changes decay shapes (relevant for ISO 3382).
- Decided: zero-phase by default for spectra and IR display; forward (causal) filtering
  available and required for decay analysis. Band filters meant to meet IEC 61260-1 class
  limits must be designed so that the squared response meets them, or filtered forward.

## D-B1: Fractional-octave bands and filters (module `bands`, from resonus-room)
- Date / status: 2026-11-03, moved from resonus-room by user decision (resonus-room
  QUESTIONS.md Q-C2); filter compliance PROVISIONAL.
- Decision: base-10 band formulas and nominal tables ported from python-acoustics
  (BSD-3-Clause, THIRD_PARTY.md); one-third-octave centres checked against ISO 9613-1:1993
  Eq. (6). Butterworth band-pass on the exact band edges (`spectral.band_sos`), order 3 by
  default, forward filtering, optional time reversal (ISO 3382-2:2008 7.3 NOTE).
- Open: IEC 61260-1:2014 was not available; class 1/2 compliance of the filters and the
  nominal labels (Annex E) are unverified (open item).

## D-A1: Atmospheric absorption (module `air`, from resonus-room)
- Date / status: 2026-11-03, moved from resonus-room by user decision (Q-C3).
- Decision: implemented from ISO 9613-1:1993 clause 6 and Annex B (not copied from
  vyhyb/sapyne, GPL-3.0); verified against all Table 1 values at 20 and 25 degrees C
  (`tests/test_air.py`). Speed of sound `c = 331 + 0.6 t` from ISO 354:2003 Eq. (6).

## D-D1: `BandResults` data type (per-band scalar results)
- Date / status: 2026-11-03, user decision to add it (resonus-room Q-C1); design proposed during development.
- Decision: one value per quantity and band (dict of arrays) with per-quantity units,
  band centres and nominal labels, optional per-band warning flags, `Metadata`. Stored
  with `data.save` as kind `"BandResults"` in the existing format (version 1; the new
  kind is additive, older readers reject it with "unknown result kind").
- Use: reverberation times, clarity, absorption coefficients etc. from any app, so the
  viewer can list and plot them.

## D-P1: Interactive plot helpers (in `plot`, from resonus-room)
- Date / status: 2026-11-04, moved from resonus-room by user decision (resonus-room
  QUESTIONS.md Q-C5).
- Decision: plain functions on `MplPlotView` (public `axes`, `widget()`, `refresh()`), so
  every app's plots navigate the same way. Scroll zoom step 1.2 per click and marker size
  4 are design choices, not normative. Toggling visibility keeps the line objects (and
  so their colours, which `set_trace` assigns by insertion order) and rebuilds the legend
  from the visible lines; apps re-apply visibility after `set_trace`, which rebuilds the
  legend with all lines.

## D-C1: Level calibration with the method of resonus-meter (user, 2026-11-05)
- Date / status: 2026-11-05, user decision (resonus-room STI work package item 1:
  "identically implemented in Python into resonus-core"); tasks 022a/b.
- Decision: `calibration` ports resonus-meter's method unchanged: tone search on the first
  second, lock-in on whole seconds, its thresholds and failure messages, and its
  append-only `calibrations.toml` (schema version 1, readable by both apps). This replaces
  the band filter of PLAN item 11: the lock-in rejects noise and hum better, and both
  apps then give the same offset. Numerical check of the method (review 2026-11-05):
  level error < 0.005 dB within ±3 % of the nominal frequency.
- Added over the meter: two optional corrections in dB, each with its source as text:
  ambient (IEC 61672-3:2013 10.2; per IEC 60942:2017 5.1.5 / Table 1 only class LS/M and
  1/M pistonphones need one, for static pressure, from their instruction manual) and
  adjustment (IEC 61672-3:2013 9.1, the manufacturer's data for the
  calibrator-microphone combination). `C = L_cal + dL_ambient + dL_adjustment - L_tone`.
  Older meter versions ignore the extra keys.
- Thresholds (8 s, ±3 %, -60 dB FS, 20 dB, 0.1 dB) are the meter's design choices, not
  normative. Clipping: `|x| >= 1 - 2**-15` (the top code of a 16-bit converter), one value
  for the whole stack (user, 2026-11-05; resonus-room and resonus-meter use it too; the
  meter used 0.999 before).
- Profiles: (device id, zero-based channel); each app keeps its own store in the same
  format (device ids differ between cpal and PortAudio).

## D-T1: Style tokens as a data file shared with the Rust apps (user, 2026-11-05)
- Date / status: 2026-11-05, user decision (resonus-meter review V5).
- Decision: the plot style tokens (font family, trace colours, sizes and colours) live in
  `resonus_core/assets/style_tokens.toml` (schema version 1; unknown keys ignored, a
  newer version rejected). `style` reads it at import, so `TOKENS`, `TRACE_COLORS` and
  `FONT_FAMILY` keep their names and values. Rust apps vendor the file from a pinned
  resonus-core tag, with a test comparing the copy to that tag, so no value is
  duplicated by hand.
- Colours (user, 2026-11-05, on a proposal): GNOME HIG palette
  (developer.gnome.org/hig/reference/palette.html, verified 2026-11-05) and the libadwaita
  1.10 view background (`--view-bg-color`: light `#ffffff`, dark `#1d1d20`). Rule: each
  trace keeps its hue and reaches 3:1 on its background (WCAG 2.1 SC 1.4.11, non-text
  contrast); the greys match the contrast of the light style. Hence light Orange 3 ->
  Orange 4; Yellow 5 (2.2:1 on white, no tone of yellow reaches 3:1) moves last; Dark 3
  leaves the trace list for the new token `neutral_color` (averages, references), so
  `TRACE_COLORS` has 7 entries (was 8). Dark style (optional `[dark]` table of the same
  file, schema still version 1): Purple 2 for Purple 3, grid Dark 3, axes and neutral
  Light 3, brand Light 5, background `#1d1d20`. `style.LIGHT`, `style.DARK`,
  `mpl_rc(dark=...)`. Following the system style in a GUI is each app's job.

## D-S1: Reader for resonus-meter bundles in resonus-core (user, 2026-11-05)
- Date / status: 2026-11-05, user decision on resonus-meter review D1 ("as proposed");
  details proposed for the user's review.
- Decision: the meter's bundle (`splmeter-session` 1.x, resonus-meter
  `docs/FILE_FORMAT.md`) stays its canonical format; `resonus_core.meter_session`
  reads it (`read_session`) into numpy arrays and the common types: band levels as
  `BandResults` (summary or one segment, quantity `LZeq`, unit dB), provenance as
  `Metadata`. The Rust app needs no HDF5 library.
- Time histories: new kind `LevelHistory` (times, named level columns, markers, start
  time; HDF5 kind, `format_version` stays 1 as a new kind), from `MeterSession.level_history()`
  (user, 2026-11-05, Q-C8 "implement according to the proposal"). Segment
  values stay arrays on `MeterSession`.
- New dependency PyYAML (MIT; pinned 6.0.3, stubs types-PyYAML 6.0.12.20261103 in dev):
  the front matter holds user text (names, tags, notes) in YAML double-quoted strings,
  which a hand-written subset parser would get wrong. `yaml.safe_load` only.
- `Metadata.standards` stays empty: the bundle names no standard, and the meter's
  compliance matrix has open rows (IEC 61672-1 Table 3, IEC 61260-1 class limits).
- `software` "SPL Meter 0.1.0" is split at the last space when the last word starts with
  a digit; the meter still writes its old name (reported to the user).

## D-25: License AGPL-3.0-only for resonus-room and resonus-core (user, 2026-11-05)
- Before publishing both repos on GitHub. User goals: own and third-party commercial use
  allowed; selling allowed as long as the code stays available. GPL-3.0 already keeps
  derivatives open when distributed; AGPL-3.0 section 13 also covers modified versions
  offered as a network service. `-only` (not `-or-later`): the user keeps control of the
  version; as sole copyright holder the user can relicense (external contributions
  would need a CLA to keep that option).
- AGPL is OSI-approved and FSF-free: GitHub, PyPI and Flathub show it as open source.
- Supersedes the GPL-3.0 of D-12. Compatible: sapyne (GPL-3.0, same author, nothing
  copied), python-acoustics (BSD-3-Clause, attribution in THIRD_PARTY.md), dependencies
  per `pip-licenses` (VERIFICATION / THIRD_PARTY.md). Not legal advice.

## D-26: Application id cz.resonus.Room (user, 2026-11-05)
- The user owns resonus.cz: app ids are `cz.resonus.<Name>` (Re:sonus architecture decision
  D7, also in the project template), independent of the code hosting. resonus-room:
  `cz.resonus.Room` (was `io.github.vyhyb.ResonusRoom`, never released). Settings are
  stored by folder name (`~/.config/resonus-room`), so existing settings stay valid.
- Flathub verification: token at `https://resonus.cz/.well-known/org.flathub.VerifiedApps.txt`
  (needs a web page on the domain).

## D-27: Publishing: private repo -> public subset -> GitHub (user, 2026-11-05)
- GitHub organisation `resonus-cz` (the name `resonus` is taken by another user). The
  private repos (Gitea) keep everything; `scripts/export_public.sh <tag> <public-clone>`
  copies only `scripts/public_files.txt` of the tag into the public repo (fresh history,
  one "Release" commit per version), after `scripts/check.sh` passed; a Gitea
  `<name>-public` repo push-mirrors to GitHub.
- The export rewrites only metadata: local (file URL) dependency URLs ->
  `https://github.com/resonus-cz/<repo>@<tag>`, and a public `.gitignore`. It refuses to
  export local paths or internal files and runs the tests on the exported tree.
- Public content (user decisions 2026-11-05): code, tests, examples, LICENSE, README,
  CHANGELOG, THIRD_PARTY, CONTRIBUTING (+ PR template), pyproject, lock file, check/lint
  scripts, docs FILE_FORMAT and DECISIONS (resonus-room also QUESTIONS, COMPLIANCE,
  VERIFICATION); those docs were reworded once to be publishable, and references in the
  code point only to them. The REW sample data stays private (its tests skip).
- Pull requests: issue first, CLA before integration (keeps relicensing possible); a PR
  is applied to the private repo (`git am`, authorship kept), released, credited with
  `export_public.sh --co-author` and the changelog, and closed with a link. Never merged
  on GitHub (the push mirror would overwrite it).
- The export refuses workflow traces and the copyright-holder placeholder of the README.

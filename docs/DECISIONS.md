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

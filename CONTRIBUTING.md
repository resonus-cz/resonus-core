# Contributing

Thank you for your interest. Bug reports, questions and suggestions are welcome as
GitHub issues; please include the version, your operating system and audio interface,
and, for measurement problems, the settings file (`settings.toml`) and the result files
if you can share them.

## Pull requests
Pull requests are welcome, but please open an issue first so we can agree on the
approach. Before a contribution can be merged, you will be asked to sign a contributor
license agreement (CLA). It lets the maintainer keep licensing the project as a whole
(the code is AGPL-3.0-only); you keep the copyright of your contribution.

Accepted pull requests are integrated into the main development line and published with
the next release, so a pull request is closed with a link to that release instead of
being merged on GitHub; your authorship is kept in the release commit and in the
changelog.

## What a change needs
- Formulas, constants and limits cite their source: standard with edition and clause,
  or a paper. Do not paste text from standards (they are copyrighted); numeric values
  needed by tests are fine.
- Tests with reference values from standard tables, closed-form results or independent
  tools, plus edge cases.
- Type hints and numpy-style docstrings; `scripts/check.sh` (pytest, ruff, mypy) passes.

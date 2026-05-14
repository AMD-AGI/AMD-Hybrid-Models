# Contributing to AMD-Hybrid-Models

Thanks for your interest in **AMD-Hybrid-Models** — the official home of
[X-EcoMLA](./X-EcoMLA) and [Zebra-Llama](./Zebra-Llama). This document
explains how to file issues, propose changes, and get them merged.

By participating in this project you agree to abide by these guidelines and
the project's [LICENSE](./LICENSE) (Apache-2.0).

---

## Table of contents
1. [Code of conduct](#code-of-conduct)
2. [Ways to contribute](#ways-to-contribute)
3. [Reporting bugs](#reporting-bugs)
4. [Requesting features](#requesting-features)
5. [Security issues](#security-issues)
6. [Development setup](#development-setup)
7. [Pull request workflow](#pull-request-workflow)
8. [Coding style](#coding-style)
9. [Commit messages & sign-off (DCO)](#commit-messages--sign-off-dco)
10. [Reviews & code owners](#reviews--code-owners)
11. [Releases](#releases)
12. [License of contributions](#license-of-contributions)

---

## Code of conduct

Be respectful, constructive, and inclusive. Harassment or personal attacks are
not tolerated. Maintainers may close issues / PRs that violate this rule.

## Ways to contribute

- Reporting reproducible bugs.
- Proposing new features or model recipes (please open an issue first to
  discuss scope).
- Improving documentation, examples, and benchmarks.
- Fixing typos, broken links, or build issues.
- Adding or improving tests.

## Reporting bugs

Before opening a bug, please:

1. Make sure you are on the latest `main`.
2. Search [existing issues](https://github.com/AMD-AGI/AMD-Hybrid-Models/issues)
   to avoid duplicates.
3. Include in your report:
   - A clear title and short description.
   - **Environment**: OS, Python version, PyTorch / ROCm / CUDA version,
     GPU model (e.g. MI300X, MI250, H100), driver version.
   - Exact command(s) you ran and the **full** error / stack trace.
   - A **minimal reproducible example** (config file, script, or
     `transformers` snippet).
   - Expected vs. actual behavior.

## Requesting features

Open an issue describing:
- The motivation (what problem does it solve?).
- Proposed API / config surface, if any.
- Whether you're willing to implement it.

For **large** changes (new training recipe, new model architecture, breaking
API change), please get maintainer sign-off in an issue *before* sending a PR.

## Security issues

**Do not** file public issues for vulnerabilities. Follow the process in
[SECURITY.md](./SECURITY.md) to report privately.

## Development setup

Each sub-project has its own environment. Install only the one you need:

```bash
# X-EcoMLA
cd X-EcoMLA && bash install.sh

# Zebra-Llama
cd Zebra-Llama && bash install.sh
```

Recommended:

- Python ≥ 3.10
- A **fresh** virtual environment (`python -m venv .venv && source .venv/bin/activate`).
- ROCm- or CUDA-capable GPU for training / benchmarking.

Smoke-check before sending a PR:

```bash
# X-EcoMLA quick eval
cd X-EcoMLA && bash eval.sh

# Zebra-Llama quick chat
cd Zebra-Llama && python chat.py --help
```

## Pull request workflow

1. **Fork** the repo and create a topic branch from `main`:
   ```bash
   git checkout -b feat/<short-description>
   ```
2. Keep the change **focused**. One logical change per PR.
3. Add or update tests / benchmarks when behavior changes.
4. Update the relevant `README.md` (root, `X-EcoMLA/`, or `Zebra-Llama/`)
   when you add a config, flag, or script.
5. Make sure the code runs and existing scripts still work.
6. Push the branch and open a PR against `main`.
7. Fill in the PR template (motivation, what changed, how it was tested,
   any benchmark numbers).
8. Be ready to iterate — code owners will review and may request changes.

### PR checklist

- [ ] The PR is rebased on top of the latest `main`.
- [ ] No unrelated changes (formatting churn, vendor bumps, etc.).
- [ ] No secrets, API keys, tokens, model weights, or large binary blobs
      are committed.
- [ ] New files are covered by the existing `.gitignore` where appropriate
      (caches, checkpoints, `wandb/`, `.env`, …).
- [ ] Public-facing changes are documented in the README.
- [ ] Commits are signed off (see [DCO](#commit-messages--sign-off-dco)).

## Coding style

- **Python**: PEP 8 with a 100-column soft limit. Prefer
  [`ruff`](https://github.com/astral-sh/ruff) / `black` for formatting.
- Use type hints on new public functions where reasonable.
- Avoid hard-coded local paths, usernames, tokens, or W&B entity names in
  committed code. Read them from environment variables or CLI flags.
- Avoid wildcard imports (`from x import *`).
- Add docstrings for non-trivial functions and classes.
- Keep notebooks out of `main` unless they're a documented demo.

## Commit messages & sign-off (DCO)

We use the **Developer Certificate of Origin** ([DCO](https://developercertificate.org/)).
Every commit must include a `Signed-off-by` trailer with your real name and
the email used on GitHub:

```
Signed-off-by: Your Name <you@example.com>
```

Easiest way: configure git once and use `-s`:

```bash
git config user.name  "Your Name"
git config user.email "you@example.com"
git commit -s -m "feat(x-ecomla): add rank-128 config for Llama-3.2-3B"
```

### Commit message format

Prefer [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short summary>

<optional body>
<optional footer>
```

Common types: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `chore`,
`build`, `ci`. Common scopes: `x-ecomla`, `zebra-llama`, `benchmark`,
`docs`, `infra`.

## Reviews & code owners

PRs are auto-routed to the maintainers listed in
[`.github/CODEOWNERS`](./.github/CODEOWNERS). At least **one** code-owner
approval is required before merge. Maintainers may squash-merge to keep
history linear.

## Releases

Releases are tagged from `main` using semantic versioning (`vMAJOR.MINOR.PATCH`)
when the maintainers decide a milestone is ready. There is no fixed cadence.

## License of contributions

By submitting a contribution you agree that your work is licensed under the
[Apache License 2.0](./LICENSE) that covers this project. You also confirm
that you have the right to submit the work (see the [DCO](https://developercertificate.org/)).

---

If anything in this document is unclear, please open an issue — we'd rather
fix the docs than block contributors.

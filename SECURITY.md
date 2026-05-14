# Security Policy

The **AMD-Hybrid-Models** project (which hosts
[X-EcoMLA](./X-EcoMLA) and [Zebra-Llama](./Zebra-Llama)) takes security
seriously. This document explains what is in scope, how to report a
vulnerability, and what you can expect from us.

> ⚠️ **Important**: Please **do not** open public GitHub issues, discussions,
> or pull requests for suspected security vulnerabilities. Use the private
> channels described below.

---

## Supported versions

This repository is an active research codebase that ships from `main`. We
support security fixes only on the latest commit of `main`. Older tags and
forks are **not** maintained.

| Version / branch | Receives security fixes |
| ---------------- | ----------------------- |
| `main` (latest)  | ✅ Yes                   |
| Tagged releases  | ⚠️ Best effort           |
| Forks            | ❌ No                    |

## What is in scope

Reports are welcome for issues such as:

- Remote code execution, command injection, or arbitrary file write
  reachable via the training, inference, or benchmarking scripts.
- Insecure deserialization (e.g. unsafe `pickle` / `torch.load` use that an
  attacker could exploit with a crafted checkpoint).
- Path traversal or symlink attacks in dataset/config loaders.
- Credential / token leakage (committed secrets, log files, W&B configs).
- Dependency-supply-chain issues (compromised pin, typo-squatted
  requirement, known-vulnerable transitive dep with a viable exploit path).
- Authentication / authorization issues in any helper scripts.

## What is **out of scope**

- Issues that require physical access to the user's machine or root on the
  host where training is run.
- Denial-of-service from feeding intentionally pathological model inputs
  (e.g. extremely long sequences) — these are research / robustness issues,
  not security ones.
- Hallucinations, biased outputs, or other model-quality concerns from the
  resulting LLMs. File those as regular issues.
- Vulnerabilities in upstream third-party dependencies (PyTorch,
  transformers, mamba-ssm, lm-eval, etc.). Please report those to the
  upstream project; we will follow their fix.
- Findings from automated scanners without a working proof of concept.

## How to report a vulnerability

Please use **one** of the following private channels:

1. **GitHub Private Vulnerability Reporting (preferred).**
   Go to the repository's
   [Security tab](https://github.com/AMD-AGI/AMD-Hybrid-Models/security)
   → "Report a vulnerability". This creates a private advisory only the
   maintainers can see.

2. **Email the maintainers.**
   Send details to the AMD AGI maintainers listed in
   [`.github/CODEOWNERS`](./.github/CODEOWNERS). When in doubt, contact
   AMD Product Security via the public AMD security page:
   <https://www.amd.com/en/resources/product-security.html>.

Please include, where possible:

- A description of the issue and its impact.
- The affected file(s), commit SHA, and version / environment.
- Step-by-step **reproduction** instructions and a minimal proof of concept.
- Any logs, stack traces, or screenshots that help us reproduce.
- Your name / handle for credit (or a note that you wish to stay anonymous).

You may encrypt sensitive details. We will provide a key on request.

## Our commitments

When you report in good faith, we commit to:

| Stage                          | Target                                   |
| ------------------------------ | ---------------------------------------- |
| Acknowledge receipt            | within **3 business days**               |
| Initial assessment / triage    | within **10 business days**              |
| Status update cadence          | at least every **14 days** until closed  |
| Fix or mitigation              | as quickly as severity allows            |

We will:

- Coordinate disclosure with you and credit you (unless you opt out).
- Not take legal action against researchers who follow this policy,
  act in good faith, and avoid privacy violations, data destruction,
  or service disruption.

## Safe-harbor / responsible disclosure

We follow a coordinated-disclosure model. Please give us a reasonable
window — typically **90 days** from the date of acknowledgement — to ship
a fix before any public disclosure. We'll work with you to agree on the
exact timing.

## Handling secrets in this repo

If you discover that a secret (API key, access token, model-hub token,
credential, private model checkpoint, etc.) has been committed to this
repository:

1. **Do not** post the secret in a public issue.
2. Report it through the channels above.
3. We will rotate / revoke the secret, rewrite history if needed, and add
   a regression check.

## Hardening guidance for users

Until a fix is available, users running this code should:

- Treat checkpoints and configs from untrusted sources as untrusted code
  (consider `weights_only=True` for `torch.load`, sandbox the process).
- Avoid running the training / inference scripts as `root`.
- Pin dependencies to the versions documented in the sub-project README
  and `install.sh` scripts.
- Never store HuggingFace, W&B, or cloud credentials in committed config
  files — use environment variables or your platform's secret manager.

---

Thank you for helping keep AMD-Hybrid-Models and its users safe.

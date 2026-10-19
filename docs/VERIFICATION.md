# Release verification: BeliefTown 1.0.0

This is the historical verification record for release 1.0.0. It describes the checks run for that release; it does not claim 100% code coverage, universal bug-freedom, remote hosting, or performance outside the declared finite model.

## GitHub publication checks — 2026-10-19

The publication was additionally checked on Windows with Python 3.13.1, NumPy 2.2.4, and CPU PyTorch 2.9.1. All **127 tests passed**, and all **four notebooks (22 code cells)** executed with errors disallowed. A rebuilt source distribution and wheel completed successfully; a separate virtual environment imported the installed wheel and passed all **five installation/CLI/resource checks**, reusing system dependencies.

The live local application was also opened in the collaborative browser over HTTP. Finding and applying the canonical minimum patch changed the Courier's next destination from Workshop to Library while retaining Workshop in the original episode. The resulting page reported one permitted delivery, and no JavaScript console errors were observed in this flow. This was a publication smoke check, not a rerun of the complete browser suite or training benchmark. The monochrome interface redesign happened after release 1.0.0; its controls have not been rerun through the historical interaction suite.

These checks exposed default Windows text-encoding failures. File operations now explicitly use UTF-8, and CI includes Windows. The bundled Linux release reports and benchmark measurements below remain historical evidence from the supplied archive; they have not been relabeled as new Windows measurements. `SHA256SUMS.txt` records the current tracked source and artifact bytes, excluding itself and generated build outputs.

## UI preview refresh — 2026-10-19

The README screenshots were recaptured at desktop and 390 × 844 mobile sizes from the local redesigned app. Both PNGs link to their full-resolution images. The screenshots and walkthrough frames use neutral grayscale with light sharpening to preserve clean text when GitHub scales them. The standalone showcase, its offline browser copy, and the captioned walkthrough use the monochrome interface. In the shared browser preview, the initial scene loaded, minimum-patch search returned a verified result, applying it changed the Courier's destination to Library, and replay advanced. This was a manual browser smoke check. The automated Playwright suite was not rerun because its configured Chromium executable was unavailable in this environment; the ten-check result below remains historical release evidence.

The Python test suite was rerun after the learning-code cleanup: **127 passed** on Windows with Python 3.13.1. One Jupyter/ZeroMQ compatibility warning was emitted; no test failed or was skipped.

## Executed gates

| Gate | Observed result | Evidence |
|---|---|---|
| Automated correctness suite | **127 passed; 0 failed; 0 skipped** | `artifacts/final_tests.log`, `junit.xml` |
| Statement coverage in that suite | **91.72%** | `artifacts/coverage.json` |
| Research notebook execution | **4 notebooks, 22 code cells, zero cell errors** | `artifacts/executed_notebooks/` |
| Saved notebook visual outputs | **5 inline PNG figures** | Executed notebook files |
| Browser application checks | **10 passed, zero JavaScript exceptions** | Historical `checks.json` for 1.0.0; refreshed desktop/mobile screenshots show the current design |
| Packaged installation | **5 wheel/CLI/resource checks passed in a separate venv** | `artifacts/wheel_smoke.json` |
| Release training | **9 real CPU checkpoints; 14 epochs each; 3 seeds per model type** | Checkpoints and complete epoch histories |
| Held-out search benchmark | **792 outcomes agree with exhaustive feasibility/minimum labels** | Raw CSV, results JSON and report |
| Synthetic data integrity | **216 scenes; 13,001 labeled candidate patches** | Hash-checked whole-scene manifests |
| Independent audit recheck | **Canonical one-delivery patch recertifies minimum** | `artifacts/example_audit.json` |
| Single-command quick pipeline | **Own data, 3 trained models, 90 correct search outcomes, 4 notebooks against the rebuilt run** | `artifacts/quick_reproduction/`, `quick_pipeline.log` |
| Launch walkthrough | **60-second captioned capture sequence; no audio** | MP4, current monochrome UI frames, capture manifest |

The ordinary test suite's coverage does not include every line exercised in separate full benchmark, browser, or notebook subprocesses. Its percentage is reported as measured, not padded with those other executions.

## What was tested mathematically

Tests include filtering versus independent latent-trajectory enumeration, stale evidence at original timestamps, duplicate-root idempotence, conflicting tiny likelihoods, illegal observations and messages, private-input invariance, shortest paths and exhaustive route choices, alternate communication paths under suppression, protected information, simultaneous-patch legality, bounded/no-solution/budget statuses, and maliciously altered audit records. Learning tests cover actual optimization, checkpoint reload, custom architecture preservation, node-permutation invariance, missing-truth features, split separation, and a three-model benchmark.

## Important execution boundaries

**Browser transport:** this build environment's browser rejected loopback navigation with an administrator policy. The browser checks therefore used a documented in-process binding to the real Python Studio service, not a mocked result table. Separate urllib tests exercised the actual loopback HTTP endpoints, token enforcement, host/origin restrictions, and responses. A fully navigated HTTP browser session on every user machine is not claimed.

**Installation:** the wheel was installed into a separate virtual environment and its import path was asserted to point to that environment, not the editable source. Third-party dependencies were reused with `--system-site-packages` because network installation was unavailable. This verifies the project wheel and packaged assets, not a fresh resolution/download of all dependencies.

**Platforms:** local gates ran on Linux with Python 3.13.5 and CPU PyTorch 2.10.0+cpu. The complete environment snapshot and pinned observed dependency closure are included. The package declares Python 3.10+; the supplied multi-version GitHub Actions workflow is not evidence that remote CI or Windows/macOS execution has occurred.

**Notebook hosting:** the local kernel-backed server is live only while its owner runs. Saved outputs and recorded HTML do not keep a server permanently hosted. Hosted-notebook port forwarding and public deployment are not configured.

**Research scope:** no human study, commercial-engine integration, psychological validation, or world-first novelty claim is included. The neural ranker did not establish a wall-clock speed advantage over the heuristic on the released small synthetic benchmark. Minimum certificates apply only to their recorded candidate spaces and cardinality bounds.

## Corrections made during integration

The release checks exposed and corrected the source-install subprocess entry point, stricter timestamp/index contracts, fabricated audit verification fields, custom checkpoint architecture preservation, one-seed report labeling, and rebuilt-notebook data-path selection. These corrections are covered by tests or end-to-end execution; they are not deferred TODO items.

## Reproduce the gates

```bash
python -m pip install -e ".[all]"
python -m pytest -q --cov=belieftown --cov-report=json:artifacts/coverage.json --junitxml=artifacts/junit.xml
python -m belieftown notebooks
python -m belieftown reproduce --profile quick --output artifacts/quick_reproduction --notebooks
python -m belieftown verify artifacts/example_audit.json
```

Optional browser/media rebuilding additionally needs the `browser` extra, Chromium, and ffmpeg for MP4 encoding. Build a wheel with `python -m pip wheel --no-deps . --wheel-dir dist`, then run `python scripts/wheel_smoke.py`. Re-running timing experiments can change timing measurements; retain the environment and protocol when comparing results.

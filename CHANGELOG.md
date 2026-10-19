# Changelog

## Unreleased

- Redesign the studio with a monochrome palette, simpler hierarchy, and collapsible evidence and diagnostic panels; refresh the recorded HTML showcase and browser screenshots.
- Rework the graph encoder, rankers, data generation, training, and benchmark modules for readability while preserving the released feature layout and checkpoint schema.
- Rewrite the model and data cards, clarify the mathematical specification, and link the technical research docs from the README.
- Keep one generated standalone showcase, remove duplicate offline HTML copies, and refresh the walkthrough and README screenshots.
- Remove the software test suite, test-only workflow and historical test outputs; keep the final research notebooks, trained models, and benchmark results.

## GitHub publication — 2026-10-19

- Publish the MIT-licensed project at `VeinDevTtv/BeliefTown` with a minimal quick start, package metadata, issue forms, pull request template, and dependency update configuration.
- Explicitly read and write UTF-8 text throughout the application and supporting tools, fixing Windows failures in the local server, showcase export, and benchmark report.
- Verify the notebooks, installed-wheel behavior, and live browser patch/replay workflow on Windows with Python 3.13.

## 1.0.0 — 2026-10-18

Initial complete research release: finite-world simulator; log-domain timestamped filtering; provenance-preserving communication; private agent views; expected-cost route planning; diagnostic replay; legal information patches; exhaustive bounded search; independent audit recomputation; graph/MLP/ablation training; scene-held-out benchmark; browser and notebook interfaces; CLI; documentation; and original example scenes.

Numerical stability, alternate-path reconstruction, terminal-action semantics, candidate-bound disclosure, and the distinction between static and live demonstrations are documented in the implementation and research notes.

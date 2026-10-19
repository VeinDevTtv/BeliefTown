# Security boundary

BeliefTown is a **local development and research tool**, not a hardened hosting platform. Keep the bundled server on loopback. It intentionally does not expose a public-interface binding option. It rejects unexpected Host headers, foreign Origin headers, missing session tokens, oversized JSON bodies, and arbitrary filesystem routes.

These controls do not turn Python’s development HTTP server into a production web server. Do not put it on an untrusted network or load untrusted Python notebooks. The official Python documentation warns that `http.server` implements only basic security checks.

Scene imports are structured JSON, never evaluated Python. Size and finite-world bounds reduce accidental excessive computation, but large exact searches still consume CPU. Model loading uses PyTorch `weights_only=True`; use the included checkpoints or models you created, rather than treating arbitrary external model files as safe.

Report security-sensitive defects privately through [GitHub vulnerability reporting](https://github.com/VeinDevTtv/BeliefTown/security/advisories/new). Include the affected version, reproduction steps, and expected impact. Please avoid disclosing exploit details in a public issue until the report has been reviewed. Ordinary bugs belong in GitHub Issues.

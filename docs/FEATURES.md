# Release feature and verification map

This table maps the agreed bounded first release to implementation and evidence. “Implemented” means working code and the listed checks, not an assertion of universal correctness.

| Requirement | Implementation | Verification evidence |
|---|---|---|
| Four-room/three-NPC/one-parcel demonstration | `scenarios.py`, packaged JSON, `world.py` | Canonical expected route and one-message patch; browser replay |
| Configurable movement, observations, inspections and structured messages | `schema.py`, `world.py` | Schema, event ordering, possession, generated inspection tests |
| Private agent inputs and privileged inspector separation | `AgentView`, `Snapshot`, `planner.py` | Hidden-world changes with matched inputs; graph no-truth feature test |
| Normalized, uncertain, time-aware beliefs | `beliefs.py` | Independent latent-path enumeration, timestamp, impossible-evidence, tiny-likelihood tests |
| Root provenance and duplicate idempotence | `world.py`, `beliefs.py` | Duplicate/independent evidence tests and live manual delivery check |
| Cost-based action selection | `planner.py` | Exhaustive order comparison, ties, shortest paths, matrix checks |
| Pause, step, play, reset, matched timelines | `studio.py`, `static/app.js` | Browser step and replay checks |
| Belief bars, entropy, route rankings, evidence chain, event log | `Snapshot.inspector`, browser assets | Desktop/mobile screenshot and DOM acceptance checks |
| Past observation/transmission suppression | `replay.py`, `Episode` | Downstream invalidation, alternate path, unchanged physical trajectory tests |
| Simultaneous legal evidence-delivery patches | `interventions.py` | Original ownership, no within-batch relay, duplicate/protected/illegal tests |
| Finite minimum-patch search | `search`, `PatchProblem` | Exhaustive matching, fixed-bound statuses, budget and feasible-first tests |
| Portable audit plus independent ranker-free recertification | `audit.py`, `verifier.py` | Round trips, altered hashes, false verdicts, nonminimum and unresolved status tests |
| Genuine graph learning and exact labels | `learning/` | Training/reload tests, nine trained release checkpoints, executed training notebook |
| Strong non-graph baseline and provenance ablation | MLP and `graph_no_provenance` | Same data/splits/verifier budgets, three seeds per variant |
| Held-out benchmark with uncertainty | `benchmark.py` | 792 actual searches; raw rows, scene bootstrap, hashes |
| Six scenario families and larger unseen graphs | `scenarios.py`, `learning/data.py` | 216 whole-scene split hashes and family-balanced manifests |
| Four executable Jupyter notebooks | `notebooks/`, notebook runner | All cells executed with errors disallowed and saved figures |
| Local application and no-install recorded export | `server.py`, `export.py` | HTTP auth/origin tests; real-service browser binding; offline replay check |
| Import/export and procedural generation | CLI and Studio | Scene round trips, file export and CLI workflow tests |
| Reproducible package and command-line workflow | `pyproject.toml`, CLI, scripts | Editable install, single-command quick pipeline, wheel smoke test |
| Open-source licensing, contribution and publication material | Root documentation and `docs/` | Final release link/integrity checks |

The feature descriptions still refer to the 1.0.0 release behavior. The screenshots, standalone showcase, and walkthrough now show the monochrome redesign. A browser smoke walkthrough confirmed the initial scene, minimum-patch search, patch application, replay step, and mobile layout; the complete automated browser suite has not been rerun for this visual update.

## Deliberately outside this release

The design narrowed to a finite debugger, not an unbounded town. Online source-trust learning, strategic message selection, free-form dialogue, nested theory of mind, moving targets during search, multiple interacting parcel variables, voice/3D art, commercial-engine plugins, and public multi-user deployment were excluded from version one. They are not placeholders in the code or hidden prerequisites for the delivered workflow.

The graph network is fully implemented and trained, but a speed advantage is **not** an implemented feature: it is a hypothesis that the released measurements did not support. No fabricated improvement is needed to use the debugger.

See `VERIFICATION.md` for the executed release gates and environment-specific limits.

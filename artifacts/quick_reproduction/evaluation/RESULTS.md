# Measured benchmark results

These are measurements from the bundled finite synthetic benchmark, not claims about arbitrary games.
The statistical unit is an independent scene. 1 trained seed(s) are averaged within scene before bootstrap confidence intervals.

| Split | Method | Calls, mean | Latency, mean (ms) | Solvable goals found ≤4 calls | Exactness |
|---|---|---:|---:|---:|---:|
| ood | graph | 78.33 | 83.85 | 0.0% | 100.0% |
| ood | graph_no_provenance | 78.50 | 84.47 | 0.0% | 100.0% |
| ood | heuristic | 79.00 | 51.31 | 0.0% | 100.0% |
| ood | mlp | 78.50 | 72.64 | 0.0% | 100.0% |
| ood | uninformed | 78.00 | 45.18 | 100.0% | 100.0% |
| test | graph | 45.50 | 37.68 | 100.0% | 100.0% |
| test | graph_no_provenance | 45.50 | 35.12 | 100.0% | 100.0% |
| test | heuristic | 45.50 | 20.07 | 100.0% | 100.0% |
| test | mlp | 45.50 | 29.85 | 100.0% | 100.0% |
| test | uninformed | 45.50 | 20.53 | 100.0% | 100.0% |

## Interpretation

A smaller verifier-call count does not establish a speedup. Neural encoding and scoring are included in request latency. No-solution certificates still require exhausting the candidate universe.

The released benchmark is deliberately small. Out-of-distribution (OOD) cases change room count, graph density, source reliability, and mobility. This is a bundled distribution shift, not a controlled estimate of each factor separately.

Full confidence intervals, paired differences against the heuristic, individual scene rows, manifest hashes, and checkpoint hashes are in the JSON and CSV files.

## Belief calibration proxy

| Split | Belief method | Brier score, mean |
|---|---|---:|
| test | provenance | 0.35434 |
| test | naive | 0.41951 |
| test | last_report | 0.63845 |
| ood | provenance | 0.65613 |
| ood | naive | 0.65613 |
| ood | last_report | 0.62451 |

Belief scores use the same received histories at the initial search checkpoint. They do not compare divergent rollouts or measure human believability.

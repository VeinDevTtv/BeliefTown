# Measured benchmark results

These are measurements from the bundled finite synthetic benchmark, not claims about arbitrary games.
The statistical unit is an independent scene. 3 trained seed(s) are averaged within scene before bootstrap confidence intervals.

| Split | Method | Calls, mean | Latency, mean (ms) | Solvable goals found ≤4 calls | Exactness |
|---|---|---:|---:|---:|---:|
| ood | graph | 53.57 | 44.84 | 86.7% | 100.0% |
| ood | graph_no_provenance | 53.57 | 40.25 | 100.0% | 100.0% |
| ood | heuristic | 53.71 | 23.75 | 80.0% | 100.0% |
| ood | mlp | 53.58 | 32.81 | 80.0% | 100.0% |
| ood | uninformed | 54.00 | 20.01 | 60.0% | 100.0% |
| test | graph | 45.40 | 31.28 | 75.8% | 100.0% |
| test | graph_no_provenance | 45.46 | 31.59 | 72.7% | 100.0% |
| test | heuristic | 45.35 | 15.49 | 81.8% | 100.0% |
| test | mlp | 45.33 | 25.51 | 81.8% | 100.0% |
| test | uninformed | 46.15 | 16.29 | 72.7% | 100.0% |

## Interpretation

A smaller verifier-call count does not establish a speedup. Neural encoding and scoring are included in request latency. No-solution certificates still require exhausting the candidate universe.

The released benchmark is deliberately small. Out-of-distribution (OOD) cases change room count, graph density, source reliability, and mobility. This is a bundled distribution shift, not a controlled estimate of each factor separately.

Full confidence intervals, paired differences against the heuristic, individual scene rows, manifest hashes, and checkpoint hashes are in the JSON and CSV files.

## Belief calibration proxy

| Split | Belief method | Brier score, mean |
|---|---|---:|
| test | provenance | 0.45087 |
| test | naive | 0.47086 |
| test | last_report | 0.59525 |
| ood | provenance | 0.53260 |
| ood | naive | 0.57283 |
| ood | last_report | 0.61653 |

Belief scores use the same received histories at the initial search checkpoint. They do not compare divergent rollouts or measure human believability.

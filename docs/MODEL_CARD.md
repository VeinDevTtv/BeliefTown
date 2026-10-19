# Model card: candidate-patch rankers, release 1.0.0

## Purpose and limits

The models estimate which candidate evidence-delivery patches are likely to produce a specified NPC search action. They are **search-ordering assistants**, not language models, character personalities, autonomous game designers, or certificates. The exact simulator and verifier determine feasibility and minimum cardinality.

Use these checkpoints only with the accompanying feature encoder and schema. The appropriate deployment is the bounded local research debugger. No performance claim is made for arbitrary commercial games, natural-language messages, real human trust, or large multi-object worlds.

## Released models

| Variant | Architecture | Parameters | Seeds |
|---|---|---:|---|
| `graph` | Two relational message-passing layers, hidden size 32, mean pooling, target/destination embeddings, global features | 52,961 | 0, 1, 2 |
| `mlp` | Globals plus per-feature node means/maxima; dense hidden layers 64 and 32; no edges | 5,633 | 0, 1, 2 |
| `graph_no_provenance` | Same graph network, selected provenance edges and direct duplication features removed | 52,961 | 0, 1, 2 |

Graphs have 20-dimensional node features, 14 global features, and 21 directed relation types. Node types are rooms, actors, root observations, received messages, and proposed deliveries. The score head combines pooled graph information with the target actor and requested-room embeddings. IDs are used to establish relations, not provided as learned numeric features. Hidden parcel truth is not encoded.

A strong scalar likelihood heuristic is included in the full graph and MLP inputs. Therefore any gain must be demonstrated over that heuristic; this is not a comparison of “reasoning from nothing” against a weak baseline. The provenance-removal variant also removes selected related global signals, so it is a bundled ablation, not a pure single-edge intervention.

## Training

Training uses 7,104 candidate patches from 120 scenes; validation uses 1,372 patches from 24 completely separate scenes. Every patch up to the configured cardinality is labeled by actual execution. All feasible alternatives are positive; the target is not a single arbitrarily chosen demonstration.

Each of the nine release runs used 14 epochs, batch size 64, AdamW at learning rate 0.002, weight decay 0.0001, binary cross-entropy with logits, and gradient clipping at norm 5. There is no test-driven model selection. The saved epoch minimizes validation BCE. Full epoch curves and metadata are in the JSON adjacent to each checkpoint.

The implementation seeds Python, NumPy, and PyTorch, uses one CPU thread for neural work, and enables deterministic algorithms. Bitwise reproducibility across different PyTorch versions, platforms, and hardware is not promised. Checkpoint and dataset hashes identify the delivered experiment.

The application chooses among bundled checkpoints by validation loss only. The benchmark reports each of all three seeds; it does not report only the selected application's checkpoint. The separate three-epoch notebook exercise is not used to produce release benchmark rows.

## Evaluation and actual finding

The released test set contains 48 scenes; a further 24 scenes form a bundled distribution shift with larger room graphs and changed generation conditions. All candidate variants from one scene stay together. The statistical unit is a scene, with trained seeds averaged within scene before bootstrap intervals.

**The graph ranker did not demonstrate a runtime advantage over the heuristic.** Neural scoring overhead exceeds the small changes in verifier-call count at this scale. Heuristic ordering remains the default. See `artifacts/evaluation/RESULTS.md`, its JSON confidence intervals, and raw scene rows rather than quoting rounded numbers from this card.

All 792 evaluated search executions agreed with exhaustive candidate labels on feasibility/minimality. This tests the correctness architecture on those cases; it does not mean that the learned classifier is 100% accurate. A classifier error changes ordering, not the definition of a valid patch.

Most benchmark goals are infeasible within bounds, which limits how much ranking can improve certification work. No ranker can certify a no-solution layer without exhausting it under this algorithm. Fixed-call-budget success and total wall-clock latency answer different questions.

## Failure modes

The learned score can be poorly calibrated, rank a useful message late, generalize poorly to a novel evidence graph, or add more overhead than it saves. The graph sees all designer-available roots, not only the target NPC's current evidence, because its task is choosing deliveries. That privilege must not be misrepresented as the NPC's own information access.

Duplicate handling is exact only for supplied provenance identities. The model does not discover unknown common sources, detect human deception, infer omitted channels, or resolve statistical dependency between separately identified roots.

## Files, license, and safe use

Checkpoints and histories are in `artifacts/checkpoints/`. They were trained from scratch on this project's synthetic scenes; no third-party pretrained model or human dataset is included. The original checkpoints use the repository's MIT license. PyTorch retains its separate license.

`Predictor.load()` uses tensor-oriented `weights_only=True` loading. Load trusted files with matching software/schema, keep models local, and retain exact verification after ranking. Removing verification or adding neural pruning changes the guarantees.

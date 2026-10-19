# Model card: information-patch rankers

This card describes the three candidate rankers released with BeliefTown 1.0.0.
It summarizes what they learn, how they were evaluated, and where their results
should not be applied.

## Intended use

Each model scores a proposed delivery of existing evidence. The score helps the
exact search procedure decide which candidate patches to check first. The
models are not language models, NPC personalities, or autonomous designers.
They do not determine whether a patch is legal, whether it achieves the
requested action, or whether it is minimum. Those claims come from simulator
execution and bounded exhaustive verification.

The checkpoints are intended for the local research debugger and the bundled
finite scene format. They have not been evaluated on commercial games, natural
language, human trust, or large worlds with multiple interacting objects.

## Model variants

| Variant | Architecture | Parameters | Seeds |
|---|---|---:|---|
| `graph` | Two relational message-passing layers, hidden size 32, mean pooling, target and destination embeddings, and global features | 52,961 | 0, 1, 2 |
| `mlp` | Dense layers of width 64 and 32 using global features and pooled node features; no graph edges | 5,633 | 0, 1, 2 |
| `graph_no_provenance` | Same graph architecture, with selected provenance relations and related duplicate indicators removed | 52,961 | 0, 1, 2 |

The graph encoder uses 20 features per node, 14 graph-level features, and 21
directed relation types. Nodes represent rooms, actors, original observations,
received messages, and proposed deliveries. The score head combines a pooled
graph representation with embeddings for the target actor and requested room.
Identifiers establish relationships but are not numeric model inputs. The
parcel's hidden location is never encoded.

The full graph and MLP models include a likelihood-based heuristic feature.
Their performance must therefore be compared with that strong baseline. The
provenance ablation removes a group of connected features, so it cannot isolate
the causal effect of one relation or feature.

## Training

The training split contains 7,104 candidate patches from 120 scenes. The
validation split contains 1,372 patches from 24 different scenes. For a scene
and proposed patch \(\Delta\), the target is

\[
y(\Delta)=\mathbb{1}[\text{exact execution meets the requested action and constraints}].
\]

Every feasible patch is labeled positive; the data does not nominate one
arbitrary successful patch as the sole target. A candidate's label comes from
simulator execution, not from a model prediction.

Each of the nine release runs used 14 epochs, batches of 64, AdamW with learning
rate \(0.002\) and weight decay \(10^{-4}\), binary cross-entropy with logits,
and gradient clipping at norm 5. The selected checkpoint is the epoch with the
lowest validation loss. Test and out-of-distribution scenes do not take part in
checkpoint selection. Per-epoch metrics and run metadata accompany each
checkpoint.

Training seeds Python, NumPy, and PyTorch, uses one CPU thread, and enables
deterministic algorithms. This does not guarantee bit-for-bit identical results
across different PyTorch versions, platforms, or processors. Dataset and
checkpoint hashes identify the released files.

The application selects a bundled checkpoint using validation loss. The
benchmark reports all three seeds for each model type rather than only the
checkpoint selected by the application. The three-epoch notebook exercise is
an integration example; it did not produce the release results.

## Evaluation and findings

The held-out test set contains 48 scenes. Another 24 scenes form a bundled
distribution shift: they use larger room graphs and different generation
settings. Every variant from one scene stays in the same split. Statistical
intervals use scenes as the independent unit, averaging the three trained
seeds within each scene before bootstrapping.

The graph ranker did not show a runtime advantage over the heuristic on the
released benchmark. At this scale, graph construction and scoring cost more
than the modest changes in verifier calls. The heuristic therefore remains the
default. See [`artifacts/evaluation/RESULTS.md`](../artifacts/evaluation/RESULTS.md)
for measurements, confidence intervals, and raw scene-level results.

All 792 evaluated searches agreed with exhaustive feasibility and minimum-size
labels. This is evidence about the verifier on those cases; it is not a claim
that the learned classifier is always correct. Ranking errors can change search
efficiency, but cannot change the verifier's decision.

Many benchmark goals have no solution within the configured bounds. Under the
released search procedure, certifying that outcome requires checking every
candidate in the bounded space. Candidate ordering cannot avoid that work.
Verifier-call counts and wall-clock latency are therefore separate measures.

## Known limitations

The scores may be poorly calibrated, put a useful patch late in the search, or
generalize poorly to unfamiliar evidence graphs. Encoding and inference may
cost more than they save. The graph ranker can see evidence roots held by any
actor because its task is to help the designer choose a delivery. This access
is not part of the NPC's private information.

Provenance handling relies on identifiers supplied by the scene. The models do
not discover hidden common sources, detect deception, infer missing channels,
or estimate dependence between separately identified roots.

## Files and use

The checkpoints and histories are in `artifacts/checkpoints/`. They were
trained from scratch on synthetic scenes created by this project; no pretrained
third-party model or human dataset is included. The project code and original
checkpoints use the MIT license. PyTorch has its own license.

`Predictor.load()` uses PyTorch's `weights_only=True` option. Use a checkpoint
with the matching feature encoder and schema, and retain exact verification
after ranking. Pruning candidates based on the learned score would change the
search guarantees.

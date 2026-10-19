# Data card: synthetic information-patch benchmark

## Source and contents

The dataset is generated entirely by this repository. It contains fictional
rooms, non-player characters, parcel locations, and structured observations.
It includes no personal information, scraped text, extracted game assets,
human annotations, or external datasets. The generator and original data are
released under the MIT license.

The generator seed is `20261018`; each split uses a separate seed range. One
JSONL row represents one complete scene, one intervention goal, and every
labeled patch in that scene's bounded candidate space. A scene, its evidence
history, and its alternate patches are kept together in a single split.

| Split | Scenes | Candidate patches | Feasible patches |
|---|---:|---:|---:|
| Train | 120 | 7,104 | 599 |
| Validation | 24 | 1,372 | 142 |
| Test | 48 | 2,935 | 329 |
| OOD | 24 | 1,590 | 64 |
| **Total** | **216** | **13,001** | **1,134** |

The number of feasible patches is not the number of scenes with a solution: a
single scene can have multiple successful patches. Positive labels are rare,
so accuracy from a classifier that always predicts “infeasible” would be
misleading. The release trains binary cross-entropy scores and evaluates the
resulting search behavior instead of promoting raw classification accuracy.

## Scene design

The six balanced scene families cover missed relocations, delayed reports,
repeated-root rumor loops, independent corroboration, conflicting evidence,
and information constraints. Some goals already match the character's current
action, making the empty patch optimal. Others have no solution within the
allowed candidate set. Constrained scenes alternate between ordinary and
protected-target configurations.

Training, validation, and test scenes have four rooms. OOD scenes have five or
six rooms and also vary graph density, source reliability, and pre-search
mobility. These factors change together, so this split is a bundled distribution
shift, not a controlled estimate of each factor's individual effect.

Procedural priors, travel costs, reports, and requested destinations vary.
Some actor priors are subjective and do not match the generator's initial
distribution for the parcel. Brier scores therefore compare belief policies on
these scenes; they do not establish that the configured posterior is globally
calibrated.

Only the Courier actively searches in the procedural benchmark. Other actors
store and forward evidence. The authored browser demonstration has three active
characters, so the benchmark focuses on choosing interventions for one target
character rather than emergent multi-agent social behavior.

## Labels

For each scene, the generator enumerates every permitted delivery subset of
size zero through three, subject to the candidate limit of 24. Exact execution
checks evidence ownership, communication channels, simultaneous-delivery
rules, information constraints, and the target character's resulting action.
Each candidate receives the binary label

\[
y(\Delta)=\begin{cases}
1,&\text{if executing patch }\Delta\text{ satisfies the requested action and constraints},\\
0,&\text{otherwise.}
\end{cases}
\]

Labels come from the reference simulator, not from an LLM or a learned ranker.
Both labels and the main verifier share the finite simulator's assumptions. An
independent belief-path calculation, exhaustive route comparison, and
adversarial audit checks provide additional evidence, but not a separate formal
proof of the entire implementation.

## Integrity and reproduction

`benchmarks/data/manifest.json` records split sizes, class counts, scene-family
counts, and SHA-256 hashes. Training metadata records the manifest hash, and
the evaluation report records the checkpoint hashes. Notebook 04 checks these
links.

Rebuild the released-size dataset with:

```bash
python -m belieftown dataset --output artifacts/rebuilt-data
```

This command uses the released sizes and generator seed by default. Even when
the generated data and model computation are deterministic, measured latency
can vary with hardware, software versions, and system load.

## Appropriate interpretation

This small synthetic dataset is a controlled testbed for the declared evidence
model and bounded patch search. It does not measure human believability, player
enjoyment, psychological theory of mind, unrestricted natural-language
reasoning, or how often bugs occur in production games.

Do not repeatedly tune a method against the released test scenes and continue
to describe those scenes as untouched. New method development should add
independent held-out seeds or publish a newly versioned benchmark.

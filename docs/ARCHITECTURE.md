# Architecture and public interfaces

## Data flow

```text
Scene JSON ──validate──> Scene
                           │
                           ▼
                      Episode replay
                           │
                           ▼
                   Snapshot at a turn
                     /            \
         restricted AgentView      privileged designer service
                │                       │
          belief vector          evidence graph + patch candidates
                │                       │
        search-order planner       optional learned ordering
                │                       │
              action             exact execution verifier
                │                       │
          next world tick         patch + audit record
                                        │
                              fork and replay episode
```

There are two very different consumers of state. An NPC acts from its observations and permitted geometric/configuration inputs. The designer can inspect the entire simulated scene. The learned patch ranker is a designer-side search assistant; it is not the NPC's hidden thought process. Actual parcel truth is excluded even from its graph features.

## Modules

| Module | Responsibility |
|---|---|
| `schema.py` | Immutable scene records, strict validation, canonical hashes, serialization |
| `beliefs.py` | Log-domain time-aware filtering and an independent latent-path reference calculation |
| `planner.py` | Shortest paths, fixed-order cost matrix, deterministic minimum expected-cost route |
| `randomness.py` | Semantically keyed random streams |
| `world.py` | Private receipts, generated inspections, event-sourced snapshots and actions |
| `interventions.py` | Candidate delivery space, legal simultaneous patches, exhaustive bounded search |
| `verifier.py` | Recheck exported results without trusting their ranker or stored verdict |
| `audit.py` | Portable scene/episode/patch bundles |
| `replay.py` | Diagnostic forks and matched branch comparison |
| `learning/graph.py` | Typed evidence-graph encoding and provenance-removal ablation |
| `learning/model.py` | Small relational GNN, tabular MLP, batching, safe checkpoint loading |
| `learning/data.py` | Procedural whole-scene splits and exact candidate labels |
| `learning/train.py` | CPU training, validation selection, histories, checkpoint metadata |
| `benchmark.py` | Fair scene-paired evaluation, bootstrap intervals, raw results |
| `studio.py` | Shared stateful application operations, independent of the browser transport |
| `server.py` | Local HTTP interface, session token, host/origin restrictions, notebook wrapper |
| `export.py` | Self-contained recorded HTML showcase |
| `cli.py` | Reproducible command-line workflows |

## Python example

```python
from belieftown.scenarios import demo_scene
from belieftown.world import Episode
from belieftown.interventions import PatchProblem, search
from belieftown.audit import make_bundle, verify_bundle

scene = demo_scene()
episode = Episode(scene)
snapshot = episode.at(scene.search_start)
problem = PatchProblem(target_actor="courier", target_room=1, max_size=3)
result = search(snapshot, problem, method="heuristic")
bundle = make_bundle(episode, snapshot.time, result)
checked = verify_bundle(bundle)
assert result.status == "verified_minimum_within_bounds"
assert result.patch is not None
print(result.patch, checked)
```

For an application, use `Studio`, which enforces UI lifecycle rules such as one actionable batch at a logical tick. For experiments, use the lower-level immutable scene and snapshot functions and preserve their explicit assumptions.

## Temporal semantics

A snapshot is the information and physical state just before decisions at a logical turn. Declared observations and scheduled deliveries for that turn are available. An actor then moves along one graph edge or inspects its current room. The resulting physical state and inspection evidence appear in the next snapshot.

An edge's numeric cost contributes to expected search effort; an edge traversal always consumes one logical turn. This is not a real-time locomotion simulator. Inspection has a separate cost and consumes a turn. Finding the parcel stops that actor's search but does not remove the parcel from other actors' worlds.

Declared sensor records are exogenous measurements: they are not a camera rendering of the actor's current room. Generated room inspections, unlike declared records, follow the actor's actual actions. Communication channels and scheduled attempts are also exogenous. Silence and strategically selected messages are not part of the likelihood model.

## Trust boundaries

Input JSON is validated before simulation. A legal patch is checked against possession and channels at its original checkpoint, then applied to a clone as one transaction. A second message in a simultaneous patch cannot depend on the first one to create a new relay.

Returned search records carry hashes and exact verification results. Rechecking a record reconstructs its candidate universe and decision. For a minimum claim it also checks every smaller subset. This is independent of neural scores but not an unrelated formal implementation of the simulator; it shares the stated belief and planning semantics.

The ranker may inspect the designer's evidence graph, but its output only changes candidate ordering. It cannot prune candidates in a certified exhaustive run, change their legality, or write a verdict.

## Serialization and compatibility

Scenes and portable bundles have explicit schema versions. Canonical JSON is hashed with SHA-256. JSON export rejects non-finite values. Generated runtime roots/messages use reserved namespaces distinct from declared IDs.

Checkpoints contain a tensor state dictionary and simple metadata, not a serialized live application. Loading uses `weights_only=True`. That narrows the loading mechanism; use checkpoints from trusted sources and supported software versions.

## Application and notebook lifecycle

`StudioServer` is a context manager for the local application. The CLI keeps its owner alive until interrupted. `launch_notebook()` returns a live server wrapper with notebook HTML representation and a separate-tab URL. Closing the kernel closes the owner process.

The browser uses only bundled HTML, CSS, and JavaScript. It does not fetch fonts, models, trackers, or CDN libraries. The recorded exporter embeds precomputed snapshots and audit records instead of embedding a second approximate Python interpreter.

## Extending safely

Introduce a new likelihood through the observation schema and corresponding reference tests. Introduce a new patch operation by extending its formal legality, simultaneous semantics, audit representation, and exhaustive oracle before adding UI controls. Introduce a new learned ranker without changing verifier behavior.

Changing to moving targets, endogenous message policies, multi-object states, or nested beliefs is a model change, not a cosmetic extension. Such work needs a new mathematical contract and evaluation; the current certificates must not be relabeled as guarantees for it.

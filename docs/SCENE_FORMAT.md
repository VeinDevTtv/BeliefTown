# Scene format, generation, and evidence contracts

A scene is UTF-8 JSON with `schema_version: 1`. The files in `scenarios/` are complete examples. Do not copy only a fragment below into the application and expect missing fields to be inferred.

Generate a valid starting point:

```bash
python -m belieftown generate --seed 41 --rooms 5 --family delayed_report --output scenarios/example.json
```

Validate an edited file:

```python
from belieftown.schema import Scene
scene = Scene.load("scenarios/example.json")
print(scene.hash)
```

## Top-level fields

| Field | Meaning |
|---|---|
| `id`, `seed` | Scene identity and deterministic generation seed |
| `rooms` | Ordered distinct names; room references are zero-based integer indexes |
| `edges` | Undirected `[u, v, positive_cost]` connections forming a connected graph |
| `actors` | Character identity, initial room, normalized prior, active flag |
| `observations` | Original sensor records with stable provenance roots |
| `messages` | Scheduled transmissions of existing roots |
| `transitions` | One square matrix per transition from time t to t+1 |
| `truth` | Actual parcel room for every time 0 through horizon |
| `search_start`, `horizon` | Integer search activation and final logical turn |
| `channels` | Directed `[sender_id, recipient_id]` communication permissions |
| `inspection_cost` | Strictly positive score cost of checking a room |
| `family`, `description` | Human-readable scenario metadata |

The implementation accepts 2–7 rooms, up to 12 actors, and a horizon no greater than 200. These bounds limit factorial planning and exhaustive search. Search accepts at most 64 candidate deliveries, patch cardinality at most 5, and at most 250,000 candidate subsets. The normal interface uses a candidate limit of 24 and cardinality 3.

`transitions[t][next_room][current_room]` is a probability. **Columns**, not rows, must sum to one. There are `horizon` matrices and `horizon + 1` truth entries. The declared true movement must have nonzero probability. At and after search activation the target is stationary and transitions are identity matrices.

Actor priors may be subjective. A subjective prior is not guaranteed to be the true generating distribution. Report any mismatch when interpreting calibration results.

## Actor record

```json
{"id":"courier","name":"Courier","room":3,"prior":[0.32,0.32,0.32,0.04],"active":true}
```

The prior is the actor's distribution over the parcel's time-zero location. Inactive actors retain and forward observations but do not run the search controller. A patch target must be active and not already finished.

## Observation record

```json
{"id":"new-library","observer":"mira","time":3,"reported_room":1,"reliability":0.94,"kind":"sighting","likelihood":null}
```

A sighting with null `likelihood` uses the symmetric noisy-room model described in `MATH.md`. An explicit likelihood vector supplies the likelihood of this observed outcome under each candidate room. For example, a reliable empty-Hall observation uses `[1,1,1,0]` in the four-room scene. Likelihoods need not sum to one across hidden states; they must be finite, nonnegative, and not all zero. The realized truth must have positive likelihood.

`time` is the original observation time. It does not become the delivery time when forwarded. The root ID identifies one original measurement, not an English sentence. Two identical sentences from distinct independent measurements may have different roots. Two copies of one measurement must not.

Configured reliability is a sensor parameter. No deception, human honesty inference, or online source-reliability learning is implemented. The model assumes distinct roots are conditionally independent given the latent location history. Shared hidden errors between different roots require a richer model.

## Message record

```json
{"id":"old-to-courier","root_id":"old-workshop","sender":"theo","recipient":"courier","time":1,"parent_id":null}
```

Payload and original observer/time remain attached to the root. The original observer can send with no message parent. A forward references a received parent transmission. The receiver, root, and chronology of a declared parent chain must agree. Invalid channels, cycles, unknown roots, and unavailable senders are rejected.

During a diagnostic replay, a missing parent transmission may be replaced by a genuine alternate receipt already possessed by the sender; the replay logs the actual provenance path. It may not manufacture ownership.

IDs must be unique within their declared class. The prefixes `inspection:` and `patch:` are reserved for generated runtime records.

## Patch and audit interchange

A delivery is a `(sender, recipient, root_id)` operation. All ownership checks occur at the unmodified snapshot. A batch is simultaneous, so its order is not an opportunity to relay newly acquired information within the same batch.

Use the CLI to generate complete portable audit bundles rather than hand-authoring them:

```bash
python -m belieftown solve --scene scenarios/example.json --room 1 --output artifacts/custom-audit.json
python -m belieftown verify artifacts/custom-audit.json
```

An audit embeds the scene, episode suppressions and earlier patches, checkpoint time, candidate-space definition, resulting patch, verdict, and hashes. A `budget_exhausted` record does not contain a minimum certificate. The scope of a certificate is the recorded candidate set, including any deterministic candidate truncation.

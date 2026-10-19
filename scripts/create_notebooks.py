"""Author the four notebooks from reusable package APIs; no duplicated simulator."""
from pathlib import Path
from textwrap import dedent
import nbformat as nbf

root=Path(__file__).resolve().parents[1]
setup='''
from pathlib import Path
import sys, json, os
ROOT = Path.cwd().resolve()
if not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
if not (ROOT / "src" / "belieftown").exists():
    raise RuntimeError("Run this notebook from the source repository or its notebooks directory.")
sys.path.insert(0, str(ROOT / "src"))
RUN_ROOT = Path(os.environ["BELIEFTOWN_RUN_ROOT"]).resolve() if os.environ.get("BELIEFTOWN_RUN_ROOT") else None
DATA = RUN_ROOT / "data" if RUN_ROOT else ROOT / "benchmarks" / "data"
CHECKPOINTS = RUN_ROOT / "checkpoints" if RUN_ROOT else ROOT / "artifacts" / "checkpoints"
REPORT = RUN_ROOT / "evaluation" if RUN_ROOT else ROOT / "artifacts" / "evaluation"
AUDIT_DIR = RUN_ROOT if RUN_ROOT else ROOT / "artifacts"
OUTPUT = AUDIT_DIR / "notebook_run"
OUTPUT.mkdir(parents=True, exist_ok=True)
import numpy as np
get_ipython().run_line_magic("matplotlib", "inline")
import matplotlib.pyplot as plt
from IPython.display import display, HTML
'''

def md(s):return nbf.v4.new_markdown_cell(dedent(s).strip())
def code(s):return nbf.v4.new_code_cell(dedent(s).strip())
def save(name,cells):
    nb=nbf.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python','version':'3.13'},
        'belieftown':{'version':'1.0.0','network_required':False}})
    nbf.write(nb,root/'notebooks'/name)

save('01_beliefs_and_evidence.ipynb',[
md(r'''
# 01 · Beliefs are probability vectors—not shared omniscience

This notebook derives and executes the core model. A character sees only its own observations and received messages. The designer has a separate privileged inspector.

**What to look for:** twenty copies of one report should not have the same effect as two independent measurements. All calculations below use the same package as the game.

The complete assumptions and derivation are in [MATH.md](../docs/MATH.md).
'''),code(setup),
md(r'''
## A finite belief state

For $L$ rooms, the character maintains $\mathbf b=(P(X=1),\ldots,P(X=L))^\top$, with nonnegative entries summing to one. A noisy report has likelihood $q$ in the reported room and $(1-q)/(L-1)$ elsewhere.

Bayes’ rule is $b'(x)\propto P(y\mid x)b(x)$. Root IDs identify shared measurements; they do **not** assert that arbitrary real-world reports are independent.
'''),code('''
from dataclasses import replace
from belieftown.schema import Observation
from belieftown.beliefs import filter_belief, brute_force_belief, entropy

root = Observation("original", "mira", 0, 1, 0.8)
prior = np.ones(4) / 4
one = filter_belief(prior, [], [root], 0)
copies = filter_belief(prior, [], [root] * 20, 0)
independent = filter_belief(prior, [], [root, replace(root, id="independent")], 0)
np.testing.assert_array_equal(one, copies)
assert np.isclose(independent[1], 0.64 / (0.64 + 3 * (0.2 / 3) ** 2))
print("One report:", one.round(6))
print("Twenty copies:", copies.round(6))
print("Two independent reports:", independent.round(6))
'''),code('''
plt.figure(figsize=(7, 4))
plt.bar(["One report", "Twenty copies", "Two independent"], [one[1], copies[1], independent[1]])
plt.ylim(0, 1.05)
plt.ylabel("Probability assigned to the reported room")
plt.title("Independent corroboration is different from repetition")
plt.tight_layout()
plt.show()
'''),md(r'''
## A late message is evidence about the past

Let $T_t[x',x]=P(X_{t+1}=x'\mid X_t=x)$. After receiving old evidence, insert it at its observation time and recompute forward. Do not pretend it describes the present.

The implementation uses log probabilities internally, including a log-sum-exp prediction step, so very confident but conflicting evidence does not prematurely destroy support through underflow.
'''),code('''
# The parcel deterministically swaps between two rooms after turn zero.
swap = (((0.0, 1.0), (1.0, 0.0)),)
old = Observation("old", "mira", 0, 0, 0.9)
late_belief = filter_belief([0.5, 0.5], swap, [old], 1)
reference = brute_force_belief([0.5, 0.5], swap, [old], 1)
np.testing.assert_allclose(late_belief, reference)
print("Current belief after the old report:", late_belief)
assert np.isclose(late_belief[1], 0.9)
'''),md(r'''
## The actual game scene

At turn four the Courier has an old workshop report and a direct observation that the Hall is empty. Mira has an undelivered library observation. No policy reads the hidden parcel location.
'''),code('''
from belieftown.scenarios import demo_scene
from belieftown.world import Episode
scene = demo_scene()
episode = Episode(scene)
snapshot = episode.at(scene.search_start)
for agent in scene.actors:
    print(agent.name, snapshot.belief(agent.id).round(4), "next:", scene.rooms[snapshot.plan(agent.id).destination])
assert "new-library" not in snapshot.known("courier")
assert not hasattr(snapshot.view("courier"), "truth")
'''),md(r'''
## Expected-cost route planning is a matrix–vector product

For each room ordering $\sigma$, let $c_\sigma(x)$ be the travel-plus-inspection cost incurred before finding the parcel if it is in room $x$. Then

$$J(\sigma\mid\mathbf b)=\mathbf c_\sigma^\top\mathbf b,\qquad \mathbf j=\mathbf C\mathbf b.$$

With four rooms, there are $4!=24$ orders. This is an exact plan among fixed orders for the stationary search phase, not a globally optimal multi-agent POMDP policy.
'''),code('''
from belieftown.planner import route_matrix
orders, C = route_matrix(len(scene.rooms), scene.edges, snapshot.positions["courier"], scene.inspection_cost)
b = snapshot.belief("courier")
expected_costs = C @ b
best = int(np.argmin(expected_costs))
print("Cost-matrix shape:", C.shape)
print("Chosen route:", " → ".join(scene.rooms[i] for i in orders[best]))
print("Expected cost:", expected_costs[best])
print("Belief entropy, in nats:", entropy(b))
assert orders[best] == snapshot.plan("courier").order
'''),md('''
**Result:** the incorrect search destination follows from available evidence and the route objective. It is not an arbitrary dialogue prompt. Continue to notebook 02 to change that evidence without changing the world.
''')])

save('02_counterfactual_debugger.ipynb',[
md('''
# 02 · Find, verify, and replay a minimum information patch

This notebook executes the complete debugger workflow and launches the interactive interface. No external service is used. A minimum claim always refers to the bounded candidate universe recorded in the audit.
'''),code(setup),code('''
from belieftown.scenarios import demo_scene
from belieftown.world import Episode
from belieftown.interventions import PatchProblem, Delivery, search, apply_patch
from belieftown.audit import make_bundle, verify_bundle, save_bundle
from belieftown.replay import fork, compare_episodes
scene = demo_scene()
episode = Episode(scene)
snapshot = episode.at(scene.search_start)
problem = PatchProblem("courier", 1, max_size=3)
result = search(snapshot, problem, method="heuristic")
print(result.status)
print("Patch:", result.patch)
print("Verifier calls:", result.verifier_calls)
assert result.status == "verified_minimum_within_bounds"
assert len(result.patch) == 1
'''),md(r'''
## The optimization problem

$$\min_{\Delta\subseteq\mathcal U_t}|\Delta|\quad\text{subject to}\quad
\pi_i(\Phi_t(\Delta))=a^*,\;\operatorname{Legal}(\Delta),\;\operatorname{Constraints}(\Delta),\;|\Delta|\leq K.$$

All deliveries are validated against the **original snapshot**, prohibiting an unmodeled multi-hop relay within a simultaneous batch. The search checks all smaller patch sizes before certifying a minimum. Model scores can reorder candidates, never certify them.
'''),code('''
bundle = make_bundle(episode, scene.search_start, result)
save_bundle(bundle, AUDIT_DIR / "example_audit.json")
print(verify_bundle(bundle))
patched = fork(episode, patch=result.patch, time=scene.search_start)
comparison = compare_episodes(episode, patched)
assert comparison["same_physical_history"]
assert patched.at(4).plan("courier").destination == 1
print("First divergence:", comparison["first_divergence"])
'''),code('''
x = np.arange(len(scene.rooms))
plt.figure(figsize=(8, 4))
plt.bar(x - 0.18, snapshot.belief("courier"), 0.36, label="Original")
plt.bar(x + 0.18, patched.at(4).belief("courier"), 0.36, label="Verified patch")
plt.xticks(x, scene.rooms)
plt.ylabel("Belief probability")
plt.ylim(0, 1)
plt.title("The parcel did not move; the Courier received different evidence")
plt.legend()
plt.tight_layout()
plt.show()
'''),md('''
## A negative control and a no-solution case

A redundant copy must leave the belief unchanged. Protecting the target from all new deliveries makes a different target action unattainable in the candidate universe. Neither case should produce an invented patch.
'''),code('''
copy = apply_patch(snapshot, [Delivery("mira", "courier", "old-workshop")])
np.testing.assert_array_equal(snapshot.belief("courier"), copy.belief("courier"))
blocked = search(snapshot, PatchProblem("courier", 1, protected_actors=("courier",)))
assert blocked.status == "no_solution_within_bounds"
verify_bundle(make_bundle(episode, 4, blocked))
print("Duplicate evidence invariant: passed")
print("Constrained search:", blocked.status)
'''),md('''
## Launch the same Python-backed application inside Jupyter

Run locally and keep the kernel alive. The embedded application has live patch search, replay controls, belief inspection, diagnostic forks, manual deliveries, scene import, and audit export. No widget extension or JavaScript build step is required.

Saved notebook outputs cannot keep a Python server alive after the kernel shuts down. Rerun this cell to make the interface live again. In a hosted notebook service, the browser may not be able to reach the kernel’s loopback port; use local Jupyter or the recorded HTML showcase instead.
'''),code('''
from belieftown.server import launch_notebook
if "studio" in globals():
    studio.close()
studio = launch_notebook(scene, CHECKPOINTS)
display(studio)
# Verify the kernel-backed HTTP endpoint, not just the iframe markup.
import urllib.request
request = urllib.request.Request(studio.url + "api/state", headers={"X-BeliefTown-Token": studio.token})
with urllib.request.urlopen(request) as response:
    served = json.load(response)
assert served["scene"]["id"] == scene.id
print("Local Python endpoint verified:", studio.url)
'''),md('''
When finished, run `studio.close()` in a new cell. The session is also closed when this kernel exits. The browser application is a local development server, not a production deployment.
''')])

save('03_train_ranker.ipynb',[
md('''
# 03 · Train a graph model to order—not certify—candidate patches

The supervised target is whether a legal candidate batch produces the requested next action. Labels come from the exact simulator. Every candidate from one underlying scene stays in the same split.

This notebook performs a **three-epoch learning exercise**. The release contains separate fourteen-epoch runs for three model types and three seeds. Do not substitute the exercise checkpoint for the reported release checkpoints without rerunning evaluation.
'''),code(setup),code('''
from belieftown.learning.data import load_cases
from belieftown.learning.graph import encode_graph
from belieftown.learning.model import GraphRanker, Predictor, configure
from belieftown.interventions import candidate_deliveries, search
manifest = json.loads((DATA / "manifest.json").read_text())
train_cases = load_cases(DATA / "train.jsonl")
validation_cases = load_cases(DATA / "validation.jsonl")
assert not ({c.scene.hash for c in train_cases} & {c.scene.hash for c in validation_cases})
print("Training scenes:", len(train_cases))
print("Validation scenes:", len(validation_cases))
print("Training candidate labels:", manifest["splits"]["train"]["candidates"])
print("Positive candidate fraction:", manifest["splits"]["train"]["positive_candidates"] / manifest["splits"]["train"]["candidates"])
'''),md(r'''
## Relational message passing

Nodes represent rooms, actors, original observations, messages, and proposed deliveries. Edges retain their semantic types. No hidden parcel truth or arbitrary numeric identifier is a feature.

$$h_v^{(\ell+1)}=\operatorname{ReLU}\left(W_{self}^{(\ell)}h_v^{(\ell)}+
\sum_r\frac{1}{\max(1,|N_r(v)|)}\sum_{u\in N_r(v)}W_r^{(\ell)}h_u^{(\ell)}\right).$$

The implementation uses two layers, 32 hidden coordinates, 20 input features, 21 relation types, and 14 graph-level numerical features. A mean-pooled graph embedding is concatenated with target and destination embeddings. The non-graph comparator receives pooled node features and the same global features, including a strong likelihood heuristic.
'''),code('''
configure(0)
case = train_cases[0]
snap = case.snapshot
candidates = candidate_deliveries(snap, case.problem)
graph = encode_graph(snap, case.problem, candidates[:1])
model = GraphRanker()
print("Node-feature matrix:", graph.x.shape)
print("Typed edges:", graph.edges.shape)
print("Trainable parameters:", sum(p.numel() for p in model.parameters()))
plt.figure(figsize=(9, 4))
plt.imshow(graph.x, aspect="auto")
plt.xlabel("Node feature coordinate")
plt.ylabel("Node index (not a learned identity)")
plt.title("The graph input is a structured numerical representation")
plt.colorbar(label="Feature value")
plt.tight_layout()
plt.show()
'''),md(r'''
## Train against execution labels

With logits $s_\theta$ and $\hat p=\operatorname{sigmoid}(s_\theta)$, minimize binary cross-entropy:

$$\mathcal L=-\frac1M\sum_n\left[y_n\log\hat p_n+(1-y_n)\log(1-\hat p_n)\right].$$

The positive class is sparse. Accuracy alone would be misleading. Validation binary cross-entropy selects a checkpoint; neither the test split nor OOD cases select training epochs.
'''),code('''
from belieftown.learning.train import train_model
EPOCHS = 3
predictor = train_model(DATA, OUTPUT / "graph_quick.pt", kind="graph", seed=11, epochs=EPOCHS)
print("Selected epoch:", predictor.metadata["best_epoch"])
print("Validation BCE:", predictor.metadata["validation_bce"])
print("Training seconds:", predictor.metadata["training_seconds"])
history = predictor.metadata["history"]
plt.figure(figsize=(7, 4))
plt.plot([h["epoch"] for h in history], [h["train_bce"] for h in history], label="Training")
plt.plot([h["epoch"] for h in history], [h["validation_bce"] for h in history], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Binary cross-entropy")
plt.title("Quick notebook training run")
plt.legend()
plt.tight_layout()
plt.show()
'''),code('''
# Saving/loading preserves predictions; the exact verifier remains the authority.
loaded = Predictor.load(OUTPUT / "graph_quick.pt")
a = predictor.score(snap, case.problem, [()])
b = loaded.score(snap, case.problem, [()])
np.testing.assert_array_equal(a, b)
from belieftown.scenarios import demo_scene
from belieftown.world import Episode
from belieftown.interventions import PatchProblem
scene = demo_scene()
result = search(Episode(scene).at(4), PatchProblem("courier", 1), method="graph", ranker=loaded)
assert result.status == "verified_minimum_within_bounds"
print("Learned ordering, exact result:", result.status, result.patch)
'''),md('''
**Do not conclude that the model is faster because it returns a valid patch.** Its encoding and inference cost must be included. Notebook 04 examines the released paired benchmark, including the result that the heuristic is faster on these small scenes.
''')])

save('04_evaluation.ipynb',[
md('''
# 04 · Evaluate the claim—not just the demonstration

The release compares two non-neural search orders, three graph-model seeds, three MLP seeds, and three provenance-ablation seeds. The held-out unit is an entire scene, never individual near-duplicate candidate patches.

Results were measured on a small synthetic benchmark. OOD changes several factors together; it does not isolate the causal effect of each factor. This is not a user study of believable characters.
'''),code(setup),code('''
import csv, hashlib
results = json.loads((REPORT / "results.json").read_text())
with (REPORT / "search_results.csv").open() as f:
    rows = list(csv.DictReader(f))
assert results["all_searches_correct"]
assert all(int(row["correct"]) == 1 for row in rows)
assert all(float(row["minimum_gap"]) == 0 for row in rows if row["minimum_gap"] != "")
print("Search runs:", len(rows))
print("All recorded feasibility / bounded-minimum outcomes agree with exhaustive labels.")
print("Statistical unit:", results["protocol"]["split_unit"])
print("Training seeds in this report:", results["protocol"]["training_seeds"])
print("Report directory:", REPORT)
'''),md('''
## Matched comparisons and uncertainty

The analysis averages a method’s three training-seed outcomes **within each scene**, then bootstraps scene means for a 95% interval. It does not treat every candidate patch as an independent data point. Paired differences use the same scenes and the heuristic as the reference.

Verifier-call budgets are not wall-clock deadlines. End-to-end request latency includes graph construction and model scoring but excludes amortized model loading and scene construction.
'''),code('''
import html
lines = ["<table><tr><th>Split</th><th>Method</th><th>Mean calls</th><th>Mean latency (ms)</th><th>95% latency interval</th></tr>"]
for item in results["summary"]:
    latency = item["latency_ms"]
    lines.append(f"<tr><td>{html.escape(item['split'])}</td><td>{html.escape(item['method'])}</td><td>{item['verifier_calls']['mean']:.2f}</td><td>{latency['mean']:.2f}</td><td>[{latency['low']:.2f}, {latency['high']:.2f}]</td></tr>")
lines.append("</table>")
display(HTML("".join(lines)))
'''),code('''
test = [x for x in results["summary"] if x["split"] == "test"]
plt.figure(figsize=(9, 4))
plt.bar([x["method"] for x in test], [x["latency_ms"]["mean"] for x in test])
plt.ylabel("Mean steady-state search latency (ms)")
plt.xticks(rotation=18, ha="right")
plt.title("Measured candidate-ordering latency, including neural overhead")
plt.tight_layout()
plt.show()
'''),md(r'''
## Belief quality is a separate measurement

For an actual room $X$, the multiclass Brier score is

$$\mathrm{BS}=\frac1M\sum_n\sum_x\left(b_n(x)-\mathbb 1[X_n=x]\right)^2.$$

The belief comparison uses the **same initial received histories**, not divergent trajectories. Generated subjective priors vary across NPCs. The score is not a measure of human-likeness or a proof that a sensor model matches reality.
'''),code('''
for item in results["belief_summary"]:
    print(item["split"], item["mode"], "mean Brier:", round(item["brier"]["mean"], 5))
# Validate data provenance before interpreting any learned result.
manifest_path = DATA / "manifest.json"
manifest = json.loads(manifest_path.read_text())
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == results["dataset_manifest_sha256"]
for split, entry in manifest["splits"].items():
    path = DATA / f"{split}.jsonl"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
for name, expected in results["checkpoint_hashes"].items():
    path = CHECKPOINTS / f"{name}.pt"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
print("Dataset and checkpoint hashes: verified")
'''),md('''
## What the experiment supports

The exact search and replay workflow produces auditable results in its finite model. Learned ordering is implemented and measurable, but it is not faster than the heuristic on this release’s small scenes. Infeasible goals still require exhaustive bounded search for a no-solution certificate.

A responsible launch emphasizes the **inspect → intervene → verify → replay** workflow, publishes the negative result, and states the assumptions. The README and model/data cards do that. Rebuilding the experiment can change measured latency; update the report rather than reusing an old timing claim.
''')])
print('Created four notebooks')

# Start here

## 1. Choose how to run it

**No installation:** open `artifacts/BeliefTown_Showcase.html`. This single file contains six recorded scenes, synchronized replays, evidence inspection, and precomputed verified patches. It is intentionally labeled recorded; it does not pretend to run Python or train a neural network in your browser.

**Full application:** from the repository root, create and activate a virtual environment, then run:

```bash
python -m pip install -e ".[all]"
python -m belieftown doctor
python -m belieftown demo
```

The server prints its local address and opens your default browser. With no browser launcher, use `python -m belieftown demo --no-browser` and open the printed address yourself. Change a busy port with `--port 8766`. Stop the application with Ctrl+C.

The lightweight alternative is `python -m pip install -e .`. This installs the NumPy-only simulator, heuristic search, verifier, local interface, and HTML exporter. Neural ordering, training, and Jupyter require the corresponding optional dependencies.

The delivered build was exercised on Linux, Python 3.13.5, and CPU PyTorch. The package declares Python 3.10+, but other operating systems and Python versions were not locally executed. `requirements-lock.txt` records the build environment rather than promising a universal lock across platforms.

## 2. Reproduce the main demonstration

The default scene pauses at turn 4. Select Courier, and inspect its Workshop/Library probabilities. Its first search destination is Workshop; the actual parcel is in Library. The world marker is privileged designer information, not an input to the Courier's policy.

Set the target to **Courier → Library**, maximum patch size to 3, and leave ordering at **heuristic**. Press **Find minimum information patch**. The recorded candidate universe includes only existing, legally deliverable evidence.

The expected result is one message: `new-library`, from Mira to Courier. Press **Apply patch & replay**. The patched Courier now chooses Library. Play, step, or drag the timeline to inspect the two branches. Download the audit while its result is displayed.

Reset the scene. Use the manual delivery control to forward `old-workshop` from Mira to Courier. This creates a new transmission record, but it does not create a new independent observation. The Courier's belief vector is unchanged at that snapshot.

Reset again and protect Courier from new information. Request Library. The result is **no solution within bounds**, because the current next action differs and all nonempty deliveries to that target violate the constraint.

## 3. Try a diagnostic counterfactual

Reset. Suppress the root observation `new-library`, or suppress a historical transmission. The episode is regenerated with the same physical parcel trajectory. A missing observation removes the availability of its descendant messages. A missing transmission does not erase an independently available route to the same root.

Diagnostic forks can change the past evidence history. Actionable patches cannot: they deliver existing evidence at the selected present checkpoint. These are separate controls and separate meanings of counterfactual.

The interface permits one actionable delivery batch at a given tick. Reset, advance, or create a new branch before requesting another batch at that same tick. Do not confuse successive batches with the simultaneous minimum-patch problem.

## 4. Verify without trusting the interface

```bash
python -m belieftown solve --room Library --output artifacts/example-audit.json
python -m belieftown verify artifacts/example-audit.json
python -m belieftown replay artifacts/example-audit.json --output artifacts/example-replay.json
```

The verifier reconstructs the scene and snapshot, checks the candidate-space hashes and patch result, and exhausts smaller candidates when certifying minimality. A saved screenshot or a neural score is not a certificate.

## 5. Open the research notebooks

```bash
jupyter lab notebooks
```

Run the four notebooks in order. They contain executable mathematics, debugger operations, actual training, and the measured evaluation. The already executed copies are in `artifacts/executed_notebooks/`.

The live debugger's launch cell starts a loopback server owned by that notebook kernel. Keep the kernel running. Reopening an old saved notebook does not restart its old server; rerun the cell. A separate-tab link is provided alongside the embedded view. Hosted notebook platforms may require their own port support; this project does not expose a public tunnel.

## 6. Change the world and rerun the experiment

```bash
python -m belieftown generate --seed 42 --rooms 5 --family conflicting_evidence --output scenarios/custom.json
python -m belieftown demo --scene scenarios/custom.json
python -m belieftown reproduce --profile quick --tests
```

Use the scene-format guide before manually editing likelihoods, transition matrices, or provenance chains. The quick pipeline creates separate data/checkpoints/results; it does not replace the released experiment. Its fewer scenes and epochs make it an integration exercise, not the published benchmark.

## Troubleshooting

**PyTorch is unavailable:** install the `ml` or `all` extra, or use heuristic ordering. The core has no PyTorch dependency.

**A report cannot be certified:** preserve the original audit JSON, scene, and software version. Editing a scene or record invalidates its hashes. An unresolved budget status is not a failed certificate of a feasible patch; it is a declaration that search did not finish.

**A custom scene is rejected:** look for the field identified in the validation error. Common errors are row-stochastic rather than column-stochastic transitions, an impossible truth trajectory, a sender that does not possess the root, and nonstationary truth after search starts.

**The neural model is slower:** that is also the released result on these small scenes. Heuristic ordering is the default deliberately.

## Optional browser acceptance tests and launch media

The release includes `artifacts/walkthrough/BeliefTown_Walkthrough.mp4`, a captioned, silent 60-second walkthrough assembled from actual UI states. It is a paced capture sequence, not a claim of continuous real-time performance. `artifacts/walkthrough/manifest.json` records its source frames and captions.

To rerun browser checks, install the optional browser extra and Chromium:

```bash
python -m pip install -e ".[browser]"
python -m playwright install chromium
python scripts/browser_smoke.py
```

A system Chromium binary is also accepted; set `CHROMIUM_PATH` when needed. The supplied environment blocked loopback navigation in its browser, so those DOM tests used a transparent binding to the real Python application service; separate HTTP tests exercised authentication, origin checks, and API responses. See the verification report for the precise scope.

Rebuild the walkthrough with `python scripts/make_walkthrough.py`; this additionally requires ffmpeg. No music, external artwork, or font files are redistributed.

# Notebook guide

The notebooks import the same installed package as the CLI and browser application. They are executable research documents, not duplicated notebook-only simulators.

| Notebook | Executed work |
|---|---|
| `01_beliefs_and_evidence.ipynb` | Independent-vs-duplicated evidence calculation, plots, delayed evidence against a brute-force oracle, route cost matrix |
| `02_counterfactual_debugger.ipynb` | Exact patch search, portable audit verification, matched replay, negative controls, live application launch |
| `03_train_ranker.ipynb` | Split integrity, graph features, parameter counts, actual three-epoch training exercise, checkpoint reload, learned ordering with exact verification |
| `04_evaluation.ipynb` | Released results, uncertainty plots, Brier scores, data/checkpoint hash checks, honest interpretation |

## Installation and interactive use

```bash
python -m pip install -e ".[all]"
jupyter lab notebooks
```

Run all cells in order. The training notebook uses the bundled labeled data and writes its own exercise checkpoint to `artifacts/notebook_run/`; it does not overwrite the nine release checkpoints. Its three-epoch exercise is not the 14-epoch, three-seed release experiment.

The live application cell in notebook 02 starts a local server and validates its state endpoint. An embedded frame and a separate-tab link refer to that live session. Keep the kernel running. Saved executed outputs preserve figures and measurements, but an old live-server URL is not permanently hosted. Rerun the launch cell when reopening a notebook.

## Headless verification

```bash
python -m belieftown notebooks
```

This uses nbclient with errors disallowed and runs all four notebooks in fresh kernels. Results are saved to `artifacts/executed_notebooks/` with `execution_report.json`. The notebook source remains unexecuted and portable. The runner locates the repository from each notebook's directory.

Matplotlib figures are rendered inline; the release includes actual PNG figure outputs, not only textual figure objects. The live notebook server is intentionally tested during execution and ends with the execution kernel.

## Hosted notebook limitations

This release targets a local Python/Jupyter installation. A browser viewing a remotely hosted notebook cannot necessarily access a loopback server on that remote machine. The project does not attempt automatic tunnels, authentication bypasses, or public deployment. Use the static HTML showcase for a transport-independent demonstration, or run the full studio locally.

## NotebookLM / Gemini Notebook

This tool is optional for literature work and is not required to run BeliefTown. Suggested source material is `MATH.md`, `MODEL_CARD.md`, `DATA_CARD.md`, `REFERENCES.md`, the measured results, and the cited original papers obtained separately. Do not mistake an AI-generated literature summary for experimental evidence.

A useful research prompt is: “Separate implemented behavior, measured findings, mathematical assumptions, and proposed extensions. For every experimental statement, identify its supporting result table or code location. Identify any claim stronger than the evidence supports.”

No account, cloud notebook, or connected application was created as part of this release. The reproducible artifact is the version-controlled Jupyter project.

## Rebuilt experiments

`python -m belieftown reproduce --profile quick --output artifacts/rebuilt --notebooks` executes the notebooks against **that rebuilt run's** data, checkpoints, and evaluation, not the bundled release results. The runner passes an explicit run-root environment to each new kernel. The execution report records it.

To revisit an existing run, use:

```bash
python -m belieftown notebooks --run-root artifacts/rebuilt --output artifacts/rebuilt/executed_notebooks
```

Ordinary notebook execution without a run root uses the bundled release artifacts. Notebook prose identifies the original release's three-seed negative finding; a rebuilt report prints its own actual seed list and measurements.

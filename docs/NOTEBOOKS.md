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

## Execute the notebooks and save outputs

```bash
python -m belieftown notebooks
```

This uses nbclient with errors disallowed and runs all four notebooks in fresh kernels. Results are saved to `artifacts/executed_notebooks/` with `execution_report.json`. The notebook source remains unexecuted and portable. The runner locates the repository from each notebook's directory.

Matplotlib figures are rendered inline; the release includes the resulting figures, not only textual figure objects. The live notebook server runs only while its execution kernel is active.

## Hosted notebook limitations

This release targets a local Python/Jupyter installation. A browser viewing a remotely hosted notebook cannot necessarily access a loopback server on that remote machine. The project does not attempt automatic tunnels, authentication bypasses, or public deployment. Use the static HTML showcase for a transport-independent demonstration, or run the full studio locally.

## Rebuilt experiments

`python -m belieftown reproduce --profile quick --output artifacts/rebuilt --notebooks` executes the notebooks against **that rebuilt run's** data, checkpoints, and evaluation, not the bundled release results. The runner passes an explicit run-root environment to each new kernel. The execution report records it.

To revisit an existing run, use:

```bash
python -m belieftown notebooks --run-root artifacts/rebuilt --output artifacts/rebuilt/executed_notebooks
```

Ordinary notebook execution without a run root uses the bundled release artifacts. Notebook prose identifies the original release's three-seed negative finding; a rebuilt report prints its own actual seed list and measurements.

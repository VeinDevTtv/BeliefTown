# Contributing

Install the project from a source checkout with `python -m pip install -e ".[all]"`, then run `python -m pytest -q`. Keep the simulator importable without PyTorch or Jupyter. Optional dependencies must be imported only by the feature that needs them.

A bug report should include the scene JSON, command, Python/dependency versions, random seed, and a minimal replay or audit bundle. Do not include personal data, credentials, or private notebook sources. For algorithmic changes, state the proposed invariant and add a regression test before changing the implementation.

Changes to filtering must still match the independent path-enumeration reference on small worlds. Changes to search must preserve candidate membership across ordering methods and distinguish feasible from minimum. A candidate rejected by a model must never be pruned from an exact search. New communication semantics need explicit temporal and dependency rules, not just another graph edge.

A contribution to learned models should report validation selection, whole-scene splits, at least one strong non-neural baseline, end-to-end latency, and a failed or unhelpful case. Never tune on the bundled test set and continue calling it held out; version the generator and create a fresh split when appropriate. Publish negative findings.

Keep examples small and nonviolent. Maintain accessible labels in the UI. A new external asset must have a documented redistribution license and attribution. Do not add model downloads, analytics, remote code execution, or internet tunneling to the default workflow.

Submit focused changes to the repository where this project is hosted. Describe which tests were run and any tests that were not run. The CI configuration is supplied, but a successful local test run is not a claim that GitHub-hosted CI has already executed.

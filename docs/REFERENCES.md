# References and attribution

BeliefTown combines established probabilistic inference, graph learning, and finite combinatorial search in a small NPC-debugging artifact. It does not claim to have invented Bayesian character beliefs, provenance tracking, graph neural networks, counterfactual explanations, or minimum-change planning.

The following primary sources informed mathematical context or implementation practice. URLs are included here for reproducible project documentation; papers are not redistributed in the repository.

1. Battaglia, P. W., et al. (2018). **Relational inductive biases, deep learning, and graph networks.** https://arxiv.org/abs/1806.01261 . Context for entity/relationship representations and message-passing models. The implemented small network is original project code, not a copied research implementation.
2. PyTorch documentation, **Reproducibility**. https://docs.pytorch.org/docs/2.10/notes/randomness.html . Context for seeding, deterministic algorithms, and limits across platforms and releases.
3. Jupyter nbclient documentation, **Executing notebooks**. https://nbclient.readthedocs.io/en/latest/client.html . The release uses programmatic fresh-kernel execution and rejects cell errors.
4. Python documentation, **http.server — HTTP servers**. https://docs.python.org/3/library/http.server.html . The local studio is a development interface, not a recommended production internet server.
5. Official GitHub Actions repositories, **checkout** and **setup-python**. https://github.com/actions/checkout and https://github.com/actions/setup-python . Workflow templates use their documented v7 interfaces. The supplied workflow has not been run on a user's remote repository.

The implementation's exact equations and assumptions are fully specified in `MATH.md`. The results in `artifacts/evaluation/` are measurements generated locally by this repository, not numbers imported from these sources. Bibliographic context is not evidence of empirical novelty or superiority.

For a publication, perform a broader, current related-work review of epistemic game agents, model reconciliation, causal agent explanation, and search-guidance learning. Keep that literature review separate from the reproducible claims of this release. Earlier brainstorming is not a sufficient basis for a “first-ever” claim.

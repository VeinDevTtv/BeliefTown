# Contributing

Install the project from a source checkout with `python -m pip install -e ".[all]"`. Keep the default simulator usable without PyTorch or Jupyter; optional packages should be imported only by the features that need them.

When reporting an issue, include the scene JSON, command, Python and dependency versions, random seed, and a minimal replay or audit bundle. Leave out personal data, credentials, and private notebook sources.

For changes to the simulator or verifier, document the assumptions and finite candidate bounds they rely on. For learned-model changes, describe the validation selection, whole-scene split, non-neural baseline, end-to-end latency, and any unhelpful result. Keep the held-out evaluation split untouched during training.

Do not add model downloads, analytics, remote code execution, internet tunneling, or external assets without a clear need and documented redistribution terms. Keep examples small and nonviolent, and preserve accessible labels in the interface.

The project is maintained at [VeinDevTtv/BeliefTown](https://github.com/VeinDevTtv/BeliefTown). Use GitHub Issues for reproducible problems and pull requests for focused changes. Describe the user-facing effect and update the relevant documentation.

# Launch draft

Why did this NPC search the wrong room—and what is the smallest piece of information that would change its decision?

BeliefTown is an open-source counterfactual debugger for that question.

In the demo, three characters share a four-room world but receive different evidence. The inspector exposes their belief vectors, evidence origins, and expected search costs. Forwarding another copy of an old report does nothing. Delivering a genuinely different observation changes the Courier's route.

The debugger searches for a minimum legal information patch, verifies it by execution, and replays the result without changing the parcel's physical history or forcing an action.

The release includes a Python simulator, interactive local studio, four executed Jupyter notebooks, a graph-based search ranker, nine trained checkpoints, and a reproducible synthetic benchmark.

An important result: the graph model was slower than the heuristic on these small scenes. The heuristic remains the default. Neural scores can suggest an ordering; they cannot certify a patch.

This is a bounded research tool, not a claim of human-like intelligence. The interesting part is making information-dependent behavior inspectable, testable, and reproducible.

Attach the matched-replay demonstration and link your published repository and technical article when posting. The source archive includes the results, assumptions, failure cases, and MIT license.

#GameAI #MachineLearning #OpenSource

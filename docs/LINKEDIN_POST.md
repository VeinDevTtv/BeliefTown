I built an NPC that heads to the Workshop while the parcel is in the Library.

It has an old report. Mira has a newer observation it hasn't received.

BeliefTown lets me inspect that evidence and ask which permitted message would change the Courier's next decision. In this demo, forwarding one existing observation is enough. The debugger verifies the result and replays both routes in the same world.

The minimum is bounded: it applies to the enumerated deliveries and configured patch-size limit.

I used Bayesian filtering, evidence provenance, exact search, and an optional graph neural network to rank patches. The graph ranker was slower than the heuristic on the published small benchmark, so the heuristic remains the default.

You can try the recorded interactive showcase without installing Python:
https://www.aitbourich.com/belieftown/demo/

Write-up: https://www.aitbourich.com/blog/belieftown-npcs-can-be-wrong
Source and notebooks: https://github.com/VeinDevTtv/BeliefTown

I'd be interested in how game developers would use evidence inspection in their own debugging tools.

#GameAI #MachineLearning #GameDevelopment #OpenSource

# Why did the NPC go to the wrong room? Building a counterfactual debugger

A courier walks toward the Workshop. The parcel is in the Library. It is tempting to call the character stupid—or fix the behavior by giving it access to the parcel's true location.

BeliefTown asks a different question: **what information made that decision reasonable, and what is the smallest legitimate piece of information that would change it?**

The project is a small, open-source research debugger. Its demonstration has four rooms, three NPCs, and one parcel. Its core is a probability model, a cost-based planner, an evidence-provenance graph, a counterfactual replay system, and a verified information-patch search. A trained graph network is included, but it has a deliberately limited job: suggest which candidate patches the exact verifier should try first.

## A copied observation is not a second observation

Each observation has a permanent root identifier, original observer, and timestamp. Forwarding creates a new transmission but preserves that root.

Suppose four rooms are initially equally likely. An observer with an 80% correct-room sensor reports Library. The posterior probability of Library is 0.8. Forwarding that same observation twenty times leaves the posterior at 0.8. A genuinely independent matching observation raises it to approximately 0.9796 under the declared conditional-independence model.

That numerical example is easy to understand and surprisingly useful for debugging. An NPC can otherwise become certain because a single noisy claim circulated through several characters.

This system does not infer whether real people are trustworthy. Source reliability is a configured sensor property; provenance is explicitly supplied. Unknown shared errors between distinct roots remain outside the model.

## Old evidence belongs in the past

A message received on turn 9 might describe an observation from turn 2. Treating it as current evidence can be wrong when the parcel moved in between.

BeliefTown inserts the observation at its original timestamp and recomputes the finite-state filter forward through the transition matrices. Probabilities are evaluated in the log domain to avoid losing very small likelihood products. A separate brute-force latent-path calculation checks small cases.

The parcel becomes stationary when search begins. This restriction is intentional: it lets the project focus on information-dependent decisions rather than confounding them with a moving-target control problem.

## A decision that can be inspected

The NPC does not read hidden simulator truth. It receives its belief vector, current room, and permitted world geometry.

For each possible room-search order, the planner computes how much travel and inspection would be required if the parcel were in each room. Multiplying that cost vector by the belief vector gives expected search cost. With four rooms there are only 24 permutations, so no approximate controller is necessary.

The inspector shows the belief vector and ranked alternatives. Changing evidence changes the expected costs; the selected action changes only when that ranking changes. The explanation is a record of actual computations, not prose invented after an action.

## Information patches instead of forced actions

At a paused checkpoint, the designer requests a next destination. The solver enumerates deliveries of observations already held by existing characters, subject to available communication channels and protected-information constraints.

All deliveries in a patch are simultaneous. A sender cannot use the first message in the batch to acquire evidence and relay it through the second. The physical history is not rewritten.

The solver checks the empty patch, then one-delivery patches, then two-delivery patches, and so on. The first success is minimum-cardinality only after every smaller alternative has been ruled out. The interface distinguishes feasible, minimum-within-bounds, no-solution-within-bounds, and budget-exhausted outcomes.

A result can be exported as a portable audit containing the scene, checkpoint, candidate space, patch, and verification record. A separate verification path reconstructs it without trusting the model's score or the interface's displayed verdict.

## What the graph network actually learns

The evidence structure becomes a typed graph of rooms, actors, root observations, transmissions, and proposed deliveries. A small relational network scores whether a candidate patch will produce the requested next action. Execution supplies the labels.

The release includes a graph model, a non-graph MLP, and a provenance-removal ablation, each trained with three seeds. Entire scenes stay together across train/validation/test splits. Alternative patches from the same scene cannot leak into a different split.

Crucially, model predictions only reorder candidates. They do not prune certified search or decide whether a patch is valid. A bad prediction can waste computation; it cannot make an invalid patch pass the verifier.

## The result that changed the default

The released dataset contains 216 scenes and 13,001 labeled candidate patches. The measured evaluation includes 792 search executions over 48 test scenes and 24 larger-world distribution-shift scenes. All returned feasibility/minimality results agreed with the exhaustive labels on those cases.

That is not a 100% classifier-accuracy claim. It is evidence that keeping verification outside the neural model preserves the search contract on the evaluated cases.

The graph network **did not establish a wall-clock speed advantage over the heuristic**. At this scale, encoding and scoring candidate graphs cost more than the small differences in verifier calls saved. Many goals also have no solution within bounds, requiring exhaustive certification regardless of ranking.

The default is therefore the heuristic. The learned model, checkpoints, histories, raw results, uncertainty intervals, and failure interpretation remain in the release. A negative result is more informative than hiding the model's overhead behind a call-count chart.

## What this does not prove

The debugger is a finite, controlled model, not a human-mind simulation. It does not support unrestricted dialogue, strategic withholding, inferred hidden provenance, moving targets during search, or arbitrary commercial-game worlds. The candidate-space bound is part of every minimum claim.

The benchmark is synthetic and small. Its larger-world split changes several generation factors together. Source priors can be subjective. No human study measures believability or enjoyment. The local browser server is not a production service.

Those limits make the result easier to reproduce. The repository includes executable Jupyter notebooks, exact mathematical contracts, tests, scene generators, trained CPU checkpoints, an interactive local application, and a no-install recorded HTML showcase.

## The useful design principle

The important feature is not that a character can be wrong. It is that a designer can inspect why it is wrong, change only legitimate information, and reproduce the resulting behavior.

BeliefTown treats knowledge as a debuggable game mechanic rather than an invisible excuse for an NPC's decision.

*Publication note: run the included release checks in your own environment, add your own development context and repository link, and preserve the measured-result qualifications when adapting this draft.*

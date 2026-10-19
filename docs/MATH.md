# Mathematical specification

This specification follows the implementation in this release. It separates model assumptions, exact finite-model calculations, learned search ordering, and the limits of a verification claim. Bayesian filtering, route enumeration, graph networks, and subset search are established tools; this project does not present their integration as a new theorem.

## 1. State and information boundaries

The parcel location at logical turn \(t\) is \(X_t\in\{0,\ldots,L-1\}\). The simulator stores the actual trajectory, but a character does not receive it. The character's policy interface, `AgentView`, contains its current room, prior, known transition model, received evidence, room-connection graph, and inspection cost.

For actor \(i\), the belief is

\[
 b_t^i(x)=P(X_t=x\mid H_t^i),\qquad
 \mathbf b_t^i\in\Delta^{L-1},
\]

where \(H_t^i\) is the available evidence history. The probability simplex is

\[
\Delta^{L-1}=\{\mathbf b\in\mathbb R^L:b(x)\ge0,\;\sum_xb(x)=1\}.
\]

The designer's inspector is privileged: it can display the parcel's actual location and each character's private evidence. It is **not** an input to the character's policy. A hidden-state invariance check changes the unseen world while holding `AgentView` fixed and confirms that the decision is unchanged.

The world supports 2–7 rooms; the canonical demonstration uses four rooms and three actors. The parcel can move during setup, then remains stationary during the search phase. Actors may move and inspect rooms. A found parcel is not removed, so other actors can independently locate it; a successful actor then becomes idle.

## 2. Observations and provenance

An original observation \(r\) has an identifier, observer, observation time \(\tau_r\), report, and declared reliability. A forwarded message adds its own identifier, sender, recipient, delivery time, and parent transmission. Its root observation remains unchanged.

For an ordinary noisy room report \(Y_r=y\), with configured reliability \(q_r\),

\[
P(Y_r=y\mid X_{\tau_r}=x)=
\begin{cases}
q_r,&x=y,\\
(1-q_r)/(L-1),&x\ne y.
\end{cases}
\]

The implementation allows per-observation reliability: observations from the same actor need not have identical sensor quality. This is neither inferred honesty nor a model of human psychology.

For an exact room inspection, the likelihood is an indicator. An empty inspection of room \(j\) has \(\ell(x)=\mathbb{1}[x\ne j]\); finding the parcel has \(\ell(x)=\mathbb{1}[x=j]\). Likelihood vectors are checked for finite values in \([0,1]\). A declared scene cannot contain an observation with zero likelihood under its physical history and sensor model.

A character conditions on the **set of received root IDs**, not the count of transmissions:

\[
\mathcal R_t^i=\{r:\text{actor }i\text{ possesses root }r\text{ by }t\}.
\]

Different roots are assumed conditionally independent given the parcel trajectory. That assumption is supplied by the controlled simulator, **not inferred from uniqueness of strings**. Two reports originating in the same noisy measurement must share a root. Correlated sensors with distinct IDs would require a richer model.

Message payloads are not distorted. Schedules are exogenous; the decision to speak or remain silent is not additional evidence. Observations are scheduled sensor records independent of the NPC’s eventual route. In contrast, the character’s physical inspections are generated from its actual location and the parcel state.

### Worked duplicate-evidence calculation

Starting with a uniform four-room prior and one report with \(q=0.8\), the posterior probability of the reported room is \(0.8\). Receiving twenty copies of that same root at the same fixed snapshot still yields \(0.8\).

Two independent matching observations yield

\[
\frac{0.8^2}{0.8^2+3(0.2/3)^2}\approx0.979592.
\]

This is a consequence of the model, not a measured claim about real-world testimony.

## 3. Timestamp-aware filtering

Transitions use the **column-stochastic** convention

\[
 T_t[x',x]=P(X_{t+1}=x'\mid X_t=x),\qquad
 \sum_{x'}T_t[x',x]=1.
\]

An old report is evidence about \(X_{\tau_r}\), not about the state at its delivery time. At current time \(t\), group received roots by observation time:

\[
 \ell_{\tau,t}^i(x)=
 \prod_{r\in\mathcal R_t^i:\tau_r=\tau}
 P(Y_r\mid X_\tau=x).
\]

Then recompute chronologically:

\[
\alpha_0=\operatorname{Normalize}(b_{prior}\odot\ell_{0,t}^i),
\]

\[
\alpha_\tau=\operatorname{Normalize}
\left(\ell_{\tau,t}^i\odot T_{\tau-1}\alpha_{\tau-1}\right),
\qquad b_t^i=\alpha_t.
\]

Inserting late evidence at its original timestamp and replaying this forward calculation conditions on all evidence currently received, including observations delivered out of order. It does not retroactively give an actor that report in earlier episode snapshots.

### Log-domain implementation

Let \(z(x)=\log b(x)\). Prediction uses

\[
 z_{pred}(x')=\operatorname{LSE}_x\left(\log T[x',x]+z(x)\right),
\]

where \(\operatorname{LSE}\) is log-sum-exp. Update adds log likelihoods and subtracts a log normalization constant. Impossible transitions and likelihoods are represented by \(-\infty\).

This avoids multiplying many tiny probabilities or permanently rounding a possible state to zero during a sequence of conflicting observations. The returned probability vector is float64; extremely small output coordinates can still be below floating-point representability, but internal recomputation retains log support. Truly inconsistent evidence raises `InconsistentEvidence`; it is not silently ignored or replaced with a uniform belief.

The straightforward runtime cost is \(O(tL^2+|\mathcal R_t^i|L)\), which is small for the released scenes. An independent reference implementation checks small cases by enumerating all \(L^{t+1}\) latent trajectories.

## 4. Search decisions as linear objectives

Rooms form a connected undirected graph with positive edge costs. Shortest-path distances \(d(u,v)\) are computed by Floyd–Warshall. Let \(c_I>0\) be the inspection cost and \(s\) the actor’s current room.

For a permutation \(\sigma=(\sigma_1,\ldots,\sigma_L)\), define \(\sigma_0=s\). If the parcel is in room \(x\), with position \(k_\sigma(x)\) in the order, the cost before finding it is

\[
 c_\sigma(x)=\sum_{j=1}^{k_\sigma(x)}
 \bigl[d(\sigma_{j-1},\sigma_j)+c_I\bigr].
\]

Its expectation is

\[
 J(\sigma\mid\mathbf b)=\sum_x b(x)c_\sigma(x)
 =\mathbf c_\sigma^\top\mathbf b.
\]

Stacking cost vectors produces \(\mathbf C\in\mathbb R^{L!\times L}\); all route values are \(\mathbf C\mathbf b\). The lowest expected cost wins. Equal computed costs use deterministic lexicographic ordering. Zero-probability rooms are not deliberately selected for inspection.

This is exact among the modeled fixed search orders, assuming a stationary parcel and no additional information during that order. The actual controller replans after new messages, movement, or inspection results. It is not a globally optimal multi-agent POMDP solution.

A logical turn is one graph-edge movement or one inspection; an edge’s numerical cost is a design cost, not its number of elapsed turns. Thus the system does not claim to simulate continuous time or variable-duration inference.

Belief uncertainty is displayed as

\[
H(\mathbf b)=-\sum_{x:b(x)>0}b(x)\log b(x).
\]

Entropy is measured in nats. Low entropy is not proof of correctness.

## 5. Legal information interventions

At a paused search decision, \(\mathcal U_t\) is the disclosed finite set of available deliveries. A delivery specifies `(sender, recipient, root_id)`. The sender must already possess that root, a permitted directed communication channel must exist, and provenance must reach an original observation.

For the immediate single-target objective, candidate generation includes deliveries to the target actor. Sending to unrelated actors cannot change this immediate action under the declared simultaneous-batch semantics. Multiple possible senders of a root remain different designer operations. Redundant known roots remain in the universe as negative controls.

Candidates are sorted canonically and capped by `candidate_limit` (24 by default). This cap is disclosed in the audit. Renaming identifiers can alter a **truncated** candidate universe, although names are not neural input features. Comparisons always use the same candidate universe.

A patch is a subset \(\Delta\subseteq\mathcal U_t\). All legality checks use the **original** snapshot, so a sender cannot forward evidence it only receives inside that same batch. The live application allows one simultaneous patch batch per logical tick.

Given target actor \(i\) and desired next search room \(a^*\), solve

\[
\Delta^*=\arg\min_{\Delta\subseteq\mathcal U_t}|\Delta|
\]

subject to

\[
\pi_i(\Phi_t(\Delta))=a^*,\quad
\operatorname{Legal}(\Delta),\quad
\operatorname{Constraints}(\Delta),\quad |\Delta|\le K.
\]

Here \(\Phi_t\) applies deliveries, updates beliefs, and recomputes decisions. Constraints in version one protect specified actors from new deliveries. The target actor must be active and not have completed its search. A hypothetical route for an idle actor is not accepted as an actual pending action.

No world fact or action is overwritten. The requested destination is a designer objective; it need not be the parcel’s true room.

### Canonical scene

Before the patch, the Courier’s belief in room order `(Workshop, Library, Garden, Hall)` is approximately

\[
[0.605773,\;0.197113,\;0.197113,\;0].
\]

After receiving Mira’s existing `new-library` root it is approximately

\[
[0.117298,\;0.813179,\;0.069523,\;0].
\]

The planner changes its next destination from Workshop to Library. No zero-delivery patch does so; one delivery therefore is minimum within the configured candidate set.

## 6. Enumeration, certificates, and audit recomputation

Search patch cardinalities in increasing order: zero, one, two, and so on. For \(m\) deliveries and bound \(K\), the total universe is

\[
N=\sum_{k=0}^{\min(K,m)}\binom{m}{k}.
\]

For \(m=24,K=3\), \(N=2325\). The release rejects searches or audit rechecks above 250,000 subsets. Larger worlds may also become expensive because route enumeration is factorial.

A successful patch is minimum only when every smaller permitted patch has been ruled out. An inclusion-minimal patch—one whose individual elements cannot be removed—is not necessarily minimum-cardinality.

The four result statuses are `verified_minimum_within_bounds`, `verified_feasible`, `no_solution_within_bounds`, and `budget_exhausted`. Optional feasibility-first search can find a working patch without a minimum certificate. A verifier-call budget does not become a wall-clock deadline claim.

An audit bundle contains the scene, replay controls, checkpoint and candidate hashes, problem definition, patch, before/after beliefs, verification result, bounded-search metadata, and software versions. `verify_bundle` regenerates the episode and executes the claim. It can enumerate every cheaper alternative independently of neural ordering. It also rejects changed candidate universes and fabricated verification fields.

The auditor shares the simulator’s semantics; it is not a separately proved theorem prover. It checks logical execution claims, not whether an asserted timing measurement was honestly recorded. Tests and independent small references reduce implementation risk but do not establish universal bug-freedom.

## 7. Learned graph ordering

The graph contains room, actor, observation, message, and proposed-delivery nodes. The implementation uses a 20-dimensional node feature vector (including type indicators and reserved coordinates), 21 directed relation types, and 14 graph-level features. Features include beliefs, distances, timestamps, source likelihoods, target markers, selected deliveries, and a non-neural likelihood heuristic. Hidden parcel truth is excluded.

With hidden width 32 and two layers,

\[
h_v^{(\ell+1)}=\operatorname{ReLU}\left[
W_{self}^{(\ell)}h_v^{(\ell)}+
\sum_r\frac{1}{\max(1,|N_r(v)|)}
\sum_{u\in N_r(v)}W_r^{(\ell)}h_u^{(\ell)}
\right].
\]

Implementation row-vector conventions are algebraically equivalent. Padding is masked. The mean graph embedding, target actor embedding, destination embedding, and global features are concatenated into an MLP, producing logit \(s_\theta\).

For scene snapshot \(S\), target actor \(i\), desired room \(a^*\), and candidate patch \(\Delta\), the binary label is

\[
y(S,i,a^*,\Delta)=\mathbb{1}[\text{exact execution satisfies the requested action and constraints}].
\]

All successful alternatives are positive labels, not just one selected solution. If the model outputs logit \(s_\theta(S,i,a^*,\Delta)\), its sigmoid is \(\hat p=\sigma(s_\theta)\). For \(N\) labeled patches, training minimizes binary cross-entropy:

\[
\mathcal L(\theta)=-\frac{1}{N}\sum_{n=1}^{N}\left[y_n\log\hat p_n+(1-y_n)\log(1-\hat p_n)\right].
\]

The implementation evaluates this loss directly from logits for numerical stability. AdamW uses learning rate \(0.002\), weight decay \(10^{-4}\), batch size 64, and gradient-norm clipping at 5. Each release run trains for 14 epochs and selects the checkpoint with the lowest validation BCE. The sigmoid output is used to train the ranker; no probability-calibration claim is made.

The graph model has 52,961 parameters. A 5,633-parameter MLP receives global and pooled node features without edges. The `graph_no_provenance` ablation removes root-to-message and forwarding relations, direct known-root indicators, and related scalar shortcut features. It is a bundled ablation, not an isolated causal estimate of a single graph edge type.

Within each cardinality, candidate patches are sorted by learned score. **None are discarded.** Incorrect model predictions can hurt efficiency, but they cannot authorize an invalid patch. Whole-scene train/validation/test splits prevent variants of one episode leaking across partitions.

## 8. Counterfactual replay

Diagnostic interventions suppress an original observation or a transmission and rebuild the episode. Dependent messages are regenerated from available information. Suppressing one transmission does not erase a root that reached a sender through another valid path. Removing a root removes its downstream availability.

World generation uses keyed random streams: a hash of seed and semantic event key initializes a local generator. Forks preserve recorded exogenous physical events and measurements while recomputing endogenous search actions and inspections. They do not depend on a shared random stream’s branch-dependent draw count.

These counterfactuals are defined by the finite simulator. They do not explain human cognition or reveal the hidden reasoning of an unrelated model.

## 9. Evaluation

For case \(n\), let \(b_n(x)\) be the belief assigned to room \(x\), and let \(X_n\) be the true room. The multiclass Brier score is

\[
\mathrm{BS}=\frac{1}{M}\sum_{n=1}^{M}\sum_x
\left(b_n(x)-\mathbb{1}[X_n=x]\right)^2.
\]

For a feasible patch with reference optimum, report \(|\Delta|-|\Delta^*|\). Report verifier calls and full search latency separately. Neural graph construction and inference count toward latency. Model loading and scene construction are amortized setup costs and excluded from steady-state timing.

Average a learned method’s three seed outcomes within a scene before bootstrapping scene means. Paired comparisons use identical scenes. Do not treat candidate patches as independent observations. Source histories for belief baselines are held fixed at the initial search checkpoint.

The released small benchmark does not show a neural runtime advantage. No-solution certification must still exhaust the bounded universe; no ordering model can avoid that obligation under this algorithm. See the measured report rather than inferring a speedup from a visually appealing demonstration.

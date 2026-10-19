"""Legal information patches, exact verification, and completeness-aware search."""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
import itertools
import json
import math
import platform
import sys
import time
from typing import Callable
import numpy as np
from .world import Snapshot, Receipt
from .schema import Message, digest


@dataclass(frozen=True,order=True)
class Delivery:
    sender: str
    recipient: str
    root_id: str

    @property
    def id(self) -> str:
        return digest(asdict(self))[:16]


@dataclass(frozen=True)
class PatchProblem:
    target_actor: str
    target_room: int
    max_size: int = 3
    protected_actors: tuple[str,...] = ()
    candidate_limit: int = 24
    def validate(self,snapshot: Snapshot):
        if snapshot.time < snapshot.scene.search_start:
            raise ValueError("Patch objectives are defined at search-phase decision snapshots")
        if not isinstance(self.target_room,int) or isinstance(self.target_room,bool) or not isinstance(self.max_size,int) or isinstance(self.max_size,bool) or not isinstance(self.candidate_limit,int) or isinstance(self.candidate_limit,bool):
            raise ValueError("Room, patch-size bound, and candidate limit must be integers")
        if self.target_actor not in snapshot.positions or not 0<=self.target_room<len(snapshot.scene.rooms):
            raise ValueError("Unknown target actor or destination")
        actor=next(a for a in snapshot.scene.actors if a.id==self.target_actor)
        if not actor.active or snapshot.found[self.target_actor]:
            raise ValueError("Target actor has no pending search action: inactive or parcel already found")
        if not 0<=self.max_size<=5 or not 0<=self.candidate_limit<=64:
            raise ValueError("Use max_size 0–5 and candidate_limit 0–64")
        if not set(self.protected_actors)<=set(snapshot.positions): raise ValueError("Unknown protected actor")


def legal_delivery(snapshot: Snapshot, d: Delivery) -> tuple[bool,str]:
    if (d.sender,d.recipient) not in snapshot.scene.channels:
        return False,"No permitted communication channel"
    if d.root_id not in snapshot.roots or d.root_id not in snapshot.known(d.sender):
        return False,"Sender does not possess this observation at the original snapshot"
    if snapshot.roots[d.root_id].time>snapshot.time:
        return False,"Future evidence is not available"
    try: snapshot.parent_for(d.sender,d.root_id)
    except ValueError as e: return False,str(e)
    return True,"legal"


def apply_patch(snapshot: Snapshot, patch: list[Delivery] | tuple[Delivery,...]) -> Snapshot:
    """All legality checks use the ORIGINAL snapshot, prohibiting intra-batch relays."""
    if len(set(patch))!=len(patch): raise ValueError("A patch cannot duplicate an identical delivery")
    for d in patch:
        ok,reason=legal_delivery(snapshot,d)
        if not ok: raise ValueError(reason)
    result=snapshot.clone()
    for d in sorted(patch):
        parent=snapshot.parent_for(d.sender,d.root_id)
        prefix=f"patch:{snapshot.time}:{d.id}"
        mid=prefix; suffix=1
        while mid in result.messages:
            mid=f"{prefix}:{suffix}"; suffix+=1
        result.messages[mid]=Message(mid,d.root_id,d.sender,d.recipient,snapshot.time,parent)
        result.receipts[d.recipient].append(Receipt(d.root_id,snapshot.time,mid))
        result.events.append({"time":snapshot.time,"type":"patch_delivery","id":mid,**asdict(d)})
    result._belief_cache.clear()
    return result


def candidate_deliveries(snapshot: Snapshot, problem: PatchProblem) -> tuple[Delivery,...]:
    """Canonical bounded candidate universe, independent of the ranker.

    Keep legal duplicates of known roots as negative controls. Keep multiple
    senders of a root as separate designer operations. Immediate target-action
    objectives need not send messages to unrelated agents, so those are excluded.
    The same finite universe is disclosed to every method and in every audit.
    """
    problem.validate(snapshot)
    candidates=[]
    for sender,recipient in sorted(snapshot.scene.channels):
        if recipient != problem.target_actor: continue
        for root in sorted(snapshot.known(sender)):
            d=Delivery(sender,recipient,root)
            if legal_delivery(snapshot,d)[0]: candidates.append(d)
    return tuple(sorted(candidates))[:problem.candidate_limit]


@dataclass
class Verification:
    feasible: bool
    legal: bool
    constraints_satisfied: bool
    action: int | None
    before_belief: list[float]
    after_belief: list[float] | None
    reason: str


def verify_patch(snapshot: Snapshot, problem: PatchProblem, patch) -> Verification:
    problem.validate(snapshot)
    before=snapshot.belief(problem.target_actor).tolist()
    if len(patch)>problem.max_size:
        return Verification(False,False,False,None,before,None,"Patch exceeds cardinality bound")
    try: after=apply_patch(snapshot,tuple(patch))
    except (ValueError,KeyError) as e:
        return Verification(False,False,False,None,before,None,str(e))
    constraints=not any(d.recipient in problem.protected_actors for d in patch)
    action=after.plan(problem.target_actor).destination
    feasible=constraints and action==problem.target_room
    return Verification(feasible,True,constraints,action,before,
                        after.belief(problem.target_actor).tolist(),
                        "verified" if feasible else ("Information constraint violated" if not constraints else "Target action not reached"))


def heuristic_scores(snapshot: Snapshot, problem: PatchProblem, patches: list[tuple[Delivery,...]]) -> list[float]:
    """Cheap likelihood/age heuristic; no counterfactual verifier calls."""
    target=problem.target_room
    scores=[]
    known=snapshot.known(problem.target_actor)
    for patch in patches:
        seen=set(known); value=0.
        for d in patch:
            if d.recipient in problem.protected_actors:
                value-=100.; continue
            if d.root_id in seen: continue
            seen.add(d.root_id)
            root=snapshot.roots[d.root_id]
            p=root.probabilities(len(snapshot.scene.rooms))
            other=(float(p.sum())-float(p[target]))/(len(p)-1)
            value+=math.log((float(p[target])+1e-9)/(other+1e-9))/(1.+snapshot.time-root.time)
        scores.append(value)
    return scores


@dataclass
class SearchResult:
    status: str
    patch: tuple[Delivery,...] | None
    verification: Verification | None
    verifier_calls: int
    candidates: tuple[Delivery,...]
    checked_per_size: dict[int,int]
    all_smaller_checked: bool
    elapsed_seconds: float
    ranking_seconds: float
    method: str
    problem: PatchProblem
    checkpoint_hash: str
    scene_hash: str
    candidate_space_size: int
    budget: int | None
    witness_trace: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d=asdict(self)
        d["schema_version"]=1
        d["scope"]="Minimum delivery count within the recorded finite candidate universe and max_size"
        d["software"]={"belieftown":"1.0.0","python":sys.version.split()[0],"numpy":np.__version__,"platform":platform.platform()}
        d["candidate_hash"]=digest([asdict(x) for x in self.candidates])
        return json.loads(json.dumps(d, allow_nan=False))


def search(snapshot: Snapshot, problem: PatchProblem, *, method: str="uninformed",
           ranker=None, budget: int | None=None, feasible_first: bool=False,
           trace: bool=True) -> SearchResult:
    """Cardinality-first search retains exact minimum guarantees.

    In feasible_first mode all cardinalities are ranked together: returning a
    feasible patch is then deliberately NOT a minimum certificate. Neural scores
    only change ordering, never candidate membership or verifier logic.
    """
    problem.validate(snapshot)
    if budget is not None and (not isinstance(budget,int) or isinstance(budget,bool) or budget<0):
        raise ValueError("Verifier budget must be a nonnegative integer")
    if method not in ("uninformed","heuristic","graph","mlp","graph_no_provenance"):
        raise ValueError("Unknown search method")
    if method not in ("uninformed","heuristic") and ranker is None: raise ValueError("A learned method requires a loaded ranker")
    started=time.perf_counter(); rank_time=0.
    candidates=candidate_deliveries(snapshot,problem)
    max_k=min(problem.max_size,len(candidates))
    space=sum(math.comb(len(candidates),k) for k in range(max_k+1))
    if space>250_000: raise ValueError("Candidate universe exceeds 250,000 subsets; lower bounds")
    checked={k:0 for k in range(max_k+1)}; calls=0; witness=[]
    result_patch=None; result_verification=None; status="no_solution_within_bounds"; smaller=True
    layers=[list(itertools.combinations(candidates,k)) for k in range(max_k+1)]
    if feasible_first: layers=[sum(layers,[])]
    for layer in layers:
        t0=time.perf_counter()
        if method=="uninformed": order=list(range(len(layer)))
        else:
            scores=heuristic_scores(snapshot,problem,layer) if method=="heuristic" else ranker.score(snapshot,problem,layer)
            if len(scores)!=len(layer) or not np.isfinite(scores).all(): raise ValueError("Ranker returned invalid scores")
            order=sorted(range(len(layer)),key=lambda j:(-float(scores[j]),j))
        rank_time+=time.perf_counter()-t0
        for j in order:
            if budget is not None and calls>=budget:
                status="budget_exhausted"; smaller=False; break
            patch=layer[j]
            verdict=verify_patch(snapshot,problem,patch)
            calls+=1; checked[len(patch)]+=1
            if trace: witness.append({"patch":[d.id for d in patch],"feasible":verdict.feasible})
            if verdict.feasible:
                result_patch=patch; result_verification=verdict
                smaller=all(checked[k]==math.comb(len(candidates),k) for k in range(len(patch)))
                status="verified_minimum_within_bounds" if smaller else "verified_feasible"
                break
        if result_patch is not None or status=="budget_exhausted": break
    return SearchResult(status,result_patch,result_verification,calls,candidates,checked,smaller,
                        time.perf_counter()-started,rank_time,method,problem,snapshot.checkpoint_hash,
                        snapshot.scene.hash,space,budget,witness)

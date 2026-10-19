"""Independent audit checker: recompute, do not trust stored neural predictions."""
from __future__ import annotations
from dataclasses import asdict
import itertools
import math
from .schema import digest
from .interventions import Delivery, PatchProblem, candidate_deliveries, verify_patch


def check_audit(snapshot, record: dict, *, certify_minimum: bool=True) -> dict:
    allowed={"verified_minimum_within_bounds","verified_feasible","no_solution_within_bounds","budget_exhausted"}
    if record.get("status") not in allowed:raise ValueError("Unknown audit status")
    if record.get("schema_version")!=1: raise ValueError("Unsupported audit schema")
    if record.get("checkpoint_hash")!=snapshot.checkpoint_hash or record.get("scene_hash")!=snapshot.scene.hash:
        raise ValueError("Audit does not belong to this exact checkpoint")
    p=dict(record["problem"]); p["protected_actors"]=tuple(p.get("protected_actors",()))
    problem=PatchProblem(**p)
    universe=candidate_deliveries(snapshot,problem)
    if sum(math.comb(len(universe),k) for k in range(min(len(universe),problem.max_size)+1))>250_000:
        raise ValueError("Audit universe exceeds the release verification bound")
    expected=[asdict(d) for d in universe]
    if record.get("candidates")!=expected or record.get("candidate_hash")!=digest(expected):
        raise ValueError("Audit candidate universe was modified")
    space=sum(math.comb(len(universe),k) for k in range(min(len(universe),problem.max_size)+1))
    if record.get("candidate_space_size")!=space:raise ValueError("Incorrect candidate-space size")
    raw=record.get("patch")
    if raw is None:
        if record.get("status")=="no_solution_within_bounds" and certify_minimum:
            calls=0
            for k in range(min(len(universe),problem.max_size)+1):
                for patch in itertools.combinations(universe,k):
                    calls+=1
                    if verify_patch(snapshot,problem,patch).feasible:
                        raise ValueError("False no-solution certificate")
            return {"valid":True,"status":"no_solution_within_bounds","recomputed_calls":calls}
        if record.get("status")!="budget_exhausted": raise ValueError("Missing patch for this audit status")
        return {"valid":True,"status":"budget_exhausted","claim":"No feasibility or minimum claim made"}
    if record.get("status") not in {"verified_minimum_within_bounds","verified_feasible"}:
        raise ValueError("An unresolved status cannot carry a certified patch")
    patch=tuple(Delivery(**d) for d in raw)
    if any(d not in universe for d in patch): raise ValueError("Patch is outside the recorded universe")
    v=verify_patch(snapshot,problem,patch)
    if not v.feasible: raise ValueError("Patch fails execution verification")
    stored=record.get("verification")
    if stored != asdict(v):raise ValueError("Stored verification does not match recomputed beliefs/action/constraints")
    calls=1; minimum=False
    if certify_minimum:
        for k in range(len(patch)):
            for smaller in itertools.combinations(universe,k):
                calls+=1
                if verify_patch(snapshot,problem,smaller).feasible:
                    if record["status"]=="verified_minimum_within_bounds":
                        raise ValueError("False minimum certificate")
                    return {"valid":True,"status":"verified_feasible","minimum":False,"recomputed_calls":calls}
        minimum=True
    return {"valid":True,"status":"verified_minimum_within_bounds" if minimum else "verified_feasible",
            "minimum":minimum,"recomputed_calls":calls,"action":v.action}

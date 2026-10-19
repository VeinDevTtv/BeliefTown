"""Deterministic event-sourced simulation with isolated agent observations."""
from __future__ import annotations
from dataclasses import dataclass, field, replace, asdict
import copy
from typing import Any
import numpy as np
from .schema import Scene, Observation, Message, digest
from .beliefs import filter_belief, last_report_belief, entropy
from .planner import plan_search, distances


@dataclass(frozen=True)
class Receipt:
    root_id: str
    received_at: int
    message_id: str | None


@dataclass(frozen=True)
class AgentView:
    """Only this view is passed into agent inference / planning. No truth field."""
    actor_id: str
    time: int
    room: int
    prior: tuple[float,...]
    transitions: tuple
    observations: tuple[Observation,...]
    received_observations: tuple[Observation,...]
    edges: tuple
    inspection_cost: float

    def belief(self, mode: str = "provenance") -> np.ndarray:
        if mode == "provenance":
            return filter_belief(self.prior,self.transitions,self.observations,self.time)
        if mode == "naive":
            return filter_belief(self.prior,self.transitions,self.received_observations,self.time,deduplicate=False)
        if mode == "last_report":
            return last_report_belief(self.prior,self.transitions,list(self.received_observations),self.time)
        raise ValueError(f"Unknown belief mode: {mode}")

    def plan(self, mode: str = "provenance"):
        return plan_search(self.belief(mode),self.room,self.edges,self.inspection_cost)


@dataclass
class Snapshot:
    scene: Scene
    time: int
    positions: dict[str,int]
    roots: dict[str,Observation]
    messages: dict[str,Message]
    receipts: dict[str,list[Receipt]]
    paths: dict[str,list[int]]
    costs: dict[str,float]
    found: dict[str,bool]
    events: list[dict]
    decisions: dict[str,dict] = field(default_factory=dict)
    _belief_cache: dict = field(default_factory=dict, repr=False)

    def known(self, actor: str) -> set[str]:
        if actor not in self.receipts: raise KeyError(f"Unknown actor: {actor}")
        return {r.root_id for r in self.receipts[actor]}

    def view(self, actor: str) -> AgentView:
        info = next((a for a in self.scene.actors if a.id == actor),None)
        if info is None: raise KeyError(actor)
        roots = tuple(self.roots[r] for r in sorted(self.known(actor)))
        received = tuple(self.roots[r.root_id] for r in self.receipts[actor])
        return AgentView(actor,self.time,self.positions[actor],info.prior,self.scene.transitions,
                         roots,received,self.scene.edges,self.scene.inspection_cost)

    def belief(self, actor: str, mode: str = "provenance") -> np.ndarray:
        key = (actor,mode)
        if key not in self._belief_cache:
            value = self.view(actor).belief(mode)
            value.setflags(write=False)
            self._belief_cache[key] = value
        return self._belief_cache[key]

    def plan(self, actor: str, mode: str = "provenance"):
        return plan_search(self.belief(actor,mode),self.positions[actor],self.scene.edges,self.scene.inspection_cost)

    def clone(self) -> Snapshot:
        # Keep the immutable Scene shared; copy mutable episode records.
        return Snapshot(self.scene,self.time,dict(self.positions),dict(self.roots),dict(self.messages),
                        {k:list(v) for k,v in self.receipts.items()},
                        {k:list(v) for k,v in self.paths.items()},dict(self.costs),dict(self.found),
                        copy.deepcopy(self.events),copy.deepcopy(self.decisions))

    def parent_for(self, sender: str, root_id: str) -> str | None:
        if root_id not in self.known(sender): raise ValueError("Sender does not possess the evidence")
        root = self.roots[root_id]
        if root.observer == sender: return None
        incoming = [m for m in self.messages.values() if m.recipient==sender and m.root_id==root_id]
        if not incoming: raise ValueError("Missing provenance path")
        return min(incoming,key=lambda m:(m.time,m.id)).id

    @property
    def checkpoint_hash(self) -> str:
        return digest({"scene":self.scene.hash,"time":self.time,"positions":self.positions,
                       "roots":{k:asdict(v) for k,v in self.roots.items()},
                       "messages":{k:asdict(v) for k,v in self.messages.items()},
                       "receipts":{k:[asdict(x) for x in v] for k,v in self.receipts.items()}})

    def inspector(self, mode: str = "provenance") -> dict:
        actors = []
        for a in self.scene.actors:
            b = self.belief(a.id,mode); p = self.plan(a.id,mode)
            actors.append({"id":a.id,"name":a.name,"room":self.positions[a.id],
                "belief":b.tolist(),"entropy":entropy(b),"destination":p.destination,
                "order":list(p.order),"expected_cost":p.expected_cost,
                "alternatives":[{"order":list(o),"cost":v} for o,v in p.alternatives],
                "known":sorted(self.known(a.id)),"receipts":[asdict(r) for r in self.receipts[a.id]],
                "path":self.paths[a.id],"travel_cost":self.costs[a.id],"found":self.found[a.id]})
        return {"time":self.time,"scene_id":self.scene.id,"checkpoint_hash":self.checkpoint_hash,
                "rooms":list(self.scene.rooms),"edges":list(self.scene.edges),"actors":actors,
                "truth":self.scene.truth[self.time],"roots":[asdict(o) for o in self.roots.values()],
                "messages":[asdict(m) for m in self.messages.values()],"events":self.events,
                "search_start":self.scene.search_start,"horizon":self.scene.horizon,"mode":mode}


class Episode:
    """Rebuild the complete event history after every intervention.

    Message schedules are exogenous. On a suppressed path a scheduled forwarding
    event is regenerated using another already-available path, if one exists.
    Physical search actions and inspection observations are always regenerated.
    """
    def __init__(self, scene: Scene, *, suppress_observations=(), suppress_messages=(),
                 patches: dict[int,list] | None = None, mode: str = "provenance"):
        self.scene=scene
        self.suppress_observations=frozenset(suppress_observations)
        self.suppress_messages=frozenset(suppress_messages)
        root_ids={o.id for o in scene.observations}; message_ids={m.id for m in scene.messages}
        if not self.suppress_observations <= root_ids:
            raise ValueError("Diagnostic suppression requires a declared observation ID")
        if not self.suppress_messages <= message_ids:
            raise ValueError("Diagnostic suppression requires a declared message ID")
        if mode not in ("provenance","naive","last_report"): raise ValueError("Unknown belief mode")
        self.patches=patches or {}; self.mode=mode
        if any(not 0<=t<=scene.horizon for t in self.patches): raise ValueError("Patch time outside episode")
        self.snapshots: list[Snapshot]=[]
        self._run()

    def _run(self):
        scene=self.scene
        s=Snapshot(scene,0,{a.id:a.room for a in scene.actors},{},{},
                   {a.id:[] for a in scene.actors},{a.id:[a.room] for a in scene.actors},
                   {a.id:0. for a in scene.actors},{a.id:False for a in scene.actors},[])
        messages={m.id:m for m in scene.messages}
        depth_cache={}
        def depth(m):
            if m.id not in depth_cache:
                depth_cache[m.id]=0 if m.parent_id is None else 1+depth(messages[m.parent_id])
            return depth_cache[m.id]
        schedule=sorted(scene.messages,key=lambda m:(m.time,depth(m),m.id))
        _,next_hop=distances(len(scene.rooms),scene.edges)
        weights={frozenset((u,v)):w for u,v,w in scene.edges}
        for t in range(scene.horizon+1):
            s.time=t; s._belief_cache.clear(); s.decisions={}
            for o in sorted(scene.observations,key=lambda o:o.id):
                if o.time==t:
                    if o.id in self.suppress_observations:
                        s.events.append({"time":t,"type":"suppressed_observation","id":o.id}); continue
                    s.roots[o.id]=o; s.receipts[o.observer].append(Receipt(o.id,t,None))
                    s.events.append({"time":t,"type":"observation","id":o.id,"actor":o.observer})
            for m in schedule:
                if m.time!=t: continue
                if m.id in self.suppress_messages:
                    s.events.append({"time":t,"type":"suppressed_message","id":m.id}); continue
                if m.root_id not in s.known(m.sender):
                    s.events.append({"time":t,"type":"unavailable_message","id":m.id}); continue
                # Preserve original parent when available, otherwise reconstruct
                # from a genuine alternate path, never from hidden simulator data.
                parent=m.parent_id
                if parent is not None and parent not in s.messages:
                    parent=s.parent_for(m.sender,m.root_id)
                actual=replace(m,parent_id=parent)
                s.messages[m.id]=actual
                s.receipts[m.recipient].append(Receipt(m.root_id,t,m.id))
                s.events.append({"time":t,"type":"message","id":m.id,"sender":m.sender,"recipient":m.recipient})
            s._belief_cache.clear()
            if t in self.patches:
                from .interventions import apply_patch
                s=apply_patch(s,self.patches[t])
            # Store the state before each decision. At tick t a movement or
            # inspection becomes visible at tick t+1 (not retroactively at t).
            actions={}
            if t>=scene.search_start:
                for a in scene.actors:
                    p=s.plan(a.id,self.mode)
                    kind="idle" if not a.active or s.found[a.id] else ("inspect" if p.destination==s.positions[a.id] else "move")
                    actions[a.id]={"kind":kind,"destination":p.destination,"order":list(p.order),"expected_cost":p.expected_cost}
            s.decisions=actions
            self.snapshots.append(s.clone())
            if t==scene.horizon: break
            for a in scene.actors:
                action=actions.get(a.id)
                if action is None or action["kind"]=="idle": continue
                current=s.positions[a.id]
                if action["kind"]=="move":
                    nxt=int(next_hop[current,action["destination"]])
                    s.positions[a.id]=nxt; s.paths[a.id].append(nxt)
                    s.costs[a.id]+=weights[frozenset((current,nxt))]
                    s.events.append({"time":t+1,"type":"move","actor":a.id,"from":current,"to":nxt})
                else:
                    found=current==scene.truth[t]
                    likelihood=tuple(float((x==current) if found else (x!=current)) for x in range(len(scene.rooms)))
                    oid=f"inspection:{a.id}:{t+1}"
                    o=Observation(oid,a.id,t+1,current,1.,"inspection_positive" if found else "inspection_negative",likelihood)
                    s.roots[oid]=o; s.receipts[a.id].append(Receipt(oid,t+1,None))
                    s.found[a.id]=found; s.costs[a.id]+=scene.inspection_cost
                    s.events.append({"time":t+1,"type":"inspection","actor":a.id,"room":current,"found":found,"id":oid})

    def at(self,time: int) -> Snapshot:
        if not 0<=time<len(self.snapshots): raise ValueError("Snapshot time outside episode")
        return self.snapshots[time]

    def to_dict(self) -> dict:
        return {"scene":self.scene.to_dict(),"mode":self.mode,
                "suppress_observations":sorted(self.suppress_observations),
                "suppress_messages":sorted(self.suppress_messages),
                "frames":[s.inspector(self.mode) for s in self.snapshots]}

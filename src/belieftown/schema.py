"""Validated, JSON-serializable scene format. No executable scene payloads."""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
import hashlib
import json
import math
from typing import Any
import numpy as np


class SceneError(ValueError):
    """A scene violates the declared finite-world model."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


@dataclass(frozen=True)
class Actor:
    id: str
    name: str
    room: int
    prior: tuple[float, ...]
    active: bool = True


@dataclass(frozen=True)
class Observation:
    id: str
    observer: str
    time: int
    reported_room: int
    reliability: float = 0.8
    kind: str = "sighting"
    # A custom likelihood is used for exact positive/negative room inspection.
    likelihood: tuple[float, ...] | None = None

    def probabilities(self, n_rooms: int) -> np.ndarray:
        if self.likelihood is not None:
            return np.asarray(self.likelihood, dtype=np.float64)
        result = np.full(n_rooms, (1.0-self.reliability)/(n_rooms-1))
        result[self.reported_room] = self.reliability
        return result


@dataclass(frozen=True)
class Message:
    id: str
    root_id: str
    sender: str
    recipient: str
    time: int
    parent_id: str | None = None


@dataclass(frozen=True)
class Scene:
    id: str
    seed: int
    rooms: tuple[str, ...]
    edges: tuple[tuple[int, int, float], ...]
    actors: tuple[Actor, ...]
    observations: tuple[Observation, ...]
    messages: tuple[Message, ...]
    # transitions[t][next_room][previous_room]; length == horizon
    transitions: tuple[tuple[tuple[float, ...], ...], ...]
    truth: tuple[int, ...]
    search_start: int
    horizon: int
    channels: tuple[tuple[str, str], ...]
    inspection_cost: float = 1.0
    family: str = "authored"
    description: str = ""
    schema_version: int = 1

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        n = len(self.rooms)
        is_int=lambda x:isinstance(x,(int,np.integer)) and not isinstance(x,bool)
        if not all(is_int(x) for x in (self.seed,self.horizon,self.search_start,self.schema_version)):
            raise SceneError("Seed, horizon, search_start, and schema version must be integers")
        if not isinstance(self.id,str) or not self.id or len(self.id)>200:
            raise SceneError("Scene identifier must be a nonempty string of at most 200 characters")
        if any(not isinstance(x,str) or not x or len(x)>80 for x in self.rooms):
            raise SceneError("Room names must be nonempty strings of at most 80 characters")
        if any(not is_int(x) for x in self.truth):
            raise SceneError("Truth locations must be integer room indices")
        if self.schema_version != 1:
            raise SceneError("Unsupported scene schema_version")
        if not 2 <= n <= 7 or len(set(self.rooms)) != n:
            raise SceneError("Use 2–7 uniquely named rooms; exact planning is factorial")
        if not self.id or self.horizon < 0 or not 0 <= self.search_start <= self.horizon:
            raise SceneError("Invalid scene identifier or time bounds")
        if self.horizon > 200:
            raise SceneError("Scene horizon exceeds the bounded-release limit of 200")
        if len(self.truth) != self.horizon+1 or any(not 0 <= x < n for x in self.truth):
            raise SceneError("truth must contain one valid room for every tick")
        if len(self.transitions) != self.horizon:
            raise SceneError("Provide a transition matrix for every tick interval")
        for t, matrix in enumerate(self.transitions):
            a = np.asarray(matrix, dtype=float)
            if a.shape != (n,n) or not np.isfinite(a).all() or (a < 0).any() or not np.allclose(a.sum(0),1):
                raise SceneError("Transitions must be finite, nonnegative, column-stochastic")
            if a[self.truth[t+1],self.truth[t]] <= 0:
                raise SceneError("The physical history has zero probability under the transition model")
            if t >= self.search_start and not np.allclose(a, np.eye(n)):
                raise SceneError("Parcel dynamics must be stationary during search")
            if t >= self.search_start and self.truth[t] != self.truth[t+1]:
                raise SceneError("The parcel cannot move during the search phase")
        if not math.isfinite(self.inspection_cost) or self.inspection_cost <= 0:
            raise SceneError("inspection_cost must be finite and positive")
        adj = {i:set() for i in range(n)}
        seen_edges = set()
        for u,v,w in self.edges:
            if not is_int(u) or not is_int(v) or not 0 <= u < n or not 0 <= v < n or u == v or not math.isfinite(w) or w <= 0:
                raise SceneError("Edges require distinct rooms and finite positive costs")
            key = tuple(sorted((u,v)))
            if key in seen_edges:
                raise SceneError("Duplicate undirected edge")
            seen_edges.add(key); adj[u].add(v); adj[v].add(u)
        reached, todo = {0}, [0]
        while todo:
            for v in adj[todo.pop()] - reached:
                reached.add(v); todo.append(v)
        if len(reached) != n:
            raise SceneError("Room graph must be connected")
        ids = {a.id for a in self.actors}
        if not 1 <= len(ids) == len(self.actors) <= 12:
            raise SceneError("Use 1–12 uniquely identified actors")
        for a in self.actors:
            if not isinstance(a.id,str) or not a.id or not isinstance(a.name,str) or not a.name or not is_int(a.room) or not isinstance(a.active,bool):
                raise SceneError("Actor IDs/names must be strings; rooms integers; active a boolean")
            p = np.asarray(a.prior)
            if not a.id or not 0 <= a.room < n or p.shape != (n,) or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(),1):
                raise SceneError("Invalid actor, location, or prior")
        if len(set(self.channels)) != len(self.channels):
            raise SceneError("Duplicate communication channel")
        for sender,recipient in self.channels:
            if sender not in ids or recipient not in ids or sender == recipient:
                raise SceneError("Invalid directed communication channel")
        roots = {o.id:o for o in self.observations}
        if len(roots) != len(self.observations) or len(roots) > 500:
            raise SceneError("Duplicate or excessive root observations")
        for o in self.observations:
            if not isinstance(o.id,str) or not is_int(o.time) or not is_int(o.reported_room):
                raise SceneError("Observation IDs must be strings and timestamps/rooms integers")
            if o.id.startswith(("inspection:","patch:")):
                raise SceneError("Observation IDs cannot use reserved runtime prefixes")
            if not o.id or o.observer not in ids or not 0 <= o.time <= self.horizon or not 0 <= o.reported_room < n:
                raise SceneError("Invalid observation identity, observer, timestamp, or room")
            if not math.isfinite(o.reliability) or not 0 <= o.reliability <= 1:
                raise SceneError("Reliability must be in [0,1]")
            p = o.probabilities(n)
            if not np.isfinite(p).all() or p.shape != (n,) or (p < 0).any() or (p > 1).any() or p.sum() <= 0:
                raise SceneError("Invalid observation likelihood")
            if p[self.truth[o.time]] <= 0:
                raise SceneError("An observation is impossible under its declared sensor model and physical history")
        messages = {m.id:m for m in self.messages}
        if len(messages) != len(self.messages) or len(messages) > 2000:
            raise SceneError("Duplicate or excessive messages")
        for m in self.messages:
            if not isinstance(m.id,str) or not is_int(m.time):
                raise SceneError("Message IDs must be strings and delivery times integers")
            if m.id.startswith(("inspection:","patch:")):
                raise SceneError("Message IDs cannot use reserved runtime prefixes")
            if m.id in roots or not m.id or m.root_id not in roots:
                raise SceneError("Unknown root or colliding message identifier")
            root = roots[m.root_id]
            if (m.sender,m.recipient) not in self.channels or not root.time <= m.time <= self.horizon:
                raise SceneError("Illegal channel or timestamp")
            if m.parent_id is None:
                if m.sender != root.observer:
                    raise SceneError("A non-observer forwarding a root requires a parent message")
            else:
                parent = messages.get(m.parent_id)
                if parent is None or parent.root_id != m.root_id or parent.recipient != m.sender or parent.time > m.time:
                    raise SceneError("Invalid provenance parent")
        # Reject cycles even when all timestamps coincide.
        visiting, visited = set(), set()
        def visit(mid: str) -> None:
            if mid in visiting: raise SceneError("Cyclic provenance")
            if mid in visited: return
            visiting.add(mid)
            parent = messages[mid].parent_id
            if parent: visit(parent)
            visiting.remove(mid); visited.add(mid)
        for mid in messages: visit(mid)

    @property
    def actor_ids(self) -> tuple[str, ...]:
        return tuple(a.id for a in self.actors)

    @property
    def hash(self) -> str:
        return digest(self.to_dict())

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False)+"\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict) -> Scene:
        d = dict(d)
        d["rooms"] = tuple(d["rooms"])
        d["edges"] = tuple(tuple(e) for e in d["edges"])
        d["actors"] = tuple(Actor(**{**a,"prior":tuple(a["prior"])}) for a in d["actors"])
        d["observations"] = tuple(Observation(**{**o,"likelihood":None if o.get("likelihood") is None else tuple(o["likelihood"])}) for o in d["observations"])
        d["messages"] = tuple(Message(**m) for m in d["messages"])
        d["transitions"] = tuple(tuple(tuple(row) for row in m) for m in d["transitions"])
        d["truth"] = tuple(d["truth"])
        d["channels"] = tuple(tuple(x) for x in d["channels"])
        return cls(**d)

    @classmethod
    def load(cls, path: str | Path) -> Scene:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

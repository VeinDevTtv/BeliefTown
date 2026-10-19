"""Exact expected-cost search-order planner for a stationary parcel."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import itertools
import numpy as np


@dataclass(frozen=True)
class Plan:
    destination: int
    order: tuple[int,...]
    expected_cost: float
    alternatives: tuple[tuple[tuple[int,...],float],...]


@lru_cache(maxsize=128)
def distances(n: int, edges: tuple) -> tuple[np.ndarray,np.ndarray]:
    d = np.full((n,n),np.inf); np.fill_diagonal(d,0)
    nxt = np.full((n,n),-1,dtype=int)
    for i in range(n): nxt[i,i]=i
    for u,v,w in edges:
        d[u,v]=d[v,u]=w; nxt[u,v]=v; nxt[v,u]=u
    for k in range(n):
        for i in range(n):
            for j in range(n):
                value = d[i,k]+d[k,j]
                if value < d[i,j]-1e-12:
                    d[i,j]=value; nxt[i,j]=nxt[i,k]
    d.setflags(write=False); nxt.setflags(write=False)
    return d,nxt


@lru_cache(maxsize=256)
def route_matrix(n: int, edges: tuple, start: int, inspection_cost: float):
    d,_ = distances(n,edges)
    orders = tuple(itertools.permutations(range(n)))
    matrix = np.zeros((len(orders),n))
    for k,order in enumerate(orders):
        previous, cumulative = start, 0.
        for room in order:
            cumulative += float(d[previous,room])+inspection_cost
            matrix[k,room]=cumulative; previous=room
    matrix.setflags(write=False)
    return orders,matrix


def plan_search(belief, room: int, edges: tuple, inspection_cost: float = 1.) -> Plan:
    b = np.asarray(belief,dtype=float)
    if b.ndim != 1 or not np.isfinite(b).all() or (b<0).any() or not np.isclose(b.sum(),1):
        raise ValueError("Planner needs a probability vector")
    orders,matrix = route_matrix(len(b),edges,room,inspection_cost)
    values = matrix @ b
    # Deterministic lexicographic tie break, including zero-probability rooms.
    best = int(np.argmin(values))
    order = orders[best]
    # Do not spend time inspecting rooms already ruled out. The route objective
    # for zero-mass rooms favors placing them last, except exact cost ties.
    destination = next(x for x in order if b[x] > 0)
    ranked = np.argsort(values,kind="stable")[:min(5,len(orders))]
    alternatives = tuple((orders[j],float(values[j])) for j in ranked)
    return Plan(destination,order,float(values[best]),alternatives)

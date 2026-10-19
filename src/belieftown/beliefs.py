"""Exact finite-state filtering, plus deliberately weaker benchmark baselines."""
from __future__ import annotations
import itertools
import numpy as np
from .schema import Observation


class InconsistentEvidence(ValueError):
    """The observation sequence has zero probability under the supplied model."""


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Probability weights must be finite and nonnegative")
    total = float(x.sum())
    if total <= 0:
        raise InconsistentEvidence("Evidence has zero probability; it cannot be silently ignored")
    return x/total


def entropy(belief: np.ndarray) -> float:
    p = np.asarray(belief)
    return float(-np.sum(p[p>0]*np.log(p[p>0])))


def filter_belief(prior, transitions, roots: list[Observation] | tuple[Observation,...], time: int,
                  *, deduplicate: bool = True, return_history: bool = False):
    """Recompute chronologically; old evidence is attached to its observation time.

    Log-domain accumulation avoids product underflow. Roots are
    sorted canonically so arrival-order changes do not introduce rounding drift.
    """
    if time < 0 or time > len(transitions):
        raise ValueError("Filter time lies outside the transition history")
    n = len(prior)
    if deduplicate:
        unique = {}
        for r in roots:
            if r.id in unique and r != unique[r.id]:
                raise ValueError("A root identifier cannot refer to different observations")
            unique[r.id] = r
        roots = list(unique.values())
    grouped: dict[int,list[Observation]] = {}
    for r in sorted(roots, key=lambda o:(o.time,o.id)):
        if not 0 <= r.time <= time:
            raise ValueError("Cannot condition on an observation from the future")
        grouped.setdefault(r.time,[]).append(r)
    p = normalize(np.asarray(prior))
    # Retain log probabilities internally: hundreds of conflicting observations
    # must not irreversibly round a possible state down to zero mid-update.
    with np.errstate(divide="ignore"):
        log_b = np.log(p)
    history = []
    for t in range(time+1):
        if t:
            matrix = np.asarray(transitions[t-1], dtype=float)
            if matrix.shape != (n,n) or not np.isfinite(matrix).all() or (matrix<0).any() or not np.allclose(matrix.sum(0),1):
                raise ValueError("Invalid transition matrix")
            with np.errstate(divide="ignore"):
                log_b = np.logaddexp.reduce(np.log(matrix)+log_b[None,:],axis=1)
        for root in grouped.get(t,[]):
            likelihood=root.probabilities(n)
            if likelihood.shape != (n,) or not np.isfinite(likelihood).all() or (likelihood<0).any() or (likelihood>1).any():
                raise ValueError("Invalid observation likelihood")
            with np.errstate(divide="ignore"):
                log_b += np.log(likelihood)
        log_z=float(np.logaddexp.reduce(log_b))
        if not np.isfinite(log_z):
            raise InconsistentEvidence("Evidence has zero probability under the model")
        log_b -= log_z
        b=normalize(np.exp(log_b))
        history.append(b.copy())
    return (b,history) if return_history else b



def last_report_belief(prior, transitions, received: list[Observation], time: int) -> np.ndarray:
    """Last *received* report only, still propagated from the time it describes."""
    if not received:
        return filter_belief(prior,transitions,[],time)
    return filter_belief(prior,transitions,[received[-1]],time)


def brute_force_belief(prior, transitions, roots, time: int) -> np.ndarray:
    """Independent exponential reference for tests, not the runtime implementation."""
    n = len(prior)
    if n**(time+1) > 2_000_000:
        raise ValueError("Reference enumeration intentionally limited to small cases")
    roots = {r.id:r for r in roots}.values()
    result = np.zeros(n)
    for path in itertools.product(range(n), repeat=time+1):
        weight = float(prior[path[0]])
        for t in range(time):
            weight *= transitions[t][path[t+1]][path[t]]
        for root in roots:
            weight *= root.probabilities(n)[path[root.time]]
        result[path[-1]] += weight
    return normalize(result)

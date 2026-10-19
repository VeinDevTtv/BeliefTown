"""Authored demonstrations and seed-addressable procedural benchmark families."""
from __future__ import annotations
from dataclasses import replace
from importlib.resources import files
from pathlib import Path
import numpy as np
from .schema import Actor, Observation, Message, Scene
from .randomness import keyed_rng

FAMILIES=("missed_relocation","delayed_report","rumor_loop","independent_corroboration","conflicting_evidence","constrained_patch")


def matrices(n: int, horizon: int, search_start: int, mobility: float=0.15):
    return tuple(tuple(tuple(float(x) for x in row) for row in
                       ((1-mobility)*np.eye(n)+mobility*np.ones((n,n))/n if t<search_start else np.eye(n)))
                 for t in range(horizon))


def demo_scene(family: str="missed_relocation") -> Scene:
    if family not in FAMILIES: raise ValueError(f"Choose one of {FAMILIES}")
    n,horizon,start=4,15,4
    rooms=("Workshop","Library","Garden","Hall")
    # Hall connects to the three task rooms; all actors begin in the Hall.
    edges=((0,3,1.),(1,3,1.),(2,3,1.),(0,2,1.5))
    prior=(0.32,0.32,0.32,0.04)
    actors=(Actor("courier","Courier",3,prior,True),Actor("mira","Mira",3,prior,True),Actor("theo","Theo",3,prior,True))
    channels=tuple((a.id,b.id) for a in actors for b in actors if a.id!=b.id)
    roots=[Observation("old-workshop","theo",0,0,0.86),
           Observation("new-library","mira",3,1,0.94),
           Observation("garden-glimpse","theo",2,2,0.42),
           Observation("hall-empty","courier",4,3,1.,"inspection_negative",(1.,1.,1.,0.))]
    msgs=[Message("old-to-courier","old-workshop","theo","courier",1),
          Message("old-to-mira","old-workshop","theo","mira",1),
          Message("rumor-back","old-workshop","mira","theo",2,"old-to-mira")]
    transition=matrices(n,horizon,start,0.20)
    truth=(0,0,1,1,1)+tuple(1 for _ in range(horizon-4))
    description="An old workshop report competes with an unseen library observation."
    if family=="delayed_report":
        msgs.append(Message("late-old-report","old-workshop","theo","courier",4))
        description="A late-arriving report still describes turn 0, not the current parcel location."
    elif family=="rumor_loop":
        msgs.extend([Message("copy-to-courier","old-workshop","mira","courier",3,"old-to-mira"),
                     Message("copy-again","old-workshop","theo","courier",4,"rumor-back")])
        description="Three deliveries repeat one observation; confidence must not inflate."
    elif family=="independent_corroboration":
        roots.append(Observation("independent-library","theo",3,1,0.84))
        description="Two genuinely independent library measurements can corroborate each other."
    elif family=="conflicting_evidence":
        roots.extend([Observation("recent-workshop","theo",3,0,0.75),
                      Observation("weak-library","mira",2,1,0.6)])
        msgs.append(Message("recent-to-courier","recent-workshop","theo","courier",4))
        description="Conflicting sources differ in reliability and observation time."
    elif family=="constrained_patch":
        description="Protect the courier from new deliveries to create an explicit no-solution case."
    return Scene(f"demo-{family}",7,rooms,edges,actors,tuple(roots),tuple(msgs),transition,truth,start,horizon,channels,
                 family=family,description=description)


def procedural_scene(seed: int, family: str, *, n_rooms: int=4, ood: bool=False) -> Scene:
    if family not in FAMILIES: raise ValueError("Unknown procedural family")
    rng=keyed_rng(seed,"scene",family,n_rooms,ood)
    n=n_rooms; start=int(rng.integers(3,6)); horizon=start+2*n+4
    # Connected random graph; held-out scenes use more rooms and longer delays.
    edges=[]
    for i in range(1,n):
        parent=int(rng.integers(0,i)); edges.append((parent,i,float(rng.choice([1.,1.5,2.]))))
    existing={frozenset((u,v)) for u,v,_ in edges}
    for i in range(n):
        for j in range(i+1,n):
            if frozenset((i,j)) not in existing and rng.random()<(.45 if ood else .20):
                edges.append((i,j,float(rng.choice([1.,1.5,2.]))))
    priors=[tuple(float(x) for x in rng.dirichlet(np.ones(n)*3)) for _ in range(3)]
    ids=("courier","mira","theo")
    actors=tuple(Actor(i,i.title(),int(rng.integers(n)),p,j==0) for j,(i,p) in enumerate(zip(ids,priors)))
    channels=tuple((a,b) for a in ids for b in ids if a!=b)
    mobility=float(rng.uniform(.08,.30) if not ood else rng.uniform(.35,.55))
    if family in ("rumor_loop","independent_corroboration"): mobility=0.
    transitions=matrices(n,horizon,start,mobility)
    truth=[int(rng.integers(n))]
    for t in range(horizon):
        r=keyed_rng(seed,"truth",t)
        truth.append(int(r.choice(n,p=np.asarray(transitions[t])[:,truth[-1]])))
    observations=[]
    nroots=int(rng.integers(5,9))
    for k in range(nroots):
        owner=ids[1+k%2]
        obs_time=int(rng.integers(0,start+1))
        if family=="delayed_report": obs_time=int(rng.integers(0,max(1,start-1)))
        if family=="independent_corroboration": obs_time=start
        reliability=float(rng.uniform(.52,.95) if not ood else rng.uniform(.37,.99))
        r=keyed_rng(seed,"sighting",k)
        probs=np.full(n,(1-reliability)/(n-1)); probs[truth[obs_time]]=reliability
        report=int(r.choice(n,p=probs))
        observations.append(Observation(f"root-{k}",owner,obs_time,report,reliability))
    messages=[]
    # A subset is already known to the target; additional legal duplicates remain
    # in the patch universe as controls. No observation is selected by a label.
    for k,o in enumerate(observations):
        if rng.random()<.27 or k==0:
            t=int(rng.integers(o.time,start+1))
            messages.append(Message(f"initial-{k}",o.id,o.observer,"courier",t))
    if family=="rumor_loop":
        root=observations[0]
        other="theo" if root.observer=="mira" else "mira"
        messages.extend([Message("relay-1",root.id,root.observer,other,start),
                         Message("relay-2",root.id,other,"courier",start,"relay-1"),
                         Message("relay-3",root.id,"courier",root.observer,start,"relay-2")])
    return Scene(f"{family}-{seed}-{'ood' if ood else 'id'}",seed,tuple(f"Room {i}" for i in range(n)),
                 tuple(edges),actors,tuple(observations),tuple(messages),transitions,tuple(truth),start,horizon,channels,
                 family=family,description="Procedurally generated; labels come from exact patch execution.")


def load_builtin(name: str="missed_relocation") -> Scene:
    return demo_scene(name)


def save_examples(directory: str | Path) -> list[Path]:
    out=Path(directory); out.mkdir(parents=True,exist_ok=True); paths=[]
    for family in FAMILIES:
        path=out/f"{family}.json"; demo_scene(family).save(path); paths.append(path)
    return paths

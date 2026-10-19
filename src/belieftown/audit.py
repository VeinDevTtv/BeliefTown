"""Portable audit bundles contain the full scene and replay controls."""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
from .schema import Scene
from .world import Episode
from .interventions import Delivery
from .verifier import check_audit


def episode_spec(episode: Episode) -> dict:
    return {'scene':episode.scene.to_dict(),'mode':episode.mode,
            'suppress_observations':sorted(episode.suppress_observations),
            'suppress_messages':sorted(episode.suppress_messages),
            'patches':{str(t):[asdict(d) for d in patch] for t,patch in episode.patches.items()}}


def episode_from_spec(d: dict) -> Episode:
    return Episode(Scene.from_dict(d['scene']),mode=d.get('mode','provenance'),
                   suppress_observations=d.get('suppress_observations',()),
                   suppress_messages=d.get('suppress_messages',()),
                   patches={int(t):[Delivery(**x) for x in p] for t,p in d.get('patches',{}).items()})


def make_bundle(episode: Episode, time: int, result, *, model_metadata: dict|None=None) -> dict:
    if episode.at(time).checkpoint_hash!=result.checkpoint_hash:
        raise ValueError('Search result belongs to another snapshot')
    return json.loads(json.dumps({'schema_version':1,'kind':'belieftown.audit','episode':episode_spec(episode),
            'time':time,'record':result.to_dict(),'model_metadata':model_metadata},allow_nan=False))


def verify_bundle(bundle: dict,*,certify_minimum: bool=True) -> dict:
    if bundle.get('kind')!='belieftown.audit' or bundle.get('schema_version')!=1:
        raise ValueError('Not a supported BeliefTown audit bundle')
    episode=episode_from_spec(bundle['episode'])
    return check_audit(episode.at(int(bundle['time'])),bundle['record'],certify_minimum=certify_minimum)


def save_bundle(bundle: dict,path: str|Path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(bundle,indent=2,allow_nan=False)+'\n', encoding="utf-8")

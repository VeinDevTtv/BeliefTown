"""Whole-scene split manifests and exact supervised candidate labels."""
from __future__ import annotations
from dataclasses import dataclass,asdict
import itertools
import json
from pathlib import Path
import numpy as np
from ..scenarios import procedural_scene,FAMILIES
from ..schema import Scene,digest
from ..world import Episode
from ..interventions import PatchProblem,Delivery,candidate_deliveries,verify_patch
from ..randomness import keyed_rng
from .graph import encode_graph


@dataclass
class Case:
    scene: Scene
    problem: PatchProblem
    split: str
    labels: list[dict]
    @property
    def snapshot(self): return Episode(self.scene).at(self.scene.search_start)
    def to_dict(self):
        return {"scene":self.scene.to_dict(),"problem":asdict(self.problem),"split":self.split,"labels":self.labels}
    @classmethod
    def from_dict(cls,d):
        p=dict(d["problem"]); p["protected_actors"]=tuple(p.get("protected_actors",()))
        return cls(Scene.from_dict(d["scene"]),PatchProblem(**p),d["split"],d["labels"])


def make_dataset(directory: str|Path,*,train: int=120,validation: int=24,test: int=48,ood: int=24,seed: int=20261018):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    manifest={"schema_version":1,"seed":seed,"split_unit":"entire scene, all candidate patches together",
              "protocol":"All patches up to cardinality 3 labeled by exact execution; no sample-level random split.","splits":{}}
    all_hashes=set()
    for split,count,offset in (("train",train,0),("validation",validation,100000),("test",test,200000),("ood",ood,300000)):
        path=directory/f"{split}.jsonl"; stats={"scenes":count,"candidates":0,"positive_candidates":0,"family_counts":{}}
        with path.open('w', encoding="utf-8") as f:
            for j in range(count):
                family=FAMILIES[j%len(FAMILIES)]
                s=procedural_scene(seed+offset+j,family,n_rooms=(5+j%2 if split=="ood" else 4),ood=split=="ood")
                snap=Episode(s).at(s.search_start)
                rng=keyed_rng(seed+offset+j,"target")
                target=int(rng.integers(len(s.rooms)))
                # Most episodes request a behavioral change, but preserve some
                # zero-patch goals so the empty-patch case is learned and tested.
                current=snap.plan("courier").destination
                if j%10 and target==current: target=(target+1)%len(s.rooms)
                protected=("courier",) if family=="constrained_patch" and (j//len(FAMILIES))%2 else ()
                problem=PatchProblem("courier",target,3,protected,24)
                candidates=candidate_deliveries(snap,problem)
                labels=[]
                for k in range(min(3,len(candidates))+1):
                    for inds in itertools.combinations(range(len(candidates)),k):
                        v=verify_patch(snap,problem,tuple(candidates[q] for q in inds))
                        labels.append({"indices":list(inds),"feasible":v.feasible})
                case=Case(s,problem,split,labels)
                f.write(json.dumps(case.to_dict(),separators=(',',':'),allow_nan=False)+'\n')
                stats["candidates"]+=len(labels); stats["positive_candidates"]+=sum(x['feasible'] for x in labels)
                stats["family_counts"][family]=stats["family_counts"].get(family,0)+1
                if s.hash in all_hashes: raise RuntimeError("Scene split leakage")
                all_hashes.add(s.hash)
        stats["sha256"]=__import__('hashlib').sha256(path.read_bytes()).hexdigest()
        manifest["splits"][split]=stats
    manifest["scene_hash_count"]=len(all_hashes)
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n', encoding="utf-8")
    return manifest


def load_cases(path: str|Path) -> list[Case]:
    with Path(path).open(encoding="utf-8") as f: return [Case.from_dict(json.loads(line)) for line in f if line.strip()]


def training_graphs(cases: list[Case],*,strip_provenance=False):
    graphs=[]
    for case in cases:
        snap=case.snapshot; candidates=candidate_deliveries(snap,case.problem)
        for label in case.labels:
            patch=tuple(candidates[i] for i in label["indices"])
            g=encode_graph(snap,case.problem,patch,strip_provenance=strip_provenance)
            g.y=float(label["feasible"]); graphs.append(g)
    return graphs

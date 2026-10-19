"""UI-independent application service, shared by the browser and notebooks."""
from __future__ import annotations
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import threading
from .scenarios import demo_scene,procedural_scene,FAMILIES
from .schema import Scene
from .world import Episode
from .interventions import Delivery,PatchProblem,search,apply_patch
from .audit import make_bundle,episode_spec


class Studio:
    def __init__(self,scene: Scene|None=None,checkpoint_dir: str|Path|None=None):
        self.lock=threading.RLock();self.scene=scene or demo_scene()
        self.baseline=Episode(self.scene);self.episode=self.baseline
        self.last_result=None;self.last_bundle=None;self.last_time=None
        self.checkpoint_dir=None if checkpoint_dir is None else Path(checkpoint_dir)
        self.models={};self.model_paths={}
        if self.checkpoint_dir and self.checkpoint_dir.exists() and importlib.util.find_spec('torch'):
            for kind in ('graph','mlp','graph_no_provenance'):
                choices=[]
                for path in self.checkpoint_dir.glob(f'{kind}_seed*.pt'):
                    try: score=float(json.loads(path.with_suffix('.json').read_text(encoding="utf-8"))['validation_bce'])
                    except (OSError,ValueError,KeyError): score=float('inf')
                    choices.append((score,str(path),path))
                if choices:self.model_paths[kind]=min(choices)[2]

    def _load_model(self,kind):
        if kind not in self.model_paths: raise ValueError('This learned model is not installed; use the heuristic or train checkpoints')
        if kind not in self.models:
            from .learning.model import Predictor,configure
            configure(0);self.models[kind]=Predictor.load(self.model_paths[kind])
        return self.models[kind]

    def state(self):
        return {'scene':self.scene.to_dict(),'families':list(FAMILIES),
                'baseline':[s.inspector(self.baseline.mode) for s in self.baseline.snapshots],
                'branch':[s.inspector(self.episode.mode) for s in self.episode.snapshots],
                'mode':self.episode.mode,'suppressed_observations':sorted(self.episode.suppress_observations),
                'suppressed_messages':sorted(self.episode.suppress_messages),
                'patched_times':sorted(self.episode.patches),'models':list(self.model_paths),
                'model_files':{k:v.name for k,v in self.model_paths.items()},'offline':False}

    def _invalidate_search(self):
        self.last_result=None;self.last_bundle=None;self.last_time=None

    def load(self,args):
        if 'scene' in args:scene=Scene.from_dict(args['scene'])
        elif 'seed' in args:scene=procedural_scene(int(args['seed']),args.get('family',FAMILIES[0]),n_rooms=int(args.get('rooms',4)))
        else:scene=demo_scene(args.get('family',FAMILIES[0]))
        self.scene=scene;self.baseline=Episode(scene);self.episode=self.baseline;self._invalidate_search()
        return self.state()

    def reset(self,args=None):
        self.baseline=Episode(self.scene);self.episode=self.baseline;self._invalidate_search();return self.state()

    def fork(self,args):
        observations=set(self.episode.suppress_observations)
        messages=set(self.episode.suppress_messages)
        if args.get('observation'):observations.add(args['observation'])
        if args.get('message'):messages.add(args['message'])
        mode=args.get('mode',self.episode.mode)
        # A diagnostic fork is evaluated without earlier actionable patches:
        # suppressing their sources could otherwise make those patches illegal.
        new=Episode(self.scene,suppress_observations=observations,suppress_messages=messages,mode=mode)
        self.episode=new;self.baseline=Episode(self.scene,mode=mode);self._invalidate_search()
        return self.state()

    def find_patch(self,args):
        if self.episode.mode!='provenance':raise ValueError('Verified patch objectives use provenance-aware policies. Switch the policy mode back first.')
        t=int(args.get('time',self.scene.search_start));snap=self.episode.at(t)
        if t in self.episode.patches:raise ValueError('A patch batch already exists at this tick. Reset or advance to another tick.')
        protected=tuple(args.get('protected',()))
        p=PatchProblem(args.get('actor','courier'),int(args['room']),int(args.get('max_size',3)),protected,int(args.get('candidate_limit',24)))
        method=args.get('method','heuristic');model=None
        if method not in ('uninformed','heuristic'):model=self._load_model(method)
        budget=args.get('budget');budget=None if budget is None else int(budget)
        result=search(snap,p,method=method,ranker=model,budget=budget)
        bundle=make_bundle(self.episode,t,result,model_metadata=None if model is None else model.metadata)
        self.last_result=result;self.last_bundle=bundle;self.last_time=t
        after=None if result.patch is None else apply_patch(snap,result.patch).inspector()
        return {'record':result.to_dict(),'bundle':bundle,'before':snap.inspector(),'after':after}

    def apply_solution(self,args=None):
        r=self.last_result;t=self.last_time
        if r is None or r.patch is None:raise ValueError('Search for a feasible patch first')
        if self.episode.at(t).checkpoint_hash!=r.checkpoint_hash:raise ValueError('The search snapshot changed; search again')
        if t in self.episode.patches:raise ValueError('Only one simultaneous patch batch is allowed per tick')
        patches={k:list(v) for k,v in self.episode.patches.items()};patches[t]=list(r.patch)
        self.episode=Episode(self.scene,suppress_observations=self.episode.suppress_observations,
            suppress_messages=self.episode.suppress_messages,patches=patches,mode=self.episode.mode)
        return self.state()

    def deliver(self,args):
        t=int(args.get('time',self.scene.search_start))
        if t<self.scene.search_start:raise ValueError('Deliver patches at search-phase decision ticks')
        if t in self.episode.patches:raise ValueError('Reset or advance before applying another patch batch')
        d=Delivery(args['sender'],args['recipient'],args['root_id'])
        apply_patch(self.episode.at(t),[d]) # validate transaction before mutating state
        patches={k:list(v) for k,v in self.episode.patches.items()};patches[t]=[d]
        episode=Episode(self.scene,suppress_observations=self.episode.suppress_observations,
            suppress_messages=self.episode.suppress_messages,patches=patches,mode=self.episode.mode)
        self.episode=episode;self._invalidate_search();return self.state()

    def export(self):
        return {'schema_version':1,'kind':'belieftown.episode','episode':episode_spec(self.episode),
                'frames':[s.inspector(self.episode.mode) for s in self.episode.snapshots]}

    def dispatch(self,action,args):
        actions={'state':lambda _:self.state(),'scene':self.load,'reset':self.reset,'fork':self.fork,
                 'search':self.find_patch,'apply':self.apply_solution,'deliver':self.deliver,'episode':lambda _:self.export()}
        if action not in actions:raise ValueError('Unknown API action')
        with self.lock:return actions[action](args)

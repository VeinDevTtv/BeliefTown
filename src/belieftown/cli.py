"""Command-line interface for the complete offline research workflow."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
import webbrowser
from .schema import Scene
from .scenarios import demo_scene,procedural_scene,FAMILIES
from .world import Episode
from .interventions import PatchProblem,search
from .audit import make_bundle,verify_bundle,save_bundle,episode_from_spec,episode_spec


def _json(data):print(json.dumps(data,indent=2,allow_nan=False))

def _scene(args):return Scene.load(args.scene) if getattr(args,'scene',None) else demo_scene(getattr(args,'family',FAMILIES[0]))

def _room(scene,value):
    try:return int(value)
    except ValueError:
        for i,name in enumerate(scene.rooms):
            if name.casefold()==value.casefold():return i
        raise ValueError(f'Unknown room {value!r}; use one of {scene.rooms}')


def parser():
    p=argparse.ArgumentParser(prog='belieftown',description='Counterfactual NPC debugger with exactly verified information patches')
    p.add_argument('--version',action='version',version='BeliefTown 1.0.0')
    sub=p.add_subparsers(dest='command',required=True)
    demo=sub.add_parser('demo',help='Start the local browser / notebook application')
    demo.add_argument('--port',type=int,default=8765);demo.add_argument('--scene',type=Path)
    demo.add_argument('--family',choices=FAMILIES,default=FAMILIES[0]);demo.add_argument('--checkpoints',type=Path,default=Path('artifacts/checkpoints'))
    demo.add_argument('--no-browser',action='store_true')
    solve=sub.add_parser('solve',help='Find a patch and write a portable audit bundle')
    solve.add_argument('--scene',type=Path);solve.add_argument('--family',choices=FAMILIES,default=FAMILIES[0])
    solve.add_argument('--time',type=int);solve.add_argument('--actor',default='courier');solve.add_argument('--room',default='Library')
    solve.add_argument('--max-size',type=int,default=3);solve.add_argument('--candidate-limit',type=int,default=24)
    solve.add_argument('--protect',action='append',default=[]);solve.add_argument('--budget',type=int)
    solve.add_argument('--method',choices=('uninformed','heuristic','graph','mlp','graph_no_provenance'),default='heuristic')
    solve.add_argument('--checkpoint',type=Path);solve.add_argument('--output',type=Path,default=Path('artifacts/audit.json'))
    verify=sub.add_parser('verify',help='Independently recompute an audit, including its minimum claim')
    verify.add_argument('audit',type=Path);verify.add_argument('--feasibility-only',action='store_true')
    replay=sub.add_parser('replay',help='Apply the verified patch in an audit and export the resulting episode')
    replay.add_argument('audit',type=Path);replay.add_argument('--output',type=Path,default=Path('artifacts/replay.json'))
    generate=sub.add_parser('generate',help='Write a validated procedural scene JSON')
    generate.add_argument('--seed',type=int,default=23);generate.add_argument('--rooms',type=int,choices=range(2,8),default=4)
    generate.add_argument('--family',choices=FAMILIES,default=FAMILIES[0]);generate.add_argument('--output',type=Path,default=Path('scenarios/generated.json'))
    data=sub.add_parser('dataset',help='Generate complete scene-level training/evaluation splits')
    data.add_argument('--output',type=Path,default=Path('benchmarks/data'));data.add_argument('--seed',type=int,default=20261018)
    for name,value in [('train',120),('validation',24),('test',48),('ood',24)]:data.add_argument('--'+name,type=int,default=value)
    train=sub.add_parser('train',help='Train a graph model or a non-graph comparator')
    train.add_argument('--data',type=Path,default=Path('benchmarks/data'));train.add_argument('--kind',choices=('graph','mlp','graph_no_provenance'),default='graph')
    train.add_argument('--seed',type=int,default=0);train.add_argument('--epochs',type=int,default=14)
    train.add_argument('--output',type=Path,default=Path('artifacts/checkpoints/custom.pt'))
    bench=sub.add_parser('benchmark',help='Compare every released model seed against exact-search baselines')
    bench.add_argument('--data',type=Path,default=Path('benchmarks/data'));bench.add_argument('--checkpoints',type=Path,default=Path('artifacts/checkpoints'))
    bench.add_argument('--output',type=Path,default=Path('artifacts/evaluation'));bench.add_argument('--limit',type=int)
    bench.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    export=sub.add_parser('export',help='Write the self-contained, recorded HTML showcase')
    export.add_argument('--output',type=Path,default=Path('artifacts/BeliefTown_Showcase.html'))
    nb=sub.add_parser('notebooks',help='Execute every notebook; fail on any error')
    nb.add_argument('--directory',type=Path,default=Path('notebooks'));nb.add_argument('--output',type=Path,default=Path('artifacts/executed_notebooks'))
    nb.add_argument('--timeout',type=int,default=240)
    nb.add_argument('--run-root',type=Path,help='Use a reproduction directory containing data/checkpoints/evaluation')
    reproduce=sub.add_parser('reproduce',help='One-command data → training → evaluation → showcase pipeline')
    reproduce.add_argument('--profile',choices=('quick','release'),default='quick');reproduce.add_argument('--output',type=Path,default=Path('artifacts/reproduction'))
    reproduce.add_argument('--notebooks',action='store_true')
    sub.add_parser('doctor',help='Report installed optional components and verify the canonical scene')
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    try:return _run(args)
    except KeyboardInterrupt:
        print('\nStopped.');return 130
    except (ValueError,KeyError,OSError,ImportError) as e:
        print(f'BeliefTown: {e}',file=sys.stderr)
        if isinstance(e,ImportError):print('Install the matching optional dependencies: python -m pip install -e ".[all]"',file=sys.stderr)
        return 2


def _run(a):
    if a.command=='demo':
        from .server import StudioServer
        from .studio import Studio
        with StudioServer(Studio(_scene(a),a.checkpoints),a.port) as server:
            print(f'BeliefTown: {server.url}\nLocal development server only. Press Ctrl+C to stop.',flush=True)
            if not a.no_browser:webbrowser.open(server.url)
            while server.thread.is_alive():server.thread.join(timeout=1)
    elif a.command=='solve':
        scene=_scene(a);episode=Episode(scene);t=scene.search_start if a.time is None else a.time
        problem=PatchProblem(a.actor,_room(scene,a.room),a.max_size,tuple(a.protect),a.candidate_limit)
        model=None
        if a.method not in ('uninformed','heuristic'):
            from .learning.model import Predictor,configure
            configure(0);path=a.checkpoint or Path(f'artifacts/checkpoints/{a.method}_seed0.pt');model=Predictor.load(path)
            if model.kind!=a.method:raise ValueError('Checkpoint model kind does not match --method')
        result=search(episode.at(t),problem,method=a.method,ranker=model,budget=a.budget)
        bundle=make_bundle(episode,t,result,model_metadata=None if model is None else model.metadata);save_bundle(bundle,a.output)
        _json({'status':result.status,'patch':None if result.patch is None else [asdict(d) for d in result.patch],
               'verifier_calls':result.verifier_calls,'audit':str(a.output)})
    elif a.command=='verify':
        _json(verify_bundle(json.loads(a.audit.read_text(encoding="utf-8")),certify_minimum=not a.feasibility_only))
    elif a.command=='replay':
        bundle=json.loads(a.audit.read_text(encoding="utf-8"));verify_bundle(bundle)
        episode=episode_from_spec(bundle['episode']);t=int(bundle['time']);raw=bundle['record']['patch']
        if raw is None:raise ValueError('This audit does not contain a feasible patch to replay')
        from .interventions import Delivery
        patches={k:list(v) for k,v in episode.patches.items()}
        if t in patches:raise ValueError('Cannot combine sequential patch rounds at the same logical tick')
        patches[t]=[Delivery(**x) for x in raw]
        result=Episode(episode.scene,mode=episode.mode,suppress_messages=episode.suppress_messages,
                       suppress_observations=episode.suppress_observations,patches=patches)
        output={'schema_version':1,'kind':'belieftown.episode','episode':episode_spec(result),
                'frames':[s.inspector(result.mode) for s in result.snapshots]}
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(output,indent=2)+'\n', encoding="utf-8");print(a.output)
    elif a.command=='generate':
        procedural_scene(a.seed,a.family,n_rooms=a.rooms).save(a.output);print(a.output)
    elif a.command=='dataset':
        from .learning.data import make_dataset
        _json(make_dataset(a.output,train=a.train,validation=a.validation,test=a.test,ood=a.ood,seed=a.seed))
    elif a.command=='train':
        from .learning.train import train_model
        model=train_model(a.data,a.output,kind=a.kind,seed=a.seed,epochs=a.epochs)
        _json({k:v for k,v in model.metadata.items() if k!='history'})
    elif a.command=='benchmark':
        from .benchmark import run_benchmark
        r=run_benchmark(a.data,a.checkpoints,a.output,limit=a.limit,seeds=tuple(a.seeds));_json({'searches':r['searches'],'all_correct':r['all_searches_correct'],'output':str(a.output)})
    elif a.command=='export':
        from .export import export_showcase
        print(export_showcase(a.output))
    elif a.command=='notebooks':
        from .notebook_runner import execute_notebooks
        _json(execute_notebooks(a.directory,a.output,timeout=a.timeout,run_root=a.run_root))
    elif a.command=='reproduce':
        from .learning.data import make_dataset
        from .learning.train import train_model
        from .benchmark import run_benchmark
        from .export import export_showcase
        out=a.output;out.mkdir(parents=True,exist_ok=True)
        release=a.profile=='release';sizes=(120,24,48,24) if release else (24,6,12,6)
        seeds=(0,1,2) if release else (0,);epochs=14 if release else 3
        make_dataset(out/'data',train=sizes[0],validation=sizes[1],test=sizes[2],ood=sizes[3])
        for kind in ('graph','mlp','graph_no_provenance'):
            for seed in seeds:
                print(f'Training {kind}, seed {seed}',flush=True)
                train_model(out/'data',out/f'checkpoints/{kind}_seed{seed}.pt',kind=kind,seed=seed,epochs=epochs)
        r=run_benchmark(out/'data',out/'checkpoints',out/'evaluation',seeds=seeds)
        export_showcase(out/'BeliefTown_Showcase.html')
        if a.notebooks:
            from .notebook_runner import execute_notebooks
            execute_notebooks('notebooks',out/'executed_notebooks',run_root=out)
        _json({'profile':a.profile,'all_searches_correct':r['all_searches_correct'],'output':str(out)})
    elif a.command=='doctor':
        components={}
        for name in ('numpy','torch','jupyterlab','nbformat','nbclient','matplotlib'):
            try:components[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:components[name]=None
        scene=demo_scene();episode=Episode(scene);r=search(episode.at(scene.search_start),PatchProblem('courier',1))
        _json({'python':sys.version.split()[0],'components':components,'canonical_scene':r.status,
               'network_required_at_runtime':False,'numpy_only_core':True})
    return 0

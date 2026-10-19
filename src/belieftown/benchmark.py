"""Paired, scene-level evaluation. Correctness, verifier effort, and latency differ."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import platform
import time
import sys
import numpy as np
from .interventions import search
from .learning.data import load_cases
from .learning.model import Predictor,configure


def bootstrap_mean(values,*,seed: int=0,repetitions: int=2000) -> dict:
    x=np.asarray(values,dtype=float)
    if not len(x): return {"mean":None,"low":None,"high":None,"n_scenes":0}
    rng=np.random.default_rng(seed)
    means=x[rng.integers(0,len(x),size=(repetitions,len(x)))].mean(1)
    return {"mean":float(x.mean()),"low":float(np.quantile(means,.025)),"high":float(np.quantile(means,.975)),"n_scenes":len(x)}


def _scene_summary(rows,metric):
    # Average the three training-seed outcomes within scene before bootstrapping.
    # Candidate patches / training-seed runs are NOT treated as independent scenes.
    grouped=defaultdict(list)
    for r in rows:
        v=r.get(metric)
        if v is not None: grouped[r['scene_id']].append(float(v))
    return bootstrap_mean([np.mean(v) for v in grouped.values()])


def summarize(rows):
    result=[]
    for split in sorted({r['split'] for r in rows}):
        for method in sorted({r['method'] for r in rows}):
            group=[r for r in rows if r['split']==split and r['method']==method]
            if not group: continue
            solvable=[r for r in group if r['oracle_minimum'] is not None]
            entry={"split":split,"method":method,"runs":len(group),
                "verifier_calls":_scene_summary(group,'verifier_calls'),
                "latency_ms":_scene_summary(group,'latency_ms'),
                "ranking_ms":_scene_summary(group,'ranking_ms'),
                "solvable_latency_ms":_scene_summary(solvable,'latency_ms'),
                "solvable_calls":_scene_summary(solvable,'verifier_calls'),
                "gap":_scene_summary(solvable,'minimum_gap'),
                "correctness":_scene_summary(group,'correct'),
                "success_at_4":_scene_summary(solvable,'success_at_4'),
                "success_at_16":_scene_summary(solvable,'success_at_16')}
            result.append(entry)
    return result


def run_benchmark(data_dir: str|Path,checkpoint_dir: str|Path,output_dir: str|Path,*,limit: int|None=None,seeds=(0,1,2)) -> dict:
    if limit is not None and (not isinstance(limit,int) or isinstance(limit,bool) or limit<1):
        raise ValueError("Benchmark limit must be a positive integer")
    if not seeds or len(set(seeds))!=len(seeds):
        raise ValueError("Provide nonempty, distinct model seeds")
    configure(0);data_dir=Path(data_dir);checkpoint_dir=Path(checkpoint_dir);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    models={}
    for kind in ('graph','mlp','graph_no_provenance'):
        for seed in seeds:
            path=checkpoint_dir/f'{kind}_seed{seed}.pt'
            if not path.exists(): raise FileNotFoundError(f'Missing required checkpoint: {path}')
            models[(kind,seed)]=Predictor.load(path)
    rows=[];belief_rows=[];start=time.perf_counter()
    checkpoint_hashes={f'{k[0]}_seed{k[1]}':hashlib.sha256((checkpoint_dir/f'{k[0]}_seed{k[1]}.pt').read_bytes()).hexdigest() for k in models}
    # Warm the tensor runtime before measured requests; model loading is an
    # amortized startup cost, excluded from steady-state search request latency.
    warm_cases=load_cases(data_dir/'test.jsonl')
    if not warm_cases or not load_cases(data_dir/'ood.jsonl'):
        raise ValueError('Benchmark requires nonempty test and OOD splits')
    warm=warm_cases[0];ws=warm.snapshot
    for model in models.values(): model.score(ws,warm.problem,[()])
    for split in ('test','ood'):
        cases=load_cases(data_dir/f'{split}.jsonl')
        if limit is not None: cases=cases[:limit]
        for j,case in enumerate(cases):
            snap=case.snapshot
            minimum=min((len(x['indices']) for x in case.labels if x['feasible']),default=None)
            truth=case.scene.truth[snap.time]
            target=np.eye(len(case.scene.rooms))[truth]
            for mode in ('provenance','naive','last_report'):
                b=snap.belief('courier',mode)
                belief_rows.append({'scene_id':case.scene.id,'split':split,'family':case.scene.family,'mode':mode,
                    'brier':float(np.square(b-target).sum()),'true_room_probability':float(b[truth]),
                    'chosen_room':snap.plan('courier',mode).destination})
            methods=[('uninformed',-1,None),('heuristic',-1,None)]+[(kind,seed,models[(kind,seed)]) for kind in ('graph','mlp','graph_no_provenance') for seed in seeds]
            for method,seed,model in methods:
                result=search(snap,case.problem,method=method,ranker=model,trace=False)
                found=None if result.patch is None else len(result.patch)
                correct=(minimum==found and (result.status=='no_solution_within_bounds' if minimum is None else result.status=='verified_minimum_within_bounds'))
                row={'scene_id':case.scene.id,'split':split,'family':case.scene.family,'rooms':len(case.scene.rooms),
                     'method':method,'model_seed':seed,'oracle_minimum':minimum,'found_size':found,
                     'minimum_gap':None if minimum is None or found is None else found-minimum,
                     'status':result.status,'correct':int(correct),'verifier_calls':result.verifier_calls,
                     'candidate_space':result.candidate_space_size,'latency_ms':result.elapsed_seconds*1000,
                     'ranking_ms':result.ranking_seconds*1000,
                     # Cardinality-first prefixes are identical to a verifier-call
                     # budget run. These are not wall-clock deadline metrics.
                     'success_at_4':int(found is not None and result.verifier_calls<=4),
                     'success_at_16':int(found is not None and result.verifier_calls<=16)}
                rows.append(row)
                if not correct: raise AssertionError(f'Exactness regression: {row}')
            if (j+1)%12==0: print(f'{split}: {j+1}/{len(cases)} scenes evaluated',flush=True)
    summary=summarize(rows)
    belief_summary=[]
    for split in ('test','ood'):
        for mode in ('provenance','naive','last_report'):
            group=[r for r in belief_rows if r['split']==split and r['mode']==mode]
            belief_summary.append({'split':split,'mode':mode,'brier':_scene_summary(group,'brier')})
    paired=[]
    for split in ('test','ood'):
        baseline={r['scene_id']:r for r in rows if r['split']==split and r['method']=='heuristic'}
        for method in ('uninformed','graph','mlp','graph_no_provenance'):
            group=[r for r in rows if r['split']==split and r['method']==method]
            by_scene=defaultdict(list)
            for row in group: by_scene[row['scene_id']].append(row)
            calls=[];latency=[]
            for sid,rs in by_scene.items():
                calls.append(np.mean([r['verifier_calls'] for r in rs])-baseline[sid]['verifier_calls'])
                latency.append(np.mean([r['latency_ms'] for r in rs])-baseline[sid]['latency_ms'])
            paired.append({'split':split,'method':method,'reference':'heuristic',
                           'calls_difference':bootstrap_mean(calls),'latency_ms_difference':bootstrap_mean(latency)})
    try:
        import resource
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_mb=float(peak)/(1024**2 if sys.platform=='darwin' else 1024)
    except ImportError:peak_mb=None
    report={'memory':{'peak_process_rss_mb':peak_mb,
                      'scope':'whole evaluation process including all loaded checkpoints; not a per-method measurement'},
        'schema_version':1,'protocol':{'split_unit':'whole scene','training_seeds':list(seeds),
        'confidence_intervals':'95% nonparametric bootstrap over scene means; seeds averaged within scene',
        'latency':'steady-state CPU search, includes graph construction and ranking, excludes model loading and scene creation',
        'budget':'verifier-call prefixes, not measured deadline success','minimum_scope':'recorded candidate universe; max patch size 3',
        'limit':limit},'environment':{'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
        'processor':platform.processor(),'cpu_threads':1},'checkpoint_hashes':checkpoint_hashes,
        'dataset_manifest_sha256':hashlib.sha256((data_dir/'manifest.json').read_bytes()).hexdigest(),
        'elapsed_seconds':time.perf_counter()-start,'summary':summary,'belief_summary':belief_summary,'paired_comparisons':paired,
        'all_searches_correct':all(r['correct'] for r in rows),'searches':len(rows)}
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n', encoding="utf-8")
    for name,data in (('search_results',rows),('belief_results',belief_rows)):
        with (out/f'{name}.csv').open('w',newline='', encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    write_report(report,out/'RESULTS.md')
    return report


def write_report(report,path):
    lines=['# Measured benchmark results','',
           'These are measurements from the bundled finite synthetic benchmark, not claims about arbitrary games.',
           f"The statistical unit is an independent scene. {len(report['protocol']['training_seeds'])} trained seed(s) are averaged within scene before bootstrap confidence intervals.",
           '', '| Split | Method | Calls, mean | Latency, mean (ms) | Solvable goals found ≤4 calls | Exactness |',
           '|---|---|---:|---:|---:|---:|']
    for x in report['summary']:
        s=x['success_at_4']['mean'];pct='n/a' if s is None else f'{s:.1%}'
        lines.append(f"| {x['split']} | {x['method']} | {x['verifier_calls']['mean']:.2f} | {x['latency_ms']['mean']:.2f} | {pct} | {x['correctness']['mean']:.1%} |")
    lines+=['','## Interpretation','','A smaller verifier-call count does not establish a speedup. Neural encoding and scoring are included in request latency. No-solution certificates still require exhausting the candidate universe.',
            '', 'The released benchmark is deliberately small. Out-of-distribution (OOD) cases change room count, graph density, source reliability, and mobility. This is a bundled distribution shift, not a controlled estimate of each factor separately.',
            '', 'Full confidence intervals, paired differences against the heuristic, individual scene rows, manifest hashes, and checkpoint hashes are in the JSON and CSV files.',
            '', '## Belief calibration proxy','','| Split | Belief method | Brier score, mean |','|---|---|---:|']
    for x in report['belief_summary']: lines.append(f"| {x['split']} | {x['mode']} | {x['brier']['mean']:.5f} |")
    lines+=['','Belief scores use the same received histories at the initial search checkpoint. They do not compare divergent rollouts or measure human believability.','']
    Path(path).write_text('\n'.join(lines), encoding="utf-8")

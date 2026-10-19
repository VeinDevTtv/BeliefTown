"""Rebuild the released synthetic dataset and all nine CPU checkpoints."""
from pathlib import Path
import json
import time
from belieftown.learning.data import make_dataset
from belieftown.learning.train import train_model
root=Path(__file__).resolve().parents[1]
start=time.perf_counter()
print('Generating scene-held-out data',flush=True)
m=make_dataset(root/'benchmarks/data',train=120,validation=24,test=48,ood=24)
print(json.dumps(m,indent=2),flush=True)
for kind in ('graph','mlp','graph_no_provenance'):
    for seed in (0,1,2):
        print(f'Training {kind} seed={seed}',flush=True)
        p=train_model(root/'benchmarks/data',root/f'artifacts/checkpoints/{kind}_seed{seed}.pt',kind=kind,seed=seed,epochs=14)
        print(json.dumps({'kind':kind,'seed':seed,'validation_bce':p.metadata['validation_bce'],'training_seconds':p.metadata['training_seconds']}),flush=True)
print('COMPLETE',time.perf_counter()-start,flush=True)

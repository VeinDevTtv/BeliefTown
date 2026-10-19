from pathlib import Path
from belieftown.benchmark import run_benchmark
root=Path(__file__).resolve().parents[1]
r=run_benchmark(root/'benchmarks/data',root/'artifacts/checkpoints',root/'artifacts/evaluation')
print('COMPLETE',r['searches'],r['all_searches_correct'],flush=True)

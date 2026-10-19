"""Execute every notebook with errors disallowed, saving outputs and a manifest."""
from __future__ import annotations
import json
import os
from pathlib import Path
import time


def execute_notebooks(directory: str|Path,output_dir: str|Path,*,timeout: int=240,run_root: str|Path|None=None) -> dict:
    import nbformat
    from nbclient import NotebookClient
    directory=Path(directory).resolve();out=Path(output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    paths=sorted(directory.glob('*.ipynb'))
    if not paths:raise ValueError(f'No notebooks found in {directory}')
    environment=dict(os.environ)
    if run_root is not None:environment['BELIEFTOWN_RUN_ROOT']=str(Path(run_root).resolve())
    else:environment.pop('BELIEFTOWN_RUN_ROOT',None)
    report={'allow_errors':False,'run_root':environment.get('BELIEFTOWN_RUN_ROOT'),'notebooks':[]}
    for path in paths:
        start=time.perf_counter();nb=nbformat.read(path,as_version=4)
        client=NotebookClient(nb,timeout=timeout,kernel_name='python3',allow_errors=False,resources={'metadata':{'path':str(directory)}})
        try:client.execute(env=environment)
        finally:nbformat.write(nb,out/path.name)
        errors=[o for c in nb.cells if c.cell_type=='code' for o in c.get('outputs',[]) if o.get('output_type')=='error']
        if errors:raise RuntimeError(f'Notebook contains error outputs: {path.name}')
        report['notebooks'].append({'file':path.name,'code_cells':sum(c.cell_type=='code' for c in nb.cells),
            'errors':0,'execution_seconds':time.perf_counter()-start})
        print(f'Executed {path.name}',flush=True)
    (out/'execution_report.json').write_text(json.dumps(report,indent=2)+'\n', encoding="utf-8")
    return report

"""Install the wheel in a separate venv and exercise its packaged resources.

Uses --system-site-packages to reuse installed third-party dependencies offline.
The assertion below proves BeliefTown itself is imported from the wheel, not the
editable source installation. This is not a clean network dependency-resolution
or a test on a different Python/OS.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import venv

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=ROOT/'artifacts/wheel_smoke.json')
args=parser.parse_args()
wheel=sorted((ROOT/'dist').glob('belieftown-*.whl'))[-1]
report={'wheel':wheel.name,'dependency_mode':'offline venv with system-site-packages; project installed from wheel','checks':[]}
with tempfile.TemporaryDirectory(prefix='belieftown-wheel-') as temp:
    base=Path(temp);environment=base/'env';venv.EnvBuilder(with_pip=True,system_site_packages=True).create(environment)
    python=environment/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
    def run(args):
        result=subprocess.run([str(python),*args],cwd=base,capture_output=True,text=True,check=True)
        return result.stdout
    run(['-m','pip','install','--no-deps','--ignore-installed',str(wheel)])
    imported=run(['-c','import belieftown; print(belieftown.__file__)']).strip()
    assert str(environment) in imported and str(ROOT/'src') not in imported,imported
    report['import_path']=imported;report['checks'].append('package imported from installed wheel')
    doctor=json.loads(run(['-m','belieftown','doctor']))
    assert doctor['canonical_scene']=='verified_minimum_within_bounds'
    report['checks'].append('packaged canonical scene and core dependencies')
    audit=base/'audit.json';replay=base/'replay.json';html=base/'showcase.html'
    run(['-m','belieftown','solve','--output',str(audit)])
    checked=json.loads(run(['-m','belieftown','verify',str(audit)]))
    assert checked['valid'] and checked['minimum']
    report['checks'].append('CLI solve and independently reverified minimum')
    run(['-m','belieftown','replay',str(audit),'--output',str(replay)])
    assert json.loads(replay.read_text(encoding="utf-8"))['frames'][4]['actors'][0]['destination']==1
    report['checks'].append('CLI patch replay changes destination')
    run(['-m','belieftown','export','--output',str(html)])
    assert html.stat().st_size>100000
    report['checks'].append('all packaged browser assets and recorded export')
report['passed']=len(report['checks'])
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(report,indent=2)+'\n', encoding="utf-8")
print(json.dumps(report,indent=2))

"""Record the installed dependency closure for this Linux/Python build.

The lock describes an observed environment; pyproject.toml defines portable
supported ranges. Run this only in the environment whose versions are intended.
"""
from __future__ import annotations
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT=Path(__file__).resolve().parents[1]
roots=['numpy','torch','jupyterlab','nbformat','nbclient','matplotlib','setuptools','pip']
queue=[(name,()) for name in roots];seen=set();versions={};missing=[]
while queue:
    name,extras=queue.pop();key=(canonicalize_name(name),tuple(sorted(extras)))
    if key in seen:continue
    seen.add(key)
    try:dist=metadata.distribution(name)
    except metadata.PackageNotFoundError:missing.append(name);continue
    versions[canonicalize_name(dist.metadata['Name'])]=dist.version
    for text in dist.requires or []:
        req=Requirement(text)
        if req.marker and not any(req.marker.evaluate({'extra':extra}) for extra in ('',*extras)):continue
        queue.append((req.name,tuple(req.extras)))
if missing:raise RuntimeError(f'Missing dependency metadata: {missing}')
header='''# Observed Linux / Python 3.13 CPU dependency closure for BeliefTown 1.0.0.
# This is a build-environment snapshot, not a cross-platform universal lock.
# Use pyproject.toml extras for the normal installation on another platform.
# Initial dependency installation requires network/package-cache access.
# The extra index provides the exact PyTorch CPU build recorded below.
--extra-index-url https://download.pytorch.org/whl/cpu

'''
(ROOT/'requirements-lock.txt').write_text(header+'\n'.join(f'{name}=={version}' for name,version in sorted(versions.items()))+'\n', encoding="utf-8")
report={'python':sys.version,'platform':platform.platform(),'machine':platform.machine(),
        'dependency_snapshot_scope':'installed transitive closure for the declared all/build components',
        'versions':versions,'root_components':roots,
        'limitations':'No fresh network resolution or cross-platform installation was executed.'}
try:
    report['cpu_model']=next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text(encoding="utf-8").splitlines() if line.startswith('model name'))
except (OSError,StopIteration):report['cpu_model']=platform.processor()
(ROOT/'artifacts/environment.json').write_text(json.dumps(report,indent=2)+'\n', encoding="utf-8")
print(f'Recorded {len(versions)} installed distributions')

"""Build a complete source archive and checksums without bundling environments.

Run the verification gates first. A wheel is included for the core application;
the repository ZIP is the complete research release with data and checkpoints.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
DEST=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else ROOT.parent
DEST.mkdir(parents=True,exist_ok=True)
report=json.loads((ROOT/'artifacts/release_verification.json').read_text(encoding="utf-8"))
if report['unit_tests']<127 or report['skipped_tests'] or report['held_out_searches']!=792:
    raise RuntimeError('The complete release verification record is required')
subprocess.run([sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation','.',
                '--wheel-dir','dist'],cwd=ROOT,check=True)
# Keep research artifacts but omit transient work directories and redundant logs.
excluded_dirs={'__pycache__','.pytest_cache','.git','.venv','build','.ipynb_checkpoints','local_validation','.ruff_cache','.mypy_cache'}
excluded_names={'.coverage','pytest_cli.log','pytest_initial.log','pytest_ui.log','SHA256SUMS.txt'}
def include(p:Path)->bool:
    rel=p.relative_to(ROOT)
    if any(part in excluded_dirs or part.endswith('.egg-info') for part in rel.parts):return False
    if p.name in excluded_names or p.suffix in {'.pyc','.pyo','.pid'}:return False
    if 'dist' in rel.parts and p.suffix!='.whl':return False
    if p.suffix.lower() in {'.ttf','.otf','.woff','.woff2'}:
        raise RuntimeError(f'Font files must not be distributed: {p}')
    return True
paths=sorted(p for p in ROOT.rglob('*') if p.is_file() and include(p))
# Source hashes make modifications visible, not cryptographically trustworthy.
lines=[f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}' for p in paths]
(ROOT/'SHA256SUMS.txt').write_text('\n'.join(lines)+'\n', encoding="utf-8")
paths.append(ROOT/'SHA256SUMS.txt')
archive=DEST/'BeliefTown_v1.0.0.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in paths:z.write(p,Path('belieftown')/p.relative_to(ROOT))
with zipfile.ZipFile(archive) as z:
    if z.testzip() is not None:raise RuntimeError('Archive CRC verification failed')
    if any(Path(n).is_absolute() or '..' in Path(n).parts for n in z.namelist()):raise RuntimeError('Unsafe archive entry')
standalone={
    'BeliefTown_Showcase.html':ROOT/'artifacts/BeliefTown_Showcase.html',
    'BeliefTown_Walkthrough.mp4':ROOT/'artifacts/walkthrough/BeliefTown_Walkthrough.mp4',
    'belieftown-1.0.0-py3-none-any.whl':ROOT/'dist/belieftown-1.0.0-py3-none-any.whl'}
for name,source in standalone.items():shutil.copy2(source,DEST/name)
outputs=[archive,*[DEST/n for n in standalone]]
(DEST/'BeliefTown_SHA256SUMS.txt').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n' for p in outputs), encoding="utf-8")
print(json.dumps({'archive':str(archive),'files':len(paths),'uncompressed_bytes':sum(p.stat().st_size for p in paths),
                  'outputs':{p.name:p.stat().st_size for p in outputs}},indent=2))

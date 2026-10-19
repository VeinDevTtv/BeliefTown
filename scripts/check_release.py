"""Validate cross-artifact consistency and write the factual verification report.

Run after tests, notebook execution, benchmark, browser checks and wheel smoke.
This does not manufacture results or run a missing stage on the user's behalf.
"""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import nbformat
from belieftown.audit import verify_bundle

ROOT=Path(__file__).resolve().parents[1]
checks=[]
def load(path):return json.loads((ROOT/path).read_text(encoding="utf-8"))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def check(condition,label):
    if not condition:raise AssertionError(label)
    checks.append(label)

suites=ET.parse(ROOT/'artifacts/junit.xml').getroot()
unit=sum(int(s.get('tests',0)) for s in suites.findall('testsuite'))
failures=sum(int(s.get('failures',0))+int(s.get('errors',0)) for s in suites.findall('testsuite'))
skipped=sum(int(s.get('skipped',0)) for s in suites.findall('testsuite'))
check(unit>=127 and failures==0 and skipped==0,'All collected release tests passed without skips')
coverage=load('artifacts/coverage.json')['totals']
check(coverage['percent_covered']>=90,'Measured test-suite statement coverage exceeds 90 percent')
nb_report=load('artifacts/executed_notebooks/execution_report.json')
check(len(nb_report['notebooks'])==4 and all(n['errors']==0 for n in nb_report['notebooks']),'All four release notebooks executed with errors disallowed')
code_cells=0;pngs=0
for p in sorted((ROOT/'artifacts/executed_notebooks').glob('*.ipynb')):
    nb=nbformat.read(p,as_version=4)
    for cell in nb.cells:
        if cell.cell_type!='code':continue
        code_cells+=1
        check(cell.execution_count is not None,f'{p.name}: code cell {code_cells} has an execution count')
        for output in cell.get('outputs',[]):
            if output.get('output_type')=='error':raise AssertionError(f'Error output in {p}')
            pngs+=int('image/png' in output.get('data',{}))
check(pngs>=5,'Executed notebooks include real inline PNG figures')
manifest=load('benchmarks/data/manifest.json')
check(manifest['scene_hash_count']==216,'Released dataset has 216 distinct scene hashes')
for split,entry in manifest['splits'].items():
    check(sha(ROOT/f'benchmarks/data/{split}.jsonl')==entry['sha256'],f'{split} data hash matches its manifest')
report=load('artifacts/evaluation/results.json')
check(report['searches']==792 and report['all_searches_correct'],'All 792 held-out search outcomes match the exhaustive labels')
check(sha(ROOT/'benchmarks/data/manifest.json')==report['dataset_manifest_sha256'],'Evaluation identifies the delivered dataset manifest')
check(len(report['checkpoint_hashes'])==9,'Evaluation includes all nine release checkpoints')
for name,expected in report['checkpoint_hashes'].items():
    check(sha(ROOT/f'artifacts/checkpoints/{name}.pt')==expected,f'{name} checkpoint hash matches measured evaluation')
    metadata=load(f'artifacts/checkpoints/{name}.json')
    check(metadata['epochs']==14 and len(metadata['history'])==14 and metadata['manifest_sha256']==report['dataset_manifest_sha256'],f'{name} has complete training history and matching data provenance')
check(verify_bundle(load('artifacts/example_audit.json'))['minimum'],'Bundled canonical audit independently recertifies minimum cardinality')
browser=load('artifacts/browser/checks.json')
check(browser['passed']==10 and not browser['browser_errors'],'Ten browser-service/layout/offline checks passed with no JavaScript errors')
wheel=load('artifacts/wheel_smoke.json')
check(wheel['passed']==5,'Five separate-environment wheel installation/resource checks passed')
quick=load('artifacts/quick_reproduction/evaluation/results.json')
quicknb=load('artifacts/quick_reproduction/executed_notebooks/execution_report.json')
check(quick['all_searches_correct'] and quick['searches']==90,'Quick reproduction generated, trained, and evaluated its own run')
check(Path(quicknb['run_root'])==(ROOT/'artifacts/quick_reproduction').resolve() and len(quicknb['notebooks'])==4,'Reproduction notebooks used the rebuilt artifacts rather than bundled results')
walk=load('artifacts/walkthrough/manifest.json')
check(walk['duration_seconds']==60 and not walk['browser_errors'] and (ROOT/'artifacts/walkthrough/BeliefTown_Walkthrough.mp4').stat().st_size>10000,'Captioned 60-second walkthrough contains real application captures')
for p in (ROOT/'src').rglob('*.py'):ast.parse(p.read_text(encoding="utf-8"),filename=str(p))
check(True,'Every package source file parses as Python')

# Write this before checking documentation links, because README links to it.
summary={'version':'1.0.0','unit_tests':unit,'skipped_tests':skipped,
         'unit_test_statement_coverage_percent':coverage['percent_covered'],
         'executed_notebooks':4,'executed_code_cells':code_cells,'inline_png_outputs':pngs,
         'browser_checks':browser['passed'],'browser_transport':browser['transport'],
         'wheel_checks':wheel['passed'],'held_out_searches':report['searches'],
         'release_checkpoints':9,'dataset_scenes':216,
         'candidate_labels':sum(s['candidates'] for s in manifest['splits'].values()),
         'quick_reproduction_searches':quick['searches'],
         'python':load('artifacts/environment.json')['python'],
         'claim':'Executed release gates, not a proof of universal correctness.'}
text=f'''# Release verification: BeliefTown 1.0.0

This report describes actual local execution. It does not claim 100% code coverage, universal bug-freedom, remote hosting, or performance outside the declared finite model.

## Executed gates

| Gate | Observed result | Evidence |
|---|---|---|
| Automated correctness suite | **{unit} passed; 0 failed; {skipped} skipped** | `artifacts/final_tests.log`, `junit.xml` |
| Statement coverage in that suite | **{coverage['percent_covered']:.2f}%** | `artifacts/coverage.json` |
| Research notebook execution | **4 notebooks, {code_cells} code cells, zero cell errors** | `artifacts/executed_notebooks/` |
| Saved notebook visual outputs | **{pngs} inline PNG figures** | Executed notebook files |
| Browser application checks | **10 passed, zero JavaScript exceptions** | `artifacts/browser/checks.json`, screenshots |
| Packaged installation | **5 wheel/CLI/resource checks passed in a separate venv** | `artifacts/wheel_smoke.json` |
| Release training | **9 real CPU checkpoints; 14 epochs each; 3 seeds per model type** | Checkpoints and complete epoch histories |
| Held-out search benchmark | **792 outcomes agree with exhaustive feasibility/minimum labels** | Raw CSV, results JSON and report |
| Synthetic data integrity | **216 scenes; 13,001 labeled candidate patches** | Hash-checked whole-scene manifests |
| Independent audit recheck | **Canonical one-delivery patch recertifies minimum** | `artifacts/example_audit.json` |
| Single-command quick pipeline | **Own data, 3 trained models, 90 correct search outcomes, 4 notebooks against the rebuilt run** | `artifacts/quick_reproduction/`, `quick_pipeline.log` |
| Launch walkthrough | **60-second captioned capture sequence; no audio** | MP4, original UI captures, capture manifest |

The ordinary test suite's coverage does not include every line exercised in separate full benchmark, browser, or notebook subprocesses. Its percentage is reported as measured, not padded with those other executions.

## What was tested mathematically

Tests include filtering versus independent latent-trajectory enumeration, stale evidence at original timestamps, duplicate-root idempotence, conflicting tiny likelihoods, illegal observations and messages, private-input invariance, shortest paths and exhaustive route choices, alternate communication paths under suppression, protected information, simultaneous-patch legality, bounded/no-solution/budget statuses, and maliciously altered audit records. Learning tests cover actual optimization, checkpoint reload, custom architecture preservation, node-permutation invariance, missing-truth features, split separation, and a three-model benchmark.

## Important execution boundaries

**Browser transport:** this build environment's browser rejected loopback navigation with an administrator policy. The browser checks therefore used a documented in-process binding to the real Python Studio service, not a mocked result table. Separate urllib tests exercised the actual loopback HTTP endpoints, token enforcement, host/origin restrictions, and responses. A fully navigated HTTP browser session on every user machine is not claimed.

**Installation:** the wheel was installed into a separate virtual environment and its import path was asserted to point to that environment, not the editable source. Third-party dependencies were reused with `--system-site-packages` because network installation was unavailable. This verifies the project wheel and packaged assets, not a fresh resolution/download of all dependencies.

**Platforms:** local gates ran on Linux with Python 3.13.5 and CPU PyTorch 2.10.0+cpu. The complete environment snapshot and pinned observed dependency closure are included. The package declares Python 3.10+; the supplied multi-version GitHub Actions workflow is not evidence that remote CI or Windows/macOS execution has occurred.

**Notebook hosting:** the local kernel-backed server is live only while its owner runs. Saved outputs and recorded HTML do not keep a server permanently hosted. Hosted-notebook port forwarding and public deployment are not configured.

**Research scope:** no human study, commercial-engine integration, psychological validation, or world-first novelty claim is included. The neural ranker did not establish a wall-clock speed advantage over the heuristic on the released small synthetic benchmark. Minimum certificates apply only to their recorded candidate spaces and cardinality bounds.

## Corrections made during integration

The release checks exposed and corrected the source-install subprocess entry point, stricter timestamp/index contracts, fabricated audit verification fields, custom checkpoint architecture preservation, one-seed report labeling, and rebuilt-notebook data-path selection. These corrections are covered by tests or end-to-end execution; they are not deferred TODO items.

## Reproduce the gates

```bash
python -m pip install -e ".[all]"
python -m pytest -q --cov=belieftown --cov-report=json:artifacts/coverage.json --junitxml=artifacts/junit.xml
python -m belieftown notebooks
python -m belieftown reproduce --profile quick --output artifacts/quick_reproduction --notebooks
python -m belieftown verify artifacts/example_audit.json
```

Optional browser/media rebuilding additionally needs the `browser` extra, Chromium, and ffmpeg for MP4 encoding. Build a wheel with `python -m pip wheel --no-deps . --wheel-dir dist`, then run `python scripts/wheel_smoke.py`. Re-running timing experiments can change timing measurements; retain the environment and protocol when comparing results.
'''
(ROOT/'docs/VERIFICATION.md').write_text(text, encoding="utf-8")
required=['docs/MATH.md','docs/ARCHITECTURE.md','docs/FEATURES.md','docs/MODEL_CARD.md',
          'docs/DATA_CARD.md','docs/NOTEBOOKS.md','docs/START_HERE.md','docs/BLOG.md',
          'docs/LINKEDIN_POST.md','docs/REFERENCES.md','docs/VERIFICATION.md','LICENSE',
          'README.md','CITATION.cff','requirements-lock.txt','.github/workflows/ci.yml',
          'artifacts/BeliefTown_Showcase.html']
for name in required:check((ROOT/name).is_file() and (ROOT/name).stat().st_size>0,f'Required release file exists: {name}')
broken=[]
for path in [ROOT/'README.md',*(ROOT/'docs').glob('*.md')]:
    for target in re.findall(r'!?\[[^\]]*\]\(([^\s)]+)',path.read_text(encoding="utf-8")):
        if '://' in target or target.startswith('#'):continue
        if not (path.parent/target.split('#')[0]).exists():broken.append((str(path.relative_to(ROOT)),target))
check(not broken,f'Documentation local links resolve: {broken}')
summary['checks']=checks;summary['passed_release_checks']=len(checks)
(ROOT/'artifacts/release_verification.json').write_text(json.dumps(summary,indent=2)+'\n', encoding="utf-8")
print(json.dumps({k:v for k,v in summary.items() if k!='checks'},indent=2))

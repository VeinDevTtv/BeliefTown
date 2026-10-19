"""Create a captioned 60-second MP4 from actual application states.

Requirements: the optional browser extra, Chromium, and ffmpeg on PATH.
This is a sequence of real UI captures, not a claim of continuous real-time
recording. A Playwright binding invokes the same Python Studio service directly.
No private browser navigation policy is bypassed and no external assets are used.
"""
from pathlib import Path
from importlib.resources import files
import json
import os
import shutil
import subprocess
from playwright.sync_api import sync_playwright
from belieftown.studio import Studio

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/walkthrough';OUT.mkdir(parents=True,exist_ok=True)
service=Studio(checkpoint_dir=ROOT/'artifacts/checkpoints')
frames=[];errors=[]
with sync_playwright() as p:
    exe=os.environ.get('CHROMIUM_PATH') or shutil.which('chromium') or shutil.which('google-chrome')
    kwargs={'headless':True,'args':['--no-sandbox','--disable-dev-shm-usage']}
    if exe:kwargs['executable_path']=exe
    browser=p.chromium.launch(**kwargs)
    page=browser.new_page(viewport={'width':1440,'height':1080},device_scale_factor=1)
    page.on('pageerror',lambda error:errors.append(str(error)))
    def binding(action,args):
        try:return {'ok':True,'value':service.dispatch(action,args)}
        except (ValueError,TypeError,KeyError) as e:return {'ok':False,'value':{'error':str(e)}}
    page.expose_function('__belieftown_test_api',binding)
    static=files('belieftown').joinpath('static')
    html=static.joinpath('index.html').read_text(encoding="utf-8").replace('__BOOTSTRAP__',json.dumps({'offline':False,'token':'local-service-binding'}))
    html=html.replace('<link rel="stylesheet" href="/styles.css">','<style>'+static.joinpath('styles.css').read_text(encoding="utf-8")+'</style>')
    shim="window.fetch=async function(url,opts={}){const r=await window.__belieftown_test_api(url.split('/').pop(),JSON.parse(opts.body||'{}'));return {ok:r.ok,json:async()=>r.value};};"
    html=html.replace('<script src="/app.js"></script>','<script>'+shim+static.joinpath('app.js').read_text(encoding="utf-8")+'</script>')
    page.set_content(html);page.wait_for_selector('.actor-card')
    page.evaluate('''() => {const b=document.createElement('div'); b.id='walkthrough-caption';
    b.style.cssText='position:fixed;left:28px;right:28px;bottom:22px;z-index:999;background:#171717f5;border:1px solid #777;border-radius:10px;padding:18px 25px;color:#fff;font:600 24px/1.35 system-ui;box-shadow:0 8px 36px #0008;pointer-events:none';document.body.appendChild(b);}''')
    def shot(name,caption,duration,y=400):
        page.evaluate('([text,y])=>{document.querySelector("#walkthrough-caption").textContent=text;window.scrollTo(0,y)}',[caption,y])
        page.wait_for_timeout(120)
        target=OUT/f'{name}.png';page.screenshot(path=str(target))
        frames.append({'file':target.name,'duration_seconds':duration,'caption':caption})
    shot('01_private_beliefs','Same world. Different evidence. The Courier chooses Workshop; the parcel is in Library.',7,400)
    page.locator('#find').click();page.wait_for_selector('#solution strong')
    assert 'Verified minimum' in page.locator('#solution').inner_text()
    shot('02_verified_patch','Ask for a different decision. The solver verifies that one existing observation is enough.',7,400)
    page.locator('#apply').click();page.wait_for_function("document.querySelector('#notice').textContent.includes('Patch applied')")
    assert 'Library' in page.locator('.actor-card').first.inner_text()
    shot('03_new_action','Mira forwards new-library. The physical history stays fixed; the Courier now chooses Library.',7,400)
    page.locator('#step').click();shot('04_replay_step','Matched replay: the two Couriers follow different evidence-driven routes.',4,400)
    page.locator('#step').click();shot('05_replay_result','The patched branch reaches the parcel through a different decision—not a forced action.',4,400)
    page.locator('#reset').click();page.wait_for_function("document.querySelector('#notice').textContent.includes('Branch reset')")
    page.locator('.advanced-tools').evaluate('(element)=>{element.open=true}')
    page.locator('#manual-root').select_option('old-workshop');page.locator('#deliver').click()
    page.wait_for_function("document.querySelector('#notice').textContent.includes('did not change')")
    shot('06_duplicate','Another copy is not another observation. The provenance-aware belief remains unchanged.',8,1290)
    page.locator('#reset').click();page.wait_for_function("document.querySelector('#notice').textContent.includes('Branch reset')")
    page.locator('#protect').check();page.locator('#find').click()
    page.wait_for_function("document.querySelector('#solution').textContent.includes('No solution within')")
    shot('07_no_solution','Protect the target from new information: no solution within the declared bounds.',8,400)
    page.locator('#protect').uncheck();page.locator('#method').select_option('graph');page.locator('#find').click()
    page.wait_for_function("document.querySelector('#solution').textContent.includes('Verified minimum')")
    shot('08_graph_ranker','A trained graph model may order candidates. Only exact execution verifies the result.',8,400)
    shot('09_open_source','BeliefTown \u00b7 MIT \u00b7 Python + Jupyter \u00b7 Source, tests, checkpoints, and honest benchmark results.',7,400)
    browser.close()
assert sum(x['duration_seconds'] for x in frames)==60
assert not errors,errors
manifest={'kind':'captioned UI-state walkthrough','duration_seconds':60,'audio':False,
          'transport':'actual Python Studio via in-process browser binding',
          'continuous_real_time_recording':False,'frames':frames,'browser_errors':errors}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n', encoding="utf-8")
concat=''.join(f"file '{x['file']}'\nduration {x['duration_seconds']}\n" for x in frames)+f"file '{frames[-1]['file']}'\n"
(OUT/'frames.txt').write_text(concat, encoding="utf-8")
ffmpeg=shutil.which('ffmpeg')
if not ffmpeg:raise RuntimeError('ffmpeg is required to encode the walkthrough')
subprocess.run([ffmpeg,'-y','-f','concat','-safe','0','-i','frames.txt','-t','60','-vf','fps=24,format=yuv420p',
                '-c:v','libx264','-preset','veryfast','-crf','23','-movflags','+faststart','BeliefTown_Walkthrough.mp4'],cwd=OUT,check=True)
print(OUT/'BeliefTown_Walkthrough.mp4')

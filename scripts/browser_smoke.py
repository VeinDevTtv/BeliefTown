"""Optional browser acceptance test. Requires Playwright and a Chromium browser."""
from pathlib import Path
import json
import shutil
import os
from playwright.sync_api import sync_playwright, Error as PlaywrightError
from importlib.resources import files
from belieftown.server import StudioServer
from belieftown.studio import Studio
from belieftown.export import export_showcase

root=Path(__file__).resolve().parents[1]
out=root/'artifacts/browser';out.mkdir(parents=True,exist_ok=True)
errors=[]
class Checks(list):
    def append(self,item):
        super().append(item);print('PASS',item,flush=True)
checks=Checks()
executable=os.environ.get('CHROMIUM_PATH') or shutil.which('chromium') or shutil.which('google-chrome')
with StudioServer(Studio(checkpoint_dir=root/'artifacts/checkpoints')) as server, sync_playwright() as p:
    kwargs={'headless':True,'args':['--no-sandbox','--disable-dev-shm-usage']}
    if executable:kwargs['executable_path']=executable
    browser=p.chromium.launch(**kwargs)
    page=browser.new_page(viewport={'width':1440,'height':1040},device_scale_factor=1)
    page.on('pageerror',lambda error:errors.append(str(error)))
    transport='loopback HTTP'
    try:
        page.goto(server.url)
    except PlaywrightError as error:
        if 'ERR_BLOCKED_BY_ADMINISTRATOR' not in str(error):raise
        # The build browser disallows loopback navigation. Exercise the actual
        # Python service through a Playwright binding instead; HTTP auth/origin
        # behavior is independently covered by test_studio_server.py.
        transport='in-process Python service binding; build-browser policy blocks loopback navigation'
        page.close();page=browser.new_page(viewport={'width':1440,'height':1040},device_scale_factor=1)
        page.on('pageerror',lambda error:(errors.append(str(error)),print('JS ERROR',str(error),flush=True)))
        page.on('console',lambda msg:print('CONSOLE',msg.type,msg.text,flush=True) if msg.type=='error' else None)
        def binding(action,args):
            try:return {'ok':True,'value':server.studio.dispatch(action,args)}
            except (ValueError,KeyError,TypeError) as e:return {'ok':False,'value':{'error':str(e)}}
        page.expose_function('__belieftown_test_api',binding)
        static=files('belieftown').joinpath('static')
        html=static.joinpath('index.html').read_text(encoding="utf-8").replace('__BOOTSTRAP__',json.dumps({'offline':False,'token':server.token}))
        html=html.replace('<link rel="stylesheet" href="/styles.css">','<style>'+static.joinpath('styles.css').read_text(encoding="utf-8")+'</style>')
        shim="window.fetch=async function(url,opts={}){const r=await window.__belieftown_test_api(url.split('/').pop(),JSON.parse(opts.body||'{}'));return {ok:r.ok,json:async()=>r.value};};"
        html=html.replace('<script src="/app.js"></script>','<script>'+shim+static.joinpath('app.js').read_text(encoding="utf-8")+'</script>')
        page.set_content(html)
    page.wait_for_selector('.actor-card')
    assert page.locator('.actor-card').count()==3;checks.append('live initial scene')
    page.locator('#find').click();page.wait_for_selector('#solution strong')
    assert 'Verified minimum' in page.locator('#solution').inner_text();checks.append('live minimum patch search')
    page.screenshot(path=str(out/'desktop.png'),full_page=True)
    page.locator('#apply').click();page.wait_for_function("document.querySelector('#notice').textContent.includes('Patch applied')")
    assert 'Library' in page.locator('.actor-card').first.inner_text();checks.append('apply changes destination')
    page.locator('#step').click();assert page.locator('#tick-label').inner_text()=='TURN 5';checks.append('step replay')
    page.locator('#reset').click();page.wait_for_function("document.querySelector('#notice').textContent.includes('Branch reset')")
    page.locator('.advanced-tools > summary').click()
    page.locator('#manual-root').select_option('old-workshop');page.locator('#deliver').click()
    page.wait_for_function("document.querySelector('#notice').textContent.includes('did not change')")
    checks.append('duplicate delivery leaves belief unchanged')
    page.locator('#reset').click();page.wait_for_function("document.querySelector('#notice').textContent.includes('Branch reset')")
    page.locator('#suppress-root').select_option('new-library');page.locator('#fork-root').click()
    page.wait_for_function("document.querySelector('#notice').textContent.includes('Observation suppressed')")
    checks.append('diagnostic observation fork')
    page.locator('#reset').click();page.wait_for_function("document.querySelector('#notice').textContent.includes('Branch reset')")
    page.locator('#protect').check();page.locator('#find').click()
    page.wait_for_function("document.querySelector('#solution').textContent.includes('No solution within')")
    checks.append('constrained no-solution status')
    page.locator('#protect').uncheck();page.locator('#method').select_option('graph');page.locator('#find').click()
    page.wait_for_function("document.querySelector('#solution').textContent.includes('Verified minimum')")
    checks.append('live learned ranker')
    page.set_viewport_size({'width':390,'height':844});page.screenshot(path=str(out/'mobile.png'),full_page=True)
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth+1');checks.append('mobile layout no horizontal overflow')
    offline=export_showcase(out/'offline.html')
    page.close();page=browser.new_page(viewport={'width':1440,'height':1040},device_scale_factor=1)
    page.on('pageerror',lambda error:(errors.append(str(error)),print('JS ERROR',str(error),flush=True)))
    page.set_content(offline.read_text(encoding="utf-8"));page.wait_for_selector('.actor-card')
    assert 'OFFLINE' in page.locator('#connection').inner_text()
    page.locator('#find').click();page.wait_for_selector('#solution strong');page.locator('#apply').click()
    assert 'Library' in page.locator('.actor-card').first.inner_text();checks.append('offline recorded patch replay')
    browser.close()
report={'checks':checks,'passed':len(checks),'browser_errors':errors,'transport':transport}
(out/'checks.json').write_text(json.dumps(report,indent=2)+'\n', encoding="utf-8")
print(json.dumps(report,indent=2))
assert not errors,errors

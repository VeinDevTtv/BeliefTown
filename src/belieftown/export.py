"""Self-contained recorded showcase. Explicitly not a browser-side Python solver."""
from __future__ import annotations
from importlib.resources import files
import json
from pathlib import Path
from .scenarios import FAMILIES,demo_scene
from .studio import Studio
from .world import Episode
from .audit import verify_bundle


def export_showcase(path: str|Path) -> Path:
    scenes={}
    for family in FAMILIES:
        studio=Studio(demo_scene(family));state=studio.state()
        featured=studio.find_patch({'time':studio.scene.search_start,'actor':'courier','room':1,'max_size':3,'method':'heuristic'})
        verify_bundle(featured['bundle'])
        if studio.last_result.patch is not None:
            after=studio.apply_solution()
            featured['replay']=after['branch']
        else: featured['replay']=state['branch']
        scenes[family]={'state':state,'featured':featured}
    boot={'offline':True,'defaultFamily':'missed_relocation','scenes':scenes}
    static=files('belieftown').joinpath('static')
    html=static.joinpath('index.html').read_text(encoding="utf-8")
    html=html.replace('<link rel="stylesheet" href="/styles.css">','<style>'+static.joinpath('styles.css').read_text(encoding="utf-8")+'</style>')
    html=html.replace('__BOOTSTRAP__',json.dumps(boot,separators=(',',':'),allow_nan=False).replace('<','\\u003c'))
    html=html.replace('<script src="/app.js"></script>','<script>'+static.joinpath('app.js').read_text(encoding="utf-8")+'</script>')
    out=Path(path)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as destination:
        destination.write(html)
    return out

import json
import urllib.request
import urllib.error
import pytest
from belieftown.studio import Studio
from belieftown.server import StudioServer
from belieftown.audit import verify_bundle
from belieftown.export import export_showcase


def test_studio_search_apply_export_reset():
    s=Studio();r=s.find_patch({'room':1,'time':4})
    assert verify_bundle(r['bundle'])['valid']
    after=s.apply_solution();assert after['branch'][4]['actors'][0]['destination']==1
    assert s.export()['episode']['patches']
    with pytest.raises(ValueError): s.apply_solution()
    s.reset();assert not s.episode.patches


def test_studio_manual_duplicate_and_fork():
    s=Studio();before=s.episode.at(4).belief('courier').copy()
    s.deliver({'time':4,'sender':'mira','recipient':'courier','root_id':'old-workshop'})
    assert (before==s.episode.at(4).belief('courier')).all()
    state=s.fork({'observation':'new-library'})
    assert not s.episode.patches and 'new-library' in state['suppressed_observations']


def test_policy_ablation_blocks_provenance_patch_claims():
    s=Studio();s.fork({'mode':'naive'})
    with pytest.raises(ValueError):s.find_patch({'room':1,'time':4})


def test_loopback_http_api_and_security():
    with StudioServer() as server:
        base=server.url
        assert 'BeliefTown' in urllib.request.urlopen(base).read().decode()
        with pytest.raises(urllib.error.HTTPError) as e:urllib.request.urlopen(base+'api/state')
        assert e.value.code==403
        request=urllib.request.Request(base+'api/search',data=json.dumps({'room':1,'time':4}).encode(),
            headers={'Content-Type':'application/json','X-BeliefTown-Token':server.token})
        response=json.load(urllib.request.urlopen(request))
        assert response['record']['status']=='verified_minimum_within_bounds'
        hostile=urllib.request.Request(base+'api/reset',data=b'{}',headers={
            'Content-Type':'application/json','X-BeliefTown-Token':server.token,'Origin':'https://untrusted.example'})
        with pytest.raises(urllib.error.HTTPError) as e:urllib.request.urlopen(hostile)
        assert e.value.code==403
        with pytest.raises(urllib.error.HTTPError):urllib.request.urlopen(base+'../../etc/passwd')


def test_standalone_export_is_self_contained(tmp_path):
    path=export_showcase(tmp_path/'showcase.html');text=path.read_text(encoding="utf-8")
    assert '__BOOTSTRAP__' not in text
    assert '<script src=' not in text and 'href="/styles.css"' not in text
    assert 'recorded' in text and 'verified_minimum_within_bounds' in text

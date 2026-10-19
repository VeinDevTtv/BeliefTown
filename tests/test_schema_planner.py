from dataclasses import replace
import itertools
import json
import numpy as np
import pytest
from belieftown.schema import Scene,SceneError,Message
from belieftown.scenarios import demo_scene
from belieftown.planner import plan_search,distances,route_matrix


def test_scene_json_round_trip(tmp_path):
    s=demo_scene(); path=tmp_path/'scene.json';s.save(path)
    loaded=Scene.load(path)
    assert loaded==s and loaded.hash==s.hash


@pytest.mark.parametrize('change',[
    {'horizon':-1}, {'rooms':('a','a')}, {'edges':()}, {'inspection_cost':0},
    {'inspection_cost':float('nan')}, {'schema_version':4}, {'truth':(0,)},
    {'transitions':()},
])
def test_invalid_scene_rejected(change):
    with pytest.raises((ValueError,TypeError)): replace(demo_scene(),**change)


def test_bad_parent_and_channel_rejected():
    s=demo_scene()
    with pytest.raises(SceneError): replace(s,messages=s.messages+(Message('bad','old-workshop','mira','courier',4),))
    with pytest.raises(SceneError): replace(s,channels=())


def test_cycles_rejected():
    s=demo_scene()
    a=Message('a','old-workshop','mira','theo',2,'b')
    b=Message('b','old-workshop','theo','mira',2,'a')
    with pytest.raises(SceneError): replace(s,messages=(a,b))


@pytest.mark.parametrize('seed',range(15))
def test_planner_matches_independent_enumeration(seed):
    rng=np.random.default_rng(seed);s=demo_scene(); b=rng.dirichlet(np.ones(4)); start=seed%4
    d,_=distances(4,s.edges); manual=[]
    for order in itertools.permutations(range(4)):
        cost=0.; weighted=0.;prev=start
        for x in order:
            cost+=d[prev,x]+1.;weighted+=b[x]*cost;prev=x
        manual.append((float(weighted),order))
    expected=min(manual,key=lambda x:(x[0],x[1])); result=plan_search(b,start,s.edges)
    assert result.order==expected[1]
    assert result.expected_cost==pytest.approx(expected[0])


def test_tie_break_is_deterministic():
    s=demo_scene(); b=np.array([.5,.5,0,0])
    orders=[plan_search(b,3,s.edges).order for _ in range(10)]
    assert len(set(orders))==1


def test_cached_route_data_is_read_only():
    s=demo_scene();_,m=route_matrix(4,s.edges,3,1.)
    with pytest.raises(ValueError): m[0,0]=0

@pytest.mark.parametrize('change',[{'seed':1.5},{'horizon':15.5},{'search_start':True}])
def test_integer_time_and_seed_contract(change):
    with pytest.raises(SceneError):replace(demo_scene(),**change)


def test_fractional_observation_timestamp_rejected():
    s=demo_scene();roots=(replace(s.observations[0],time=.5),)+s.observations[1:]
    with pytest.raises(SceneError):replace(s,observations=roots)


def test_declared_ids_cannot_collide_with_runtime_inspections():
    s=demo_scene();roots=(replace(s.observations[0],id='inspection:courier:5'),)+s.observations[1:]
    with pytest.raises(SceneError):replace(s,observations=roots)

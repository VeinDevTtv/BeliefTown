from dataclasses import replace
import numpy as np
import pytest
from belieftown.schema import Observation
from belieftown.beliefs import filter_belief,brute_force_belief,entropy,InconsistentEvidence


def test_independent_vs_copied_evidence():
    a=Observation('a','mira',0,1,.8)
    one=filter_belief([.25]*4,[],[a],0)
    duplicate=filter_belief([.25]*4,[],[a]*20,0)
    two=filter_belief([.25]*4,[],[a,replace(a,id='b')],0)
    assert one[1]==pytest.approx(.8)
    np.testing.assert_array_equal(one,duplicate)
    assert two[1]==pytest.approx(.64/(.64+3*(.2/3)**2))


@pytest.mark.parametrize('seed',range(20))
def test_filter_matches_independent_path_enumeration(seed):
    rng=np.random.default_rng(seed); n=3; t=3
    prior=rng.dirichlet(np.ones(n))
    ts=tuple(tuple(map(tuple,rng.dirichlet(np.ones(n),size=n).T)) for _ in range(t))
    roots=[Observation(f'r{k}','a',int(rng.integers(t+1)),int(rng.integers(n)),float(rng.uniform(.4,.95))) for k in range(4)]
    expected=brute_force_belief(prior,ts,roots,t)
    result=filter_belief(prior,ts,roots,t)
    np.testing.assert_allclose(result,expected,atol=1e-12)
    np.testing.assert_allclose(result,filter_belief(prior,ts,list(reversed(roots)),t),atol=1e-12)


def test_delayed_evidence_uses_observation_time():
    swap=(((0.,1.),(1.,0.)),)
    old=Observation('old','a',0,0,.9)
    recent=replace(old,id='recent',time=1)
    assert filter_belief([.5,.5],swap,[old],1)[1]==pytest.approx(.9)
    assert filter_belief([.5,.5],swap,[recent],1)[0]==pytest.approx(.9)


def test_log_domain_recovers_from_extreme_conflicting_evidence():
    roots=[Observation(f'a{i:03}','a',0,0,.999999) for i in range(200)]
    roots += [Observation(f'z{i:03}','b',0,1,.999999) for i in range(200)]
    np.testing.assert_allclose(filter_belief([.5,.5],[],roots,0),[.5,.5],atol=1e-10)


def test_contradictions_fail_explicitly():
    roots=[Observation('a','a',0,0,1.),Observation('b','b',0,1,1.)]
    with pytest.raises(InconsistentEvidence): filter_belief([.5,.5],[],roots,0)


def test_conflicting_root_id_is_rejected():
    root=Observation('r','a',0,0,.8)
    with pytest.raises(ValueError): filter_belief([.5,.5],[],[root,replace(root,reported_room=1)],0)


def test_future_observation_and_invalid_transition_are_rejected():
    with pytest.raises(ValueError): filter_belief([.5,.5],[],[Observation('r','a',1,0,.8)],0)
    with pytest.raises(ValueError): filter_belief([.5,.5],[[[1,1],[1,1]]],[],1)


def test_entropy_bounds():
    assert entropy(np.array([1.,0.]))==0.
    assert entropy(np.ones(4)/4)==pytest.approx(np.log(4))

from dataclasses import replace
import copy
import itertools
import numpy as np
import pytest
from belieftown.schema import Actor,Observation,Scene
from belieftown.scenarios import demo_scene,procedural_scene,FAMILIES,matrices
from belieftown.world import Episode
from belieftown.interventions import Delivery,PatchProblem,candidate_deliveries,apply_patch,verify_patch,search
from belieftown.verifier import check_audit
from belieftown.replay import fork


@pytest.fixture
def snap():
    s=demo_scene();return Episode(s).at(s.search_start)


def test_demo_patch_is_verified_minimum(snap):
    p=PatchProblem('courier',1);r=search(snap,p)
    assert r.status=='verified_minimum_within_bounds' and len(r.patch)==1
    assert r.patch[0].root_id=='new-library'
    assert check_audit(snap,r.to_dict())['minimum']
    e=fork(Episode(snap.scene),patch=r.patch,time=snap.time)
    assert e.at(snap.time).plan('courier').destination==1
    assert e.scene.truth==snap.scene.truth


def test_duplicate_root_does_not_change_belief(snap):
    d=Delivery('mira','courier','old-workshop')
    after=apply_patch(snap,[d])
    np.testing.assert_array_equal(snap.belief('courier'),after.belief('courier'))
    assert len(after.receipts['courier'])==len(snap.receipts['courier'])+1


def test_batch_cannot_relay_newly_received_root(snap):
    with pytest.raises(ValueError):
        apply_patch(snap,[Delivery('mira','theo','new-library'),Delivery('theo','courier','new-library')])


def test_patch_is_transactional_and_does_not_mutate_original(snap):
    before=snap.checkpoint_hash
    with pytest.raises(ValueError):
        apply_patch(snap,[Delivery('mira','courier','new-library'),Delivery('theo','courier','new-library')])
    assert snap.checkpoint_hash==before


def test_constraints_and_no_solution(snap):
    p=PatchProblem('courier',1,3,('courier',))
    r=search(snap,p)
    assert r.status=='no_solution_within_bounds'
    assert check_audit(snap,r.to_dict())['status']==r.status


def test_empty_patch_and_empty_universe(snap):
    current=snap.plan('courier').destination
    r=search(snap,PatchProblem('courier',current,candidate_limit=0))
    assert r.patch==() and r.status=='verified_minimum_within_bounds'
    r=search(snap,PatchProblem('courier',1,candidate_limit=0))
    assert r.status=='no_solution_within_bounds'


@pytest.mark.parametrize('budget',[0,1])
def test_budget_exhaustion_is_not_no_solution(snap,budget):
    r=search(snap,PatchProblem('courier',1),budget=budget)
    assert r.status=='budget_exhausted' and r.patch is None
    assert r.verifier_calls==budget
    assert check_audit(snap,r.to_dict())['valid']


@pytest.mark.parametrize('seed',range(12))
def test_search_matches_full_enumeration(seed):
    s=procedural_scene(seed,FAMILIES[seed%6]);snap=Episode(s).at(s.search_start)
    p=PatchProblem('courier',seed%4,max_size=2)
    c=candidate_deliveries(snap,p);optimum=None
    for k in range(3):
        if any(verify_patch(snap,p,patch).feasible for patch in itertools.combinations(c,k)):
            optimum=k;break
    for method in ('uninformed','heuristic'):
        r=search(snap,p,method=method)
        assert (None if r.patch is None else len(r.patch))==optimum
        if r.patch is not None: assert check_audit(snap,r.to_dict())['minimum']


def test_two_delivery_patch_is_really_needed():
    actors=(Actor('courier','Courier',2,(.72,.27,.01)),Actor('mira','Mira',2,(.72,.27,.01),False),Actor('theo','Theo',2,(.72,.27,.01),False))
    # Bayes factor per observation is 2.33; the prior odds are 2.67, so one
    # report is insufficient but two independent reports reverse the preference.
    roots=(Observation('a','mira',0,1,.54),Observation('b','theo',0,1,.54),
           Observation('empty','courier',0,2,1.,'inspection_negative',(1.,1.,0.)))
    s=Scene('two-required',0,('A','B','C'),((0,2,1.),(1,2,1.)),actors,roots,(),matrices(3,5,0),tuple(1 for _ in range(6)),0,5,
            tuple((a.id,b.id) for a in actors for b in actors if a.id!=b.id))
    snap=Episode(s).at(0);p=PatchProblem('courier',1)
    r=search(snap,p)
    assert r.status=='verified_minimum_within_bounds' and len(r.patch)==2


def test_audit_tampering_rejected(snap):
    record=search(snap,PatchProblem('courier',1)).to_dict()
    bad=copy.deepcopy(record);bad['checkpoint_hash']='wrong'
    with pytest.raises(ValueError): check_audit(snap,bad)
    bad=copy.deepcopy(record);bad['candidates']=bad['candidates'][1:]
    with pytest.raises(ValueError): check_audit(snap,bad)
    bad=copy.deepcopy(record);bad['patch']=[]
    with pytest.raises(ValueError): check_audit(snap,bad)


def test_feasible_first_does_not_invent_minimum(snap):
    class BigFirst:
        def score(self,s,p,patches):return [len(x)*100+sum(d.root_id=='new-library' for d in x) for x in patches]
    r=search(snap,PatchProblem('courier',1),method='graph',ranker=BigFirst(),feasible_first=True)
    assert len(r.patch)>1 and r.status=='verified_feasible' and not r.all_smaller_checked
    assert not check_audit(snap,r.to_dict())['minimum']


def test_bad_ranker_cannot_accept_invalid_candidate(snap):
    class Bad:
        def score(self,s,p,patches):return [float('nan')]*len(patches)
    with pytest.raises(ValueError): search(snap,PatchProblem('courier',1),method='graph',ranker=Bad())


def test_no_goal_for_terminal_or_inactive_actor(snap):
    terminal=snap.clone();terminal.found['courier']=True
    with pytest.raises(ValueError):search(terminal,PatchProblem('courier',1))
    inactive=snap.clone();inactive.scene=replace(snap.scene,actors=(replace(snap.scene.actors[0],active=False),)+snap.scene.actors[1:])
    with pytest.raises(ValueError):search(inactive,PatchProblem('courier',1))


def test_auditor_rejects_fabricated_verification_fields(snap):
    record=search(snap,PatchProblem('courier',1)).to_dict()
    record['verification']['after_belief']=[0.,1.,0.,0.]
    with pytest.raises(ValueError):check_audit(snap,record)

@pytest.mark.parametrize('budget',[True,1.5,-1])
def test_invalid_verifier_budget(budget):
    s=demo_scene();snap=Episode(s).at(4)
    with pytest.raises(ValueError):search(snap,PatchProblem('courier',1),budget=budget)

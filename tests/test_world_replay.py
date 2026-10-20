from dataclasses import replace
import numpy as np
import pytest
from belieftown.scenarios import demo_scene,FAMILIES
from belieftown.schema import Message
from belieftown.world import Episode
from belieftown.replay import fork,compare_episodes
from belieftown.randomness import keyed_rng


@pytest.mark.parametrize('family',FAMILIES)
def test_all_authored_scenes_run(family):
    s=demo_scene(family);e=Episode(s)
    assert len(e.snapshots)==s.horizon+1
    assert e.snapshots[-1].found['courier']
    for snap in e.snapshots:
        for actor in s.actor_ids:
            assert np.isclose(snap.belief(actor).sum(),1)


def test_no_omniscience():
    s=demo_scene();snap=Episode(s).at(s.search_start);other=snap.clone()
    other.scene=replace(s,truth=tuple(2 for _ in s.truth))
    assert snap.view('courier')==other.view('courier')
    np.testing.assert_array_equal(snap.belief('courier'),other.belief('courier'))
    assert snap.plan('courier')==other.plan('courier')
    assert not hasattr(snap.view('courier'),'truth')


def test_removing_root_removes_downstream_messages():
    s=demo_scene('rumor_loop');e=Episode(s,suppress_observations=['old-workshop'])
    snap=e.at(s.search_start)
    assert 'old-workshop' not in snap.roots
    assert not any(m.root_id=='old-workshop' for m in snap.messages.values())


def test_suppress_one_path_preserves_real_alternate_path():
    s=demo_scene()
    messages=(Message('p1','old-workshop','theo','mira',1),
              Message('p2','old-workshop','theo','mira',2),
              Message('forward','old-workshop','mira','courier',3,'p1'))
    s=replace(s,messages=messages)
    snap=Episode(s,suppress_messages=['p1']).at(4)
    assert 'old-workshop' in snap.known('courier')
    assert snap.messages['forward'].parent_id=='p2'
    lost=Episode(s,suppress_messages=['p1','p2']).at(4)
    assert 'old-workshop' not in lost.known('courier')


def test_replay_is_deterministic_and_regenerates_actions():
    s=demo_scene();a=Episode(s);b=Episode(s)
    assert a.to_dict()==b.to_dict()
    c=fork(a,suppress_observations=['new-library'])
    summary=compare_episodes(a,c)
    assert summary['same_physical_history']
    assert summary['first_divergence'] is not None
    assert a.at(4).plan('mira').destination!=c.at(4).plan('mira').destination


def test_rng_keys_are_branch_independent():
    first=keyed_rng(12,'gate',8).random(10)
    keyed_rng(12,'another-event',1).random(100)
    np.testing.assert_array_equal(first,keyed_rng(12,'gate',8).random(10))
    assert not np.array_equal(first,keyed_rng(12,'gate',9).random(10))


def test_inspection_is_visible_only_next_tick():
    s=demo_scene();e=Episode(s)
    for snap in e.snapshots:
        assert all(r.time<=snap.time for r in snap.roots.values())
        assert all(r.received_at<=snap.time for receipts in snap.receipts.values() for r in receipts)


def test_invalid_suppression_fails():
    with pytest.raises(ValueError): Episode(demo_scene(),suppress_messages=['not-real'])

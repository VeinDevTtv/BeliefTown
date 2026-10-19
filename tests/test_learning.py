from dataclasses import replace
import json
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from belieftown.scenarios import demo_scene
from belieftown.world import Episode
from belieftown.interventions import PatchProblem,candidate_deliveries,search
from belieftown.learning.graph import encode_graph,GraphData
from belieftown.learning.model import configure,GraphRanker,MLPRanker,collate,Predictor
from belieftown.learning.data import make_dataset,load_cases
from belieftown.learning.train import train_model


def example():
    s=demo_scene();snap=Episode(s).at(4);p=PatchProblem('courier',1)
    return snap,p,(candidate_deliveries(snap,p)[0],)


def test_graph_does_not_contain_hidden_truth():
    snap,p,patch=example();other=snap.clone()
    other.scene=replace(snap.scene,truth=tuple(2 for _ in snap.scene.truth))
    a=encode_graph(snap,p,patch);b=encode_graph(other,p,patch)
    np.testing.assert_array_equal(a.x,b.x);np.testing.assert_array_equal(a.edges,b.edges)
    np.testing.assert_array_equal(a.global_features,b.global_features)


def test_graph_network_is_node_permutation_invariant():
    configure(0);snap,p,patch=example();g=encode_graph(snap,p,patch)
    perm=np.random.default_rng(1).permutation(len(g.x));inverse=np.argsort(perm)
    edges=g.edges.copy();edges[:,1]=inverse[edges[:,1]];edges[:,2]=inverse[edges[:,2]]
    shuffled=GraphData(g.x[perm],edges,g.global_features,inverse[g.target_index],inverse[g.destination_index])
    model=GraphRanker().eval()
    with torch.inference_mode(): values=model(collate([g,shuffled])).numpy()
    np.testing.assert_allclose(values[0],values[1],atol=1e-6)


@pytest.mark.parametrize('kind',['graph','mlp','graph_no_provenance'])
def test_checkpoint_roundtrip_and_exact_search(tmp_path,kind):
    configure(1);model=MLPRanker() if kind=='mlp' else GraphRanker();p=Predictor(model,kind)
    path=tmp_path/'model.pt';p.save(path);q=Predictor.load(path)
    snap,problem,patch=example()
    np.testing.assert_allclose(p.score(snap,problem,[patch]),q.score(snap,problem,[patch]),atol=0)
    result=search(snap,problem,method=kind,ranker=q)
    assert result.status=='verified_minimum_within_bounds' and len(result.patch)==1


def test_scene_splits_labels_and_training_end_to_end(tmp_path):
    manifest=make_dataset(tmp_path/'data',train=6,validation=3,test=3,ood=2,seed=13)
    hashes=[]
    for split in ('train','validation','test','ood'):
        cases=load_cases(tmp_path/'data'/f'{split}.jsonl')
        hashes.extend(c.scene.hash for c in cases)
        assert all(c.split==split for c in cases)
    assert len(hashes)==len(set(hashes))==14
    p=train_model(tmp_path/'data',tmp_path/'tiny.pt',kind='mlp',epochs=2,seed=1)
    assert np.isfinite(p.metadata['validation_bce'])
    assert len(p.metadata['history'])==2

@pytest.mark.parametrize('kind',['graph','mlp'])
def test_custom_architecture_checkpoint_roundtrip(tmp_path,kind):
    configure(0)
    model=MLPRanker(hidden=17) if kind=='mlp' else GraphRanker(hidden=16,layers=3)
    predictor=Predictor(model,kind);path=tmp_path/'custom.pt';predictor.save(path)
    restored=Predictor.load(path);snap,problem,patch=example()
    np.testing.assert_allclose(predictor.score(snap,problem,[patch]),restored.score(snap,problem,[patch]),atol=0)

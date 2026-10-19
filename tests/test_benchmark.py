"""An actual tiny training/evaluation pipeline plus statistical-contract tests."""
import json
import pytest
pytest.importorskip('torch')
from belieftown.benchmark import bootstrap_mean,run_benchmark
from belieftown.learning.data import make_dataset
from belieftown.learning.train import train_model


def test_bootstrap_units_and_empty_input():
    assert bootstrap_mean([])['n_scenes']==0
    assert bootstrap_mean([2.,2.,2.])=={'mean':2.,'low':2.,'high':2.,'n_scenes':3}


def test_small_three_model_benchmark(tmp_path):
    make_dataset(tmp_path/'data',train=6,validation=3,test=2,ood=2,seed=31)
    for kind in ('graph','mlp','graph_no_provenance'):
        train_model(tmp_path/'data',tmp_path/f'checkpoints/{kind}_seed0.pt',kind=kind,epochs=1)
    report=run_benchmark(tmp_path/'data',tmp_path/'checkpoints',tmp_path/'results',seeds=(0,))
    assert report['searches']==20 and report['all_searches_correct']
    assert len(report['summary'])==10 and len(report['paired_comparisons'])==8
    assert all(x['correctness']['n_scenes']==2 for x in report['summary'])
    assert '1 trained seed(s)' in (tmp_path/'results/RESULTS.md').read_text(encoding="utf-8")
    assert json.loads((tmp_path/'results/results.json').read_text(encoding="utf-8"))['protocol']['training_seeds']==[0]


@pytest.mark.parametrize('kwargs',[{'limit':0},{'limit':-1},{'seeds':()},{'seeds':(0,0)}])
def test_invalid_benchmark_request(tmp_path,kwargs):
    with pytest.raises(ValueError):run_benchmark(tmp_path,tmp_path,tmp_path,**kwargs)

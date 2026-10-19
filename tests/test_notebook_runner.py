"""Tiny runner tests; the release separately executes all four real notebooks."""
import pytest
nbformat=pytest.importorskip('nbformat')
pytest.importorskip('nbclient')
from belieftown.notebook_runner import execute_notebooks


def test_runner_rejects_empty_directory(tmp_path):
    with pytest.raises(ValueError):execute_notebooks(tmp_path,tmp_path/'out')


def test_runner_executes_and_records_outputs(tmp_path):
    source=tmp_path/'source';source.mkdir()
    nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('assert 6 * 7 == 42\nprint(42)')])
    nbformat.write(nb,source/'tiny.ipynb')
    report=execute_notebooks(source,tmp_path/'out')
    assert report['notebooks'][0]['errors']==0
    result=nbformat.read(tmp_path/'out/tiny.ipynb',as_version=4)
    assert result.cells[0].outputs[0].text.strip()=='42'


def test_runner_passes_reproduction_root_to_kernel(tmp_path):
    source=tmp_path/'source';source.mkdir()
    expected=str((tmp_path/'rebuilt').resolve())
    nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(f'import os\nassert os.environ["BELIEFTOWN_RUN_ROOT"] == {expected!r}')])
    nbformat.write(nb,source/'root.ipynb')
    report=execute_notebooks(source,tmp_path/'out',run_root=tmp_path/'rebuilt')
    assert report['run_root']==expected and report['notebooks'][0]['errors']==0

import json
import os
import subprocess
import sys
from belieftown.cli import main


def test_cli_solve_verify_replay(tmp_path,capsys):
    audit=tmp_path/'audit.json';replay=tmp_path/'replay.json'
    assert main(['solve','--room','Library','--output',str(audit)])==0
    assert main(['verify',str(audit)])==0
    assert main(['replay',str(audit),'--output',str(replay)])==0
    data=json.loads(replay.read_text(encoding="utf-8"))
    assert data['frames'][4]['actors'][0]['destination']==1


def test_cli_doctor_generate_errors(tmp_path,capsys):
    assert main(['doctor'])==0
    scene=tmp_path/'scene.json'
    assert main(['generate','--seed','3','--output',str(scene)])==0
    assert scene.exists()
    assert main(['solve','--room','No such room','--output',str(tmp_path/'bad.json')])==2
    assert not (tmp_path/'bad.json').exists()


def test_module_entrypoint():
    r=subprocess.run([sys.executable,'-m','belieftown','--version'],capture_output=True,text=True)
    assert r.returncode==0 and '1.0.0' in r.stdout


def test_help_with_legacy_windows_output_encoding():
    env = dict(os.environ, PYTHONIOENCODING='cp1252')
    result = subprocess.run(
        [sys.executable, '-m', 'belieftown', '--help'],
        capture_output=True, env=env,
    )
    assert result.returncode == 0, result.stderr.decode('cp1252')
    assert b'reproduce' in result.stdout

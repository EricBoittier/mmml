"""CPU-only tests of benchmark gating, provenance and ASV orchestration."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

_SPEC = importlib.util.spec_from_file_location(
    'gpu_report_runner', Path(__file__).resolve().parents[2] / 'benchmarks/run_gpu_report.py'
)
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


def test_gate_refuses_skips_and_empty_suites(tmp_path):
    report = tmp_path / 'gate.xml'
    for xml in ('<testsuites/>', '<testsuite><testcase><skipped/></testcase></testsuite>',
                '<testsuite><testcase><failure/></testcase></testsuite>'):
        report.write_text(xml)
        with pytest.raises(RuntimeError):
            runner.require_gate_report(report)
    report.write_text('<testsuite><testcase name="parity"/></testsuite>')
    assert runner.require_gate_report(report)['tests'] == 1


def test_result_summary_does_not_hide_partial_skips(tmp_path):
    (tmp_path / 'result.json').write_text(json.dumps({
        'commit_hash': 'abc', 'result_columns': ['result'],
        'results': {'bench': [[0.1, None, 0.2]], 'other': [None]},
    }))
    assert runner.result_summary(tmp_path) == {
        'measured_cases': 2, 'missing_or_failed_cases': ['bench', 'other'],
    }


def test_checkpoint_hash_tracks_contents(tmp_path):
    ckpt = tmp_path / 'params.json'
    ckpt.write_text('one')
    first = runner.checkpoint_hash(ckpt)
    ckpt.write_text('two')
    assert runner.checkpoint_hash(ckpt) != first


@pytest.mark.parametrize('fail_gate', [False, True])
def test_run_publishes_only_after_gate_and_saved_results(tmp_path, fail_gate):
    (tmp_path / 'asv.conf.json').write_text('{}')
    ckpt = tmp_path / 'checkpoint.json'
    ckpt.write_text('{}')
    calls = []

    def capture(*args):
        if args[:2] == ('git', 'status'):
            return ''
        if args[0] == 'git':
            return 'a' * 40
        return '0, GPU-busy, RTX 5090, 25000, 100\n1, GPU-free, RTX 5090, 1, 0'

    def run(cmd, **kwargs):
        calls.append(cmd)
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == '1'
        assert kwargs['env']['JAX_PLATFORMS'] == 'cuda'
        log = kwargs['stdout']
        if log.name.endswith('preflight.log'):
            log.write('{"devices":["gpu:0"]}\n')
        if 'pytest' in cmd:
            if fail_gate:
                raise subprocess.CalledProcessError(1, cmd)
            xml = Path(next(s.split('=', 1)[1] for s in cmd if s.startswith('--junitxml=')))
            xml.write_text('<testsuite><testcase name="parity"/></testsuite>')
        if 'asv' in cmd and 'run' in cmd:
            assert '--record-samples' in cmd and '--set-commit-hash' in cmd
            conf = json.loads(Path(cmd[cmd.index('--config') + 1]).read_text())
            results = Path(conf['results_dir'])
            results.mkdir()
            (results / 'a.json').write_text(json.dumps({
                'commit_hash': 'a' * 40, 'result_columns': ['result'],
                'results': {'benchmark': [[1.0]]},
            }))
        if 'publish' in cmd:
            conf = json.loads(Path(cmd[cmd.index('--config') + 1]).read_text())
            dest = Path(conf['html_dir'])
            dest.mkdir()
            (dest / 'index.html').write_text('ASV')

    with patch.object(runner, 'ROOT', tmp_path), patch.object(runner, 'capture', capture), \
            patch.object(runner.subprocess, 'run', run):
        rc = runner.main(['--checkpoint', str(ckpt), '--output', str(tmp_path / 'reports')])
    assert rc == int(fail_gate)
    assert any('publish' in c for c in calls) == (not fail_gate)
    report = next((tmp_path / 'reports').glob('*/status.json'))
    info = json.loads(report.read_text())
    assert info['status'] == ('failed' if fail_gate else 'complete')
    assert info['checkpoint_sha256'] == runner.checkpoint_hash(ckpt)
    assert info['gpu'][1] == 'GPU-free'


def test_busy_gpu_is_never_started(tmp_path):
    ckpt = tmp_path / 'checkpoint.json'
    ckpt.write_text('{}')

    def capture(*args):
        if args[:2] == ('git', 'status'):
            return ''
        if args[0] == 'git':
            return 'b' * 40
        return '1, GPU-busy, RTX 5090, 20000, 100'

    with patch.object(runner, 'capture', capture), patch.object(runner.subprocess, 'run') as run:
        with pytest.raises(SystemExit):
            runner.main(['--checkpoint', str(ckpt)])
        run.assert_not_called()


def test_report_is_served_and_escapes_metadata(tmp_path):
    import functools
    import http.server
    import threading
    import urllib.request

    runner.write_status(tmp_path, {'status': 'complete', 'commit': 'abc', 'host': '<unsafe>'})
    server = http.server.ThreadingHTTPServer(
        ('127.0.0.1', 0),
        functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path)),
    )
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{server.server_port}/') as response:
            page = response.read().decode()
        assert '&lt;unsafe&gt;' in page and '<unsafe>' not in page
        assert 'status.json' in page
    finally:
        server.shutdown()
        server.server_close()
        worker.join()

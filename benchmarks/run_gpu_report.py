#!/usr/bin/env python3
"""Validate a clean checkout, run ASV on one idle GPU, and serve its report.

Run with the benchmark environment's Python. No checkout, install, commit,
remote upload, or GPU-process termination is performed.
"""
from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import html
import http.server
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCH = r"bench_neighbors.PairListBackends|bench_calculator.JaxMLEnergy"
GATES = [
    "tests/unit/test_callback_buffers.py",
    "tests/unit/test_mm_pairs_vectorized.py",
    "tests/unit/test_pbc_utils_jax.py",
    "tests/unit/test_mm_pair_list_radius.py",
]


def capture(*cmd: str) -> str:
    return subprocess.check_output(cmd, cwd=ROOT, text=True).strip()


def checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"Empty checkpoint: {path}")
    for file in files:
        digest.update((str(file.relative_to(path)) if path.is_dir() else file.name).encode())
        with file.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


def require_gate_report(path: Path) -> dict:
    cases = list(ET.parse(path).getroot().iter('testcase'))
    counts = {key: sum(c.find(key) is not None for c in cases)
              for key in ('failure', 'error', 'skipped')}
    counts['tests'] = len(cases)
    if not cases or any(counts[k] for k in ('failure', 'error', 'skipped')):
        raise RuntimeError(f"Correctness gate incomplete: {counts}; see gate.log")
    return counts


def result_summary(results_dir: Path) -> dict:
    valid, missing = 0, []
    for file in results_dir.rglob('*.json'):
        data = json.loads(file.read_text())
        if 'commit_hash' not in data:
            continue
        index = data['result_columns'].index('result')
        for name, row in data['results'].items():
            values = row[index] if len(row) > index else None
            for value in values if isinstance(values, list) else [values]:
                if isinstance(value, (int, float)) and math.isfinite(value):
                    valid += 1
                else:
                    missing.append(name)
    return {'measured_cases': valid, 'missing_or_failed_cases': missing}


def write_status(run: Path, info: dict) -> None:
    info['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    tmp = run / 'status.tmp'
    tmp.write_text(json.dumps(info, indent=2) + '\n')
    tmp.replace(run / 'status.json')
    links = ''.join(f'<li><a href="{p.name}">{p.name}</a></li>'
                    for p in sorted(run.glob('*.log')))
    asv_link = '<a href="asv/">Open ASV charts →</a>' if (run / 'asv/index.html').exists() else ''
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MMML GPU benchmark</title><style>
body{{font:17px system-ui;max-width:980px;margin:50px auto;padding:0 24px;background:#101820;color:#e5edf4}}
a{{color:#72dbc8}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#1c2935;padding:20px;border-radius:12px}}
small{{color:#a9bbc9}}</style><h1>MMML GPU benchmark</h1>
<p><strong>{html.escape(info['status'])}</strong> · {html.escape(info.get('commit','')[:12])}</p>
<p>{asv_link}</p><p>ASV kernel/calculator benchmarks. Synthetic MD benchmarks, if selected,
are not the production ETOH PyCHARMM workload. Unit gates do not establish long-run conservation.</p>
<p>Only compare runs with matching hardware, precision, checkpoint and workload. Other GPU jobs may share host resources.</p>
<ul>{links}<li><a href="status.json">Provenance and status JSON</a></li></ul>
<pre>{html.escape(json.dumps(info, indent=2))}</pre><small>Refresh this page for progress. Logs update during each phase.</small></html>'''
    tmp = run / 'index.tmp'
    tmp.write_text(page)
    tmp.replace(run / 'index.html')


def main(argv=None) -> int:
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=ROOT, help='Clean checkout to benchmark (allows keeping this runner outside it)')
    parser.add_argument('--gpu', default='1', help='Physical NVIDIA GPU index (default: 1)')
    parser.add_argument('--bench', default=DEFAULT_BENCH, help='ASV benchmark regex')
    parser.add_argument('--x64', choices=['0', '1'], default='1')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--checkpoint', type=Path, default=None)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--serve', action='store_true', help='Serve this report until Ctrl-C, including during execution')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--dry-run', action='store_true', help='Print configuration without running tests or using a GPU')
    args = parser.parse_args(argv)
    ROOT = args.repo.expanduser().resolve()
    args.checkpoint = args.checkpoint or ROOT / 'examples/ckpts_json/DESdimers_params.json'
    args.output = args.output or ROOT / '.asv/gpu-reports'
    if not re.fullmatch(r'\d+', args.gpu) or args.threads < 1:
        parser.error('--gpu must be an index and --threads must be positive')
    if args.dry_run:
        print(json.dumps({'gpu': args.gpu, 'bench': args.bench, 'x64': args.x64,
                          'gates': GATES, 'output': str(args.output.resolve()),
                          'python': sys.executable, 'serve': args.serve}, indent=2))
        return 0
    dirty = capture('git', 'status', '--porcelain')
    if dirty:
        parser.error('Commit/stash changes in a dedicated checkout first. Refusing to label a dirty tree with a clean commit SHA.\n' + dirty)
    commit = capture('git', 'rev-parse', 'HEAD')
    ckpt = args.checkpoint.expanduser().resolve()
    ckpt_hash = checkpoint_hash(ckpt)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, JAX_PLATFORMS='cuda',
               MMML_BENCH_X64=args.x64, JAX_ENABLE_X64=args.x64,
               OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads),
               MKL_NUM_THREADS=str(args.threads), XLA_PYTHON_CLIENT_PREALLOCATE='false',
               MMML_CKPT=str(ckpt), MMML_BENCH_CKPT=str(ckpt),
               ASV_PYTHONPATH=str(ROOT), PYTHONPATH=str(ROOT))
    gpu_csv = capture('nvidia-smi', '--query-gpu=index,uuid,name,memory.used,utilization.gpu', '--format=csv,noheader,nounits')
    rows = [[x.strip() for x in row] for row in csv.reader(gpu_csv.splitlines())]
    selected = next((r for r in rows if r[0] == args.gpu), None)
    if selected is None or float(selected[3]) > 512 or float(selected[4]) > 5:
        parser.error(f'GPU {args.gpu} is unavailable or busy; no jobs were touched.\n{gpu_csv}')
    run = args.output.expanduser().resolve() / (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + commit[:10])
    run.mkdir(parents=True, exist_ok=False)
    info = {'status': 'preflight', 'commit': commit, 'host': socket.gethostname(),
            'gpu': selected, 'all_gpus_before': rows, 'x64': args.x64,
            'threads': args.threads, 'benchmark_regex': args.bench,
            'checkpoint': str(ckpt), 'checkpoint_sha256': ckpt_hash,
            'python': sys.executable, 'runner_sha256': checkpoint_hash(Path(__file__).resolve()),
            'scope': 'ASV suite; no production trajectory replay or NVE campaign'}
    write_status(run, info)
    server = None
    if args.serve:
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(run))
        server = http.server.ThreadingHTTPServer(('127.0.0.1', args.port), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f'Report: http://127.0.0.1:{args.port}/\nFrom your laptop: ssh -N -L {args.port}:127.0.0.1:{args.port} {socket.gethostname()}', flush=True)

    def execute(label: str, cmd: list[str]) -> None:
        info['status'] = label
        with (run / f'{label}.log').open('w') as log:
            write_status(run, info)
            print(f'{label}: {" ".join(cmd)}', flush=True)
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    code = 0
    try:
        probe = """import json, pathlib, jax, mmml, asv
print(json.dumps({'jax':jax.__version__, 'asv':asv.__version__, 'mmml':list(mmml.__path__), 'devices':[str(d) for d in jax.devices()]}))
assert len(jax.devices()) == 1 and jax.devices()[0].platform == 'gpu'
assert pathlib.Path(mmml.__file__).resolve().is_relative_to(pathlib.Path.cwd())
"""
        execute('preflight', [sys.executable, '-c', probe])
        info['runtime'] = json.loads((run / 'preflight.log').read_text().strip().splitlines()[-1])
        execute('packages', [sys.executable, '-c',
                             'import importlib.metadata as m, json; print(json.dumps(sorted((d.metadata.get("Name", ""), d.version) for d in m.distributions()), indent=2))'])
        execute('gate', [sys.executable, '-m', 'pytest', '-q', *GATES, f'--junitxml={run / "gate.xml"}'])
        info['gate'] = require_gate_report(run / 'gate.xml')
        machine = re.sub(r'[^a-zA-Z0-9_-]', '-', f'{socket.gethostname()}-{selected[1]}-x64{args.x64}-t{args.threads}-ck{ckpt_hash[:10]}')
        config = json.loads((ROOT / 'asv.conf.json').read_text())
        config.update(repo=str(ROOT), branches=[commit], benchmark_dir=str(ROOT / 'benchmarks/benchmarks'),
                      results_dir=str(run / 'results'), html_dir=str(run / 'asv'), env_dir=str(run / 'env'))
        conf = run / 'asv.conf.json'
        conf.write_text(json.dumps(config, indent=2))
        asv = [sys.executable, '-m', 'asv']
        execute('machine', [*asv, 'machine', '--yes', '--machine', machine, '--config', str(conf)])
        execute('benchmark', [*asv, 'run', '--config', str(conf), '--machine', machine,
                             '--set-commit-hash', commit, '--bench', args.bench,
                             '--record-samples', '--show-stderr', '--no-pull'])
        info['results'] = result_summary(run / 'results')
        if not info['results']['measured_cases'] or info['results']['missing_or_failed_cases']:
            raise RuntimeError('ASV has missing/failed/skipped results; inspect benchmark.log. Not publishing a success report.')
        if capture('git', 'rev-parse', 'HEAD') != commit or capture('git', 'status', '--porcelain'):
            raise RuntimeError('Checkout changed during the run; results are not validated.')
        execute('publish', [*asv, 'publish', '--config', str(conf)])
        info['status'] = 'complete'
    except (Exception, KeyboardInterrupt) as exc:
        info['status'] = 'failed'
        info['error'] = str(exc) or type(exc).__name__
        code = 1
    finally:
        write_status(run, info)
    print(f'{info["status"]}: {run / "index.html"}', flush=True)
    if server:
        print('Serving report until Ctrl-C. To serve it again: ' +
              f'{sys.executable} -m http.server {args.port} --bind 127.0.0.1 --directory {run}', flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            server.shutdown()
    return code


if __name__ == '__main__':
    raise SystemExit(main())

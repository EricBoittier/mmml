# Remote MD runs with live streaming to a local UI

Run `mmml md-system` on a remote/HPC host and watch its logs, output files, and
trajectory frames from a browser on your laptop — live, as the run produces them.

This reuses the existing `mmml gui` viewer and adds an opt-in **job runner** that
launches `md-system` subprocesses and streams their progress over
[Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events).

## Architecture

```
 your laptop                         remote host (login/compute node)
┌──────────────┐   SSH tunnel   ┌──────────────────────────────────────┐
│ browser      │◄──────────────►│ mmml gui --enable-runner             │
│ localhost:8000│   :8000        │  ├─ JobManager: spawn `mmml md-system`│
└──────────────┘                │  │    capture stdout/stderr          │
                                │  ├─ SSE /api/jobs/{id}/events         │
                                │  │    log + file + status events      │
                                │  └─ /api/file, /api/frame …           │
                                │       serve trajectory as it grows    │
                                └──────────────────────────────────────┘
```

- **Runner** (`mmml/gui/api/runner.py`): `JobManager` launches an allowlisted
  `mmml` subprocess in its own process group, tails stdout/stderr into a bounded
  ring buffer with monotonic sequence numbers, and polls the run's
  `--output-dir` for new/changed files.
- **Routes** (`mmml/gui/api/runner_routes.py`): REST + SSE endpoints, mounted
  only when `--enable-runner` is passed.
- **Viewer**: the existing `/api/file` / `/api/frame` endpoints render frames
  from the job's output directory as trajectory files are written.

## Security model

The runner executes subprocesses, so it is **off by default** and, when on,
**refuses to bind to a non-loopback interface**. The intended deployment is to
bind to `127.0.0.1` on the remote host and reach it through an SSH port-forward —
no public port, and it rides your existing SSH authentication.

Only the `mmml` command is allowlisted (`JobManager.ALLOWED_COMMANDS`); arbitrary
shell commands are rejected with HTTP 400.

To bind to a routable address anyway (e.g. inside a trusted private network),
set `MMML_GUI_ALLOW_REMOTE_RUNNER=1`.

## Quick start

On the remote host:

```bash
# bind to loopback; runner enabled
mmml gui --enable-runner --data-dir /scratch/$USER/runs --no-browser --port 8000
```

On your laptop:

```bash
# forward local :8000 to the remote's loopback :8000
ssh -N -L 8000:127.0.0.1:8000 user@remote-host
# then open http://localhost:8000
```

## Launching a job

`POST /api/jobs` with either a full `argv` or a shell-style `command` string
(first token must be `mmml`):

```bash
curl -s localhost:8000/api/jobs -H 'content-type: application/json' -d '{
  "command": "mmml md-system --setup pbc_nvt --backend jaxmd --composition DCM:64 --checkpoint /scratch/ckpt --output-dir runs/dcm_nvt --ps 50 --temperature 260",
  "label": "dcm-nvt"
}'
# -> {"id":"ab12cd34ef56","status":"running","output_dir":"...","last_seq":1, ...}
```

The runner auto-detects `--output-dir` from the argv and watches it. Pass
`output_dir` in the body to override what gets watched.

## Streaming progress

Tail the combined event stream (log lines, file events, status changes). A late
subscriber first receives a replay of the retained history, then live events,
then an `end` event when the job terminates:

```bash
curl -N localhost:8000/api/jobs/ab12cd34ef56/events
# event: log
# data: {"kind":"log","seq":12,"time":...,"stream":"stdout","text":"step 500 ..."}
# event: file
# data: {"kind":"file","seq":13,"relpath":"traj.dcd","size":1048576,"mtime":...}
# event: status
# data: {"kind":"status","seq":99,"status":"succeeded","exit_code":0}
```

Browser side, use the native `EventSource` API:

```js
const es = new EventSource(`/api/jobs/${id}/events`);
es.addEventListener("log",    e => appendLine(JSON.parse(e.data)));
es.addEventListener("file",   e => refreshFileList(JSON.parse(e.data)));
es.addEventListener("status", e => setStatus(JSON.parse(e.data)));
es.addEventListener("end",    () => es.close());
```

## Endpoint reference

| Method & path | Purpose |
| --- | --- |
| `POST /api/jobs` | Launch a job (`argv` or `command`); returns a status snapshot |
| `GET /api/jobs` | List jobs |
| `GET /api/jobs/{id}` | One job's status snapshot |
| `GET /api/jobs/{id}/logs?since=SEQ` | Captured log lines after `SEQ` (poll fallback) |
| `GET /api/jobs/{id}/files` | Output-dir file manifest (relpath/size/mtime) |
| `GET /api/jobs/{id}/events?replay=true` | SSE stream of log/file/status events |
| `POST /api/jobs/{id}/stop` | SIGTERM the process group, escalate to SIGKILL |
| `GET /api/runner/config` | Runner settings (cwd, allowlist, limits) |

## Notes and limitations

- **In-memory job registry.** Jobs live in the server process; restarting
  `mmml gui` forgets them (the output files on disk remain). Persisting job
  metadata across restarts is a possible follow-up.
- **File watching is poll-based** (default 1 s). Good enough for MD cadence;
  swap for `watchdog` if sub-second latency is needed.
- **SLURM / batch schedulers.** This runs `md-system` directly on the host
  serving the GUI. To drive batch jobs, wrap submission in the `command`
  (e.g. `mmml md-system ...` on an allocated interactive node), or extend the
  runner with a scheduler adapter.

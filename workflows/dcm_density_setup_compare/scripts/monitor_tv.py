#!/usr/bin/env python3
"""Rotating multi-job monitor ("TV channels") for dcm_density_setup_compare."""

from __future__ import annotations

import argparse
import json
import os
import re
import select
import subprocess
import sys
import termios
import time
import tty
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from campaign_lib import (  # noqa: E402
    cell_run_tag,
    iter_matrix_cells,
    load_config,
    repo_root,
)

try:
    from rich import box
    from rich.align import Align
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError as exc:  # pragma: no cover - rich is a project dep
    raise SystemExit("monitor_tv.py requires rich (install mmml deps)") from exc

_ANSI = re.compile(r"\x1b\[[0-9;]*(?:[0-9;]*[A-Za-z])|\x1b\][^\x07]*(?:\x07|\x1b\\)")
_GRMS_PRE_OK = re.compile(r"Pre-dynamics GRMS OK:\s*([0-9.]+)")
_GRMS_PRE_FAIL = re.compile(r"Pre-dynamics GRMS\s+([0-9.]+)\s*>\s*([0-9.]+)")
_HEAT_SEG = re.compile(r"heat segment\s+(\d+)/(\d+)", re.I)
_ERROR = re.compile(r"pycharmm_mlpot: error:\s*(.+)", re.I)
_STAGE = re.compile(
    r"(Packmol|MLpot SD|Pre-dynamics|heat segment|pycharmm_equi|pycharmm_prod|jaxmd|ase_prod)",
    re.I,
)
_TAG_SHORT = re.compile(r"_dcm_(\d+)_t(\d+)_l(\d+)")


@dataclass
class CellSnapshot:
    tag: str
    log_path: Path
    done: bool
    failed: bool
    live: bool
    grms: str | None
    stage: str | None
    heat: str | None
    error: str | None
    slurm_job: str | None
    mtime: float
    tail_lines: list[str] = field(default_factory=list)


def _strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


def _short_label(tag: str) -> str:
    m = _TAG_SHORT.search(tag)
    if m:
        return f"N={m.group(1)} T={m.group(2)} L={m.group(3)}"
    if len(tag) > 42:
        return tag[:19] + "…" + tag[-19:]
    return tag


def _workflow_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_config(path: str | None) -> Path:
    wf = _workflow_root()
    raw = path or os.environ.get("MMML_WORKFLOW_CONFIG", "config.yaml")
    p = Path(raw)
    if not p.is_absolute():
        p = wf / p
    return p.resolve()


def _artifact_root(cfg: dict[str, Any]) -> Path:
    raw = cfg.get("output_root", "artifacts/dcm_density_setup_compare")
    root = Path(raw)
    if not root.is_absolute():
        root = repo_root() / root
    return root.resolve()


def _state_path() -> Path:
    return _workflow_root() / ".monitor_tv" / "state.json"


def _load_state() -> dict[str, Any]:
    p = _state_path()
    if not p.is_file():
        return {"index": 0, "paused": False, "channels": []}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {"index": 0, "paused": False, "channels": []}


def _save_state(state: dict[str, Any]) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2))


def _squeue_rows() -> list[dict[str, str]]:
    if not shutil_which("squeue"):
        return []
    try:
        out = subprocess.check_output(
            [
                "squeue",
                "-u",
                os.environ.get("USER", ""),
                "-h",
                "-o",
                "%i|%P|%M|%T|%j",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    rows: list[dict[str, str]] = []
    for line in out.splitlines():
        parts = line.split("|", 4)
        if len(parts) < 5:
            continue
        rows.append(
            {
                "id": parts[0].strip(),
                "partition": parts[1].strip(),
                "time": parts[2].strip(),
                "state": parts[3].strip(),
                "name": parts[4].strip(),
            }
        )
    return rows


def shutil_which(cmd: str) -> str | None:
    from shutil import which

    return which(cmd)


def _match_slurm_job(tag: str, rows: list[dict[str, str]]) -> str | None:
    needle = tag.replace("_ht_", "_").replace("_sw_", "_")
    for row in rows:
        name = row["name"]
        if tag in name or needle in name:
            return row["id"]
    # Snakemake often embeds tag at end of job name
    for row in rows:
        if tag.split("_")[-1] in row["name"]:
            return row["id"]
    return None


def discover_channels(
    cfg: dict[str, Any],
    artifact_root: Path,
    explicit: list[str] | None,
    *,
    include_done: bool = False,
) -> list[str]:
    if explicit:
        return list(dict.fromkeys(explicit))

    matrix_tags = [cell_run_tag(c, cfg) for c in iter_matrix_cells(cfg)]
    seen: set[str] = set()
    ordered: list[str] = []

    def add(tag: str) -> None:
        if tag not in seen:
            seen.add(tag)
            ordered.append(tag)

    # Active / incomplete matrix cells first
    for tag in matrix_tags:
        art = artifact_root / tag
        done = (art / "done.txt").is_file()
        log = art / "stdout.log"
        if not done or (log.is_file() and log.stat().st_mtime > time.time() - 3600):
            add(tag)

    # Recent artifact dirs (prep sweep tags, reruns)
    if artifact_root.is_dir():
        recent: list[tuple[float, str]] = []
        for d in artifact_root.iterdir():
            if not d.is_dir():
                continue
            log = d / "stdout.log"
            if log.is_file():
                recent.append((log.stat().st_mtime, d.name))
        for _, tag in sorted(recent, reverse=True)[:40]:
            add(tag)

    # Remaining matrix tags
    for tag in matrix_tags:
        add(tag)

    if not include_done:
        filtered = []
        for tag in ordered:
            if (artifact_root / tag / "done.txt").is_file():
                continue
            filtered.append(tag)
        if filtered:
            return filtered
    return ordered if ordered else matrix_tags


def _parse_log(tag: str, art: Path, rows: list[dict[str, str]]) -> CellSnapshot:
    log = art / "stdout.log"
    done = (art / "done.txt").is_file()
    text = ""
    mtime = 0.0
    if log.is_file():
        mtime = log.stat().st_mtime
        text = _strip_ansi(log.read_text(errors="replace"))
    plain_lines = text.splitlines()
    tail = plain_lines[-18:] if plain_lines else ["(no stdout.log yet — queued or not started)"]

    grms: str | None = None
    m = _GRMS_PRE_OK.search(text)
    if m:
        grms = f"OK {m.group(1)}"
    else:
        m = _GRMS_PRE_FAIL.search(text)
        if m:
            grms = f"FAIL {m.group(1)}>{m.group(2)}"

    heat: str | None = None
    hm = list(_HEAT_SEG.finditer(text))
    if hm:
        last = hm[-1]
        heat = f"{last.group(1)}/{last.group(2)}"

    stage: str | None = None
    for pat in _STAGE.finditer(text):
        stage = pat.group(1)
    if stage and stage.lower().startswith("heat"):
        stage = "heat"

    err: str | None = None
    em = _ERROR.search(text)
    if em:
        err = em.group(1).strip()[:120]

    failed = bool(err) or ("Campaign summary reports failed" in text)
    live = (time.time() - mtime) < 90 if mtime else False

    return CellSnapshot(
        tag=tag,
        log_path=log,
        done=done,
        failed=failed and not done,
        live=live and not done,
        grms=grms,
        stage=stage,
        heat=heat,
        error=err,
        slurm_job=_match_slurm_job(tag, rows),
        mtime=mtime,
        tail_lines=tail,
    )


def _status_badge(snap: CellSnapshot) -> Text:
    if snap.done:
        return Text(" DONE ", style="bold white on green")
    if snap.failed:
        return Text(" FAIL ", style="bold white on red")
    if snap.live:
        return Text(" LIVE ", style="bold black on yellow")
    if snap.slurm_job:
        return Text(" RUN  ", style="bold white on blue")
    return Text(" IDLE ", style="bold white on grey50")


def _age(mtime: float) -> str:
    if mtime <= 0:
        return "—"
    delta = max(0, int(time.time() - mtime))
    if delta < 60:
        return f"{delta}s ago"
    if delta < 3600:
        return f"{delta // 60}m ago"
    return f"{delta // 3600}h ago"


def _matrix_board(snapshots: dict[str, CellSnapshot], channels: list[str]) -> Table:
    cols = 6
    table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    row: list[Any] = []
    for tag in channels:
        snap = snapshots.get(tag)
        if snap is None:
            sym, style = "·", "dim"
        elif snap.done:
            sym, style = "✓", "bold green"
        elif snap.failed:
            sym, style = "✗", "bold red"
        elif snap.live:
            sym, style = "▶", "bold yellow"
        elif snap.slurm_job:
            sym, style = "◉", "bold cyan"
        else:
            sym, style = "○", "dim"
        label = _short_label(tag)
        if len(label) > 16:
            label = label[:14] + "…"
        row.append(Text.assemble((sym + " ", style), (label, "cyan")))
        if len(row) == cols:
            table.add_row(*row)
            row = []
    if row:
        table.add_row(*row, *[Text("") for _ in range(cols - len(row))])
    return table


def _queue_table(rows: list[dict[str, str]]) -> Panel:
    t = Table(box=box.SIMPLE_HEAVY, expand=True, show_lines=False)
    t.add_column("JOB", style="bold", width=10)
    t.add_column("ST", width=4)
    t.add_column("TIME", width=8)
    t.add_column("NAME", overflow="ellipsis")
    if not rows:
        t.add_row("—", "—", "—", "(empty queue)")
    else:
        for r in rows[:12]:
            st = r["state"]
            st_style = "green" if st == "RUNNING" else "yellow" if st == "PENDING" else "red"
            t.add_row(r["id"], Text(st, style=st_style), r["time"], r["name"][:48])
    return Panel(t, title="[bold]Slurm queue[/]", border_style="blue")


def render_dashboard(
    channels: list[str],
    index: int,
    snapshots: dict[str, CellSnapshot],
    *,
    interval: float,
    paused: bool,
    driver_log: str,
) -> Layout:
    if not channels:
        channels = ["(no channels)"]
    index %= len(channels)
    tag = channels[index]
    snap = snapshots.get(tag)

    root = Layout(name="root")
    root.split_column(
        Layout(name="main", ratio=3),
        Layout(name="footer", ratio=2),
    )
    root["footer"].split_row(
        Layout(name="queue", ratio=2),
        Layout(name="board", ratio=3),
    )

    done_n = sum(1 for t in channels if snapshots.get(t) and snapshots[t].done)
    fail_n = sum(1 for t in channels if snapshots.get(t) and snapshots[t].failed)
    live_n = sum(1 for t in channels if snapshots.get(t) and snapshots[t].live)

    now = datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S")
    header = Text.assemble(
        (" 📺 ", "bold"),
        ("DCM SETUP COMPARE", "bold magenta"),
        ("  ·  ", "dim"),
        (f"CH {index + 1:02d}/{len(channels):02d}", "bold cyan"),
        ("  ·  ", "dim"),
        (now, "dim"),
    )
    if paused:
        header.append_text(Text("  ⏸ PAUSED", style="bold yellow"))

    if snap is None:
        body = Text(f"Unknown channel: {tag}", style="red")
        badge = Text(" ???? ", style="white on red")
    else:
        badge = _status_badge(snap)
        meta = Table.grid(padding=(0, 2))
        meta.add_row(
            "Tag", Text(snap.tag, style="bold white"),
            "Short", _short_label(snap.tag),
        )
        meta.add_row(
            "GRMS", snap.grms or "—",
            "Stage", snap.stage or "—",
        )
        meta.add_row(
            "Heat", snap.heat or "—",
            "Slurm", snap.slurm_job or "—",
        )
        meta.add_row(
            "Log age", _age(snap.mtime),
            "Done", "yes" if snap.done else "no",
        )
        if snap.error:
            meta.add_row("Error", Text(snap.error, style="bold red"))

        log_body = Text("\n".join(snap.tail_lines), style="white")
        body = Group(
            meta,
            Panel(log_body, title="[dim]stdout tail[/]", border_style="dim", height=14),
        )

    stats = Text.assemble(
        ("✓ ", "green"), (str(done_n), "bold green"), (" done  ", "dim"),
        ("✗ ", "red"), (str(fail_n), "bold red"), (" fail  ", "dim"),
        ("▶ ", "yellow"), (str(live_n), "bold yellow"), (" live  ", "dim"),
        (f"rotate {interval:.0f}s", "dim"),
        ("  │  ", "dim"),
        (driver_log, "dim cyan"),
    )
    controls = Text(
        "Focus this pane: n next · p prev · Space pause  │  any pane: Ctrl-b then n/p/Space",
        style="dim italic",
    )

    main_panel = Panel(
        Group(
            Align.center(Text.assemble(header, "\n", badge, " ", Text(tag, style="bold cyan"))),
            body,
            stats,
            controls,
        ),
        border_style="magenta",
        box=box.DOUBLE,
        title="[bold]MAIN CHANNEL[/]",
    )
    root["main"].update(main_panel)

    rows = _squeue_rows()
    root["queue"].update(_queue_table(rows))
    root["board"].update(
        Panel(
            _matrix_board(snapshots, channels),
            title=f"[bold]Matrix ({done_n}/{len(channels)})[/]",
            border_style="green",
        )
    )
    return root


def _channel_status_line(state: dict[str, Any]) -> str:
    channels = state.get("channels") or []
    n = max(len(channels), 1)
    idx = int(state.get("index", 0)) % n
    tag = channels[idx] if channels else "(empty)"
    paused = " ⏸" if state.get("paused") else ""
    return f"CH {idx + 1}/{n}{paused} · {tag}"


def _apply_ctl_action(state: dict[str, Any], action: str) -> dict[str, Any]:
    channels = state.get("channels") or []
    n = max(len(channels), 1)
    idx = int(state.get("index", 0))
    if action == "next":
        state["index"] = (idx + 1) % n
    elif action == "prev":
        state["index"] = (idx - 1) % n
    elif action == "pause":
        state["paused"] = not bool(state.get("paused", False))
    else:
        raise ValueError(f"unknown action: {action}")
    return state


def _poll_keyboard(timeout: float) -> str | None:
    """Non-blocking single-key read when stdin is a tty (TV pane focused)."""
    if not sys.stdin.isatty():
        return None
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        return None
    ch = sys.stdin.read(1)
    if not ch:
        return None
    if ch == "\x1b":  # arrow keys / escape sequences
        extra, _, _ = select.select([sys.stdin], [], [], 0.02)
        if extra:
            seq = ch + sys.stdin.read(2)
            if seq.endswith("C"):
                return "next"
            if seq.endswith("D"):
                return "prev"
        return None
    if ch in ("n", "N"):
        return "next"
    if ch in ("p", "P"):
        return "prev"
    if ch == " ":
        return "pause"
    return None


def run_live(args: argparse.Namespace) -> int:
    cfg_path = _resolve_config(args.config)
    cfg = load_config(cfg_path)
    art = _artifact_root(cfg)

    channels = discover_channels(
        cfg, art, args.tags or None, include_done=args.include_done
    )
    state = _load_state()
    if args.tags:
        state["channels"] = channels
    elif state.get("channels"):
        # Keep order but refresh with any new tags
        old = state["channels"]
        channels = [t for t in old if t in channels] + [t for t in channels if t not in old]
    state["channels"] = channels
    state.setdefault("index", 0)
    state.setdefault("paused", False)
    _save_state(state)

    console = Console()
    last_rotate = 0.0
    last_mtime = 0.0
    stdin_mode: list[Any] | None = None
    if sys.stdin.isatty():
        stdin_mode = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    try:
        with Live(console=console, refresh_per_second=4, screen=True) as live:
            while True:
                state = _load_state()
                channels = state.get("channels") or channels
                if not channels:
                    channels = ["(empty)"]
                idx = int(state.get("index", 0)) % len(channels)
                paused = bool(state.get("paused", False))

                rows = _squeue_rows()
                snapshots = {
                    t: _parse_log(t, art / t, rows) for t in channels if t != "(empty)"
                }

                live.update(
                    render_dashboard(
                        channels,
                        idx,
                        snapshots,
                        interval=args.interval,
                        paused=paused,
                        driver_log=args.driver_log,
                    )
                )

                # Keys in this pane (no Ctrl-b prefix needed)
                key_action = _poll_keyboard(0.25)
                if key_action:
                    state = _apply_ctl_action(state, key_action)
                    _save_state(state)
                    last_rotate = time.time()
                    last_mtime = _state_path().stat().st_mtime
                    continue

                # External ctl (tmux / monitor_tv_ctl.sh)
                st_path = _state_path()
                mtime = st_path.stat().st_mtime if st_path.is_file() else 0.0
                if mtime != last_mtime:
                    last_mtime = mtime
                    last_rotate = time.time()

                if not paused and (time.time() - last_rotate) >= args.interval:
                    state["index"] = (idx + 1) % len(channels)
                    _save_state(state)
                    last_rotate = time.time()
                    last_mtime = _state_path().stat().st_mtime
    finally:
        if stdin_mode is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, stdin_mode)
    return 0


def cmd_ctl(action: str, *, message: bool = False) -> int:
    try:
        state = _apply_ctl_action(_load_state(), action)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    _save_state(state)
    if message:
        print(_channel_status_line(state))
    return 0


def init_state_from_config(args: argparse.Namespace) -> list[str]:
    cfg = load_config(_resolve_config(args.config))
    art = _artifact_root(cfg)
    channels = discover_channels(
        cfg, art, args.tags or None, include_done=args.include_done
    )
    _save_state({"index": 0, "paused": False, "channels": channels})
    return channels


def cmd_init(args: argparse.Namespace) -> int:
    init_state_from_config(args)
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    cfg = load_config(_resolve_config(args.config))
    art = _artifact_root(cfg)
    channels = discover_channels(cfg, art, args.tags or None, include_done=True)
    rows = _squeue_rows()
    for i, tag in enumerate(channels):
        snap = _parse_log(tag, art / tag, rows)
        mark = (
            "DONE"
            if snap.done
            else "FAIL"
            if snap.failed
            else "LIVE"
            if snap.live
            else "RUN"
            if snap.slurm_job
            else "idle"
        )
        print(f"{i + 1:3d}  [{mark:4s}]  {tag}")
    return 0


def main() -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="workflow config (default: MMML_WORKFLOW_CONFIG)")
    common.add_argument("--tags", nargs="*", help="explicit channel list")
    common.add_argument(
        "--include-done",
        action="store_true",
        help="keep finished cells in rotation",
    )

    parser = argparse.ArgumentParser(
        description=__doc__,
        parents=[common],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--interval", type=float, default=12.0, help="auto-rotate seconds")
    parser.add_argument("--driver-log", default="snakemake_slurm.log")

    sub = parser.add_subparsers(dest="cmd", required=False)
    sub.add_parser("live", help="full-screen TV dashboard (default)")
    sub.add_parser("init", help="seed .monitor_tv/state.json channel list")
    sub.add_parser("list", help="print channel list")

    p_ctl = sub.add_parser("ctl", help="next|prev|pause (for tmux bindings)")
    p_ctl.add_argument("action", choices=["next", "prev", "pause"])
    p_ctl.add_argument(
        "--message",
        action="store_true",
        help="print channel status line (for tmux display-message)",
    )

    args = parser.parse_args()
    cmd = args.cmd or "live"
    if cmd == "ctl":
        return cmd_ctl(args.action, message=bool(args.message))
    if cmd == "init":
        return cmd_init(args)
    if cmd == "list":
        return cmd_list(args)
    return run_live(args)


if __name__ == "__main__":
    raise SystemExit(main())

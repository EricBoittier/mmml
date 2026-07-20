import json
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=REPO, check=True, capture_output=True, text=True
    )
    return [path for path in result.stdout.splitlines() if (REPO / path).exists()]


def test_runtime_logs_are_not_tracked() -> None:
    offenders = [path for path in _tracked_files() if Path(path).name == "nohup.out"]
    assert offenders == []


def test_cleaned_exploration_notebook_is_named_and_has_no_outputs() -> None:
    tracked = _tracked_files()
    assert "examples/Untitled1.ipynb" not in tracked
    relative = "examples/pycharmm_jax_nonbonded_exploration.ipynb"
    assert relative in tracked
    notebook = json.loads((REPO / relative).read_text(encoding="utf-8"))
    assert not any(cell.get("outputs") for cell in notebook.get("cells", []))


def test_active_workflow_code_has_no_personal_checkpoint_fallback() -> None:
    needle = "/mmhome/boittier/home/mmml_tutorial/acodcm/ckpts"
    offenders: list[str] = []
    for relative in _tracked_files():
        path = Path(relative)
        if not relative.startswith("workflows/") or path.suffix not in {".py", ".sh"}:
            continue
        if needle in (REPO / relative).read_text(encoding="utf-8", errors="replace"):
            offenders.append(relative)
    assert offenders == []

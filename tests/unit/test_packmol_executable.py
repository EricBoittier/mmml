from __future__ import annotations

from pathlib import Path
from unittest import mock

from mmml.interfaces.pycharmmInterface import packmol_placement


def test_binary_runs_on_host_rejects_linux_elf_on_darwin(monkeypatch, tmp_path):
    path = tmp_path / "packmol-stub"
    path.write_bytes(b"stub")
    monkeypatch.setattr(packmol_placement.os, "access", lambda *_a, **_k: True)
    monkeypatch.setattr(packmol_placement.sys, "platform", "darwin")
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.packmol_placement.subprocess.run",
        return_value=mock.Mock(
            returncode=0,
            stdout="ELF 64-bit LSB executable, x86-64",
        ),
    ):
        assert packmol_placement._binary_runs_on_host(path) is False


def test_binary_runs_on_host_accepts_mach_o_on_darwin(monkeypatch, tmp_path):
    path = tmp_path / "packmol-stub"
    path.write_bytes(b"stub")
    monkeypatch.setattr(packmol_placement.os, "access", lambda *_a, **_k: True)
    monkeypatch.setattr(packmol_placement.sys, "platform", "darwin")
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.packmol_placement.subprocess.run",
        return_value=mock.Mock(
            returncode=0,
            stdout="Mach-O 64-bit executable arm64",
        ),
    ):
        assert packmol_placement._binary_runs_on_host(path) is True


def test_packmol_executable_uses_build_cache(monkeypatch, tmp_path):
    binary = tmp_path / "install" / "bin" / "packmol"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"stub")
    monkeypatch.setenv("PACKMOL_BUILD_DIR", str(tmp_path))

    def _runs(path: Path) -> bool:
        return Path(path) == binary

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.packmol_placement._binary_runs_on_host",
        side_effect=_runs,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.packmol_placement.shutil.which",
        return_value=None,
    ):
        assert packmol_placement.packmol_executable() == str(binary)


def test_packmol_executable_skips_foreign_bundled_binary(monkeypatch):
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.packmol_placement._binary_runs_on_host",
        return_value=False,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.packmol_placement.shutil.which",
        return_value="/usr/local/bin/packmol",
    ):
        assert packmol_placement.packmol_executable() == "/usr/local/bin/packmol"

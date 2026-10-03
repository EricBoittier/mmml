import pytest

import pycharmm.read as read
import pycharmm.script as script
import pycharmm.write as write


def _capture_scripts(monkeypatch):
    scripts = []

    def fake_run(self, append="", raise_on_error=None):
        scripts.append(self.create_script_string() + append)
        return self

    monkeypatch.setattr(script.CommandScript, "run", fake_run)
    return scripts


def test_read_mmcif_builds_coordinate_command(monkeypatch):
    scripts = _capture_scripts(monkeypatch)

    read.mmcif("model.cif", resid=True, model=2, label=True)

    assert scripts == ["read coor mmcif -\n name model.cif -\n resid -\n model 2 -\n label\n"]


def test_sequence_mmcif_builds_sequence_command(monkeypatch):
    scripts = _capture_scripts(monkeypatch)

    read.sequence_mmcif("model.cif", segi="WAT1", hetatm=True, noatom=True)

    assert scripts == [
        "read sequence mmcif -\n name model.cif -\n segi WAT1 -\n hetatm -\n noatom\n"
    ]


def test_sequence_mmcif_rejects_multiple_skip_values():
    with pytest.raises(ValueError, match="skip accepts one residue name"):
        read.sequence_mmcif("model.cif", skip=["HOH", "SO4"])


def test_sequence_mmcif_builds_alias_pairs(monkeypatch):
    scripts = _capture_scripts(monkeypatch)

    read.sequence_mmcif(
        "model.cif",
        alias={"HSD": "HIS", "HSE": "HIS"},
    )

    assert scripts == ["read sequence mmcif -\n name model.cif -\n alias HSD HIS ALIAS HSE HIS\n"]


def test_sequence_mmcif_ignores_empty_alias_map(monkeypatch):
    scripts = _capture_scripts(monkeypatch)

    read.sequence_mmcif("model.cif", alias={})

    assert scripts == ["read sequence mmcif -\n name model.cif\n"]


def test_write_mmcif_builds_coordinate_command(monkeypatch):
    scripts = _capture_scripts(monkeypatch)

    write.coor_mmcif("out.cif", model=1)

    assert scripts == ["write name out.cif -\n coor mmcif -\n model 1\n"]


def test_write_mmcif_rejects_ignored_options():
    with pytest.raises(ValueError, match="Unsupported mmCIF write option"):
        write.coor_mmcif("out.cif", first=True)

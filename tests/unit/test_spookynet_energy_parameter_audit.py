from __future__ import annotations

import json

from scripts.audit_spookynet_energy_parameters import audit


def test_audit_distinguishes_vdw_head_from_trainable_zbl(tmp_path):
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "config": {"zbl": True, "cutoff": 6.0},
                "params": {
                    "params": {
                        "Dense_13": {"kernel": [[0.0]], "bias": [0.0]},
                        "global_vdw_scale": [0.15],
                        "element_vdw_scale": [1.0] * 18,
                        "repulsion": {
                            "a_coefficient": 0.004,
                            "a_exponent": 8.0,
                            "phi_coefficients": [0.18175, 0.50986, 0.28022, 0.02817],
                            "phi_exponents": [3.19980, 0.94229, 0.40290, 0.20162],
                        },
                    }
                },
            }
        )
    )

    report = audit(checkpoint)

    assert report["unit_contract"] == {"distance": "angstrom", "energy": "eV"}
    assert report["cgenff_vdw"]["inferred_predict_atomic_vdw_scale"] is True
    assert report["cgenff_vdw"]["inferred_learn_cgenff_vdw_scale"] is True
    assert any("a_coefficient" in warning for warning in report["warnings"])
    assert any("a_exponent" in warning for warning in report["warnings"])

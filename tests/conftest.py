import sys
import types

# Stub heavy modules to avoid GPU/model deps during tests

# Stub mmml.dcmnet.dcmnet.models to prevent importing e3x/large weights
stub_models = types.ModuleType("mmml.dcmnet.dcmnet.models")
for name in ["DCM1", "DCM2", "DCM3", "DCM4", "DCM5", "DCM6", "DCM7"]:
    setattr(stub_models, name, object())
for name in [
    "dcm1_params",
    "dcm2_params",
    "dcm3_params",
    "dcm4_params",
    "dcm5_params",
    "dcm6_params",
    "dcm7_params",
]:
    setattr(stub_models, name, {})
sys.modules["mmml.dcmnet.dcmnet.models"] = stub_models

# Stub e3x if imported indirectly
sys.modules.setdefault("e3x", types.ModuleType("e3x"))



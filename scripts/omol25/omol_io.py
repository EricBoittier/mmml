"""Read-only OMol25 readers (ASE-LMDB and ColabFit parquet) -> uniform records.

Every reader yields dicts with:
    Z (n,) int32, R (n,3) float64 [A], E float [eV], F (n,3) float64 [eV/A],
    charge int, spin int (OMol25 "spin" == multiplicity 2S+1),
    data_id str (OMol25 subset, e.g. 'biomolecules', 'elytes', 'metal_complexes'),
    source str (raw path string, kept only for provenance),
    dipole (3,) float or None, key str (unique id within the input).

Safety: downloaded data is treated as untrusted.  LMDB envs are opened
readonly/lock=False; records are zlib-compressed JSON (ase_db_backends.aselmdb
writes them with orjson), decoded with json.loads -- *no pickle is ever used*.
Nothing shipped with the data is imported or executed.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import zlib
from typing import Iterator

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEPS = os.path.join(os.path.dirname(_HERE), "pydeps")  # lmdb, pyarrow (pip --target)
if os.path.isdir(_DEPS) and _DEPS not in sys.path:
    sys.path.append(_DEPS)


# ----------------------------------------------------------------------------- ASE-LMDB
def _lmdb_get_json(txn, key: str):
    raw = txn.get(key.encode("ascii"))
    if raw is None:
        return None
    return json.loads(zlib.decompress(raw))


def _np_from_json(x):
    """ase.io.jsonio-style ndarray encodings or plain nested lists."""
    if isinstance(x, dict) and "__ndarray__" in x:
        shape, dtype, flat = x["__ndarray__"]
        return np.asarray(flat, dtype=dtype).reshape(shape)
    return np.asarray(x)


def list_aselmdb_files(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    files = sorted(glob.glob(os.path.join(path, "**", "*.aselmdb"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no *.aselmdb under {path}")
    return files


def aselmdb_count(path: str) -> int:
    import lmdb

    n = 0
    for f in list_aselmdb_files(path):
        env = lmdb.open(f, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin() as txn:
            nextid = _lmdb_get_json(txn, "nextid") or 1
            deleted = _lmdb_get_json(txn, "deleted_ids") or []
            n += int(nextid) - 1 - len(deleted)
        env.close()
    return n


def _record_from_row(dct: dict, key: str) -> dict:
    kvp = dct.get("key_value_pairs") or {}
    data = dct.get("data") or {}
    info = {**data, **kvp}
    Z = _np_from_json(dct["numbers"]).astype(np.int32)
    R = _np_from_json(dct["positions"]).astype(np.float64).reshape(-1, 3)
    E = dct.get("energy", info.get("energy"))
    F = dct.get("forces", info.get("forces"))
    if E is None or F is None:
        raise KeyError(f"{key}: missing energy/forces (keys={sorted(dct)})")
    dip = dct.get("dipole", info.get("dipole"))
    return {
        "Z": Z,
        "R": R,
        "E": float(E),
        "F": _np_from_json(F).astype(np.float64).reshape(-1, 3),
        "charge": int(round(float(info.get("charge", 0)))),
        "spin": int(round(float(info.get("spin", 1)))),
        "data_id": str(info.get("data_id", "unknown")),
        "source": str(info.get("source", "")),
        "dipole": None if dip is None else _np_from_json(dip).astype(np.float64).reshape(3),
        "key": key,
    }


def iter_aselmdb(path: str, *, file_slice: slice | None = None, max_per_file: int | None = None,
                 raw_keys: bool = False) -> Iterator[dict]:
    """Yield records from every *.aselmdb under ``path`` (read-only)."""
    import lmdb

    files = list_aselmdb_files(path)
    if file_slice is not None:
        files = files[file_slice]
    for f in files:
        env = lmdb.open(f, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
        tag = os.path.basename(f)
        with env.begin() as txn:
            nextid = int(_lmdb_get_json(txn, "nextid") or 1)
            deleted = set(_lmdb_get_json(txn, "deleted_ids") or [])
            n = 0
            for i in range(1, nextid):
                if i in deleted:
                    continue
                dct = _lmdb_get_json(txn, str(i))
                if dct is None:
                    continue
                if raw_keys:
                    yield {"_raw_keys": sorted(dct), "_kvp": dct.get("key_value_pairs"),
                           "_data_keys": sorted((dct.get("data") or {}).keys())}
                    return
                yield _record_from_row(dct, f"{tag}:{i}")
                n += 1
                if max_per_file is not None and n >= max_per_file:
                    break
        env.close()


# ----------------------------------------------------------------------------- ColabFit
def iter_colabfit_parquet(path: str, *, max_rows: int | None = None, batch_rows: int = 2048) -> Iterator[dict]:
    """ColabFit OMol25 parquet (energy eV, forces eV/A; charge/spin/data_id in property_metadata JSON)."""
    import pyarrow.parquet as pq

    files = [path] if os.path.isfile(path) else sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))
    cols = ["atomic_numbers", "positions", "energy", "atomic_forces", "multiplicity", "names", "property_metadata"]
    n = 0
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=batch_rows, columns=cols):
            for row in batch.to_pylist():
                md = json.loads(row["property_metadata"] or "{}")
                yield {
                    "Z": np.asarray(row["atomic_numbers"], dtype=np.int32),
                    "R": np.asarray(row["positions"], dtype=np.float64).reshape(-1, 3),
                    "E": float(row["energy"]),
                    "F": np.asarray(row["atomic_forces"], dtype=np.float64).reshape(-1, 3),
                    "charge": int(md.get("charge", 0)),
                    "spin": int(md.get("spin", row["multiplicity"] or 1)),
                    "data_id": str(md.get("data_id", "unknown")),
                    "source": str(md.get("source", "")),
                    "dipole": None,
                    "key": (row["names"] or [f"{os.path.basename(f)}:{n}"])[0],
                }
                n += 1
                if max_rows is not None and n >= max_rows:
                    return


def iter_records(path: str, fmt: str = "auto", **kw) -> Iterator[dict]:
    if fmt == "auto":
        fmt = "parquet" if (path.endswith(".parquet") or glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True)) else "aselmdb"
    if fmt == "parquet":
        yield from iter_colabfit_parquet(path, max_rows=kw.get("max_rows"))
    else:
        mr = kw.get("max_rows")
        it = iter_aselmdb(path, file_slice=kw.get("file_slice"), max_per_file=kw.get("max_per_file"))
        for i, r in enumerate(it):
            if mr is not None and i >= mr:
                return
            yield r


# ----------------------------------------------------------------------------- refs
def load_iso_atom_refs(yaml_path: str, key: str = "omol_elem_refs") -> dict[int, dict[int, float]]:
    """{Z: {atomic charge: E_iso [eV]}} from fairchem references/iso_atom_elem_refs.yaml (yaml.safe_load)."""
    import yaml

    with open(yaml_path) as fh:
        d = yaml.safe_load(fh)
    return {int(z): {int(q): float(e) for q, e in v.items()} for z, v in d[key].items()}


def neutral_ref_table(refs: dict[int, dict[int, float]], zmax: int = 100) -> np.ndarray:
    tab = np.full(zmax + 1, np.nan)
    for z, v in refs.items():
        if 0 in v and z <= zmax:
            tab[z] = v[0]
    return tab

import numpy as np
import pandas as pd
import chemcoord as cc
import importlib
import warnings


def patch_chemcoord_for_pandas3():
    gz = importlib.import_module(
        "chemcoord.cartesian_coordinates._cartesian_class_get_zmat"
    )
    zc = importlib.import_module(
        "chemcoord.internal_coordinates._zmat_class_core"
    )
    idx = importlib.import_module(
        "chemcoord.internal_coordinates._indexers"
    )

    # Resolve internals from the module namespace first, instead of guessing paths
    transformation = getattr(zc, "transformation", None)
    constants = getattr(zc, "constants", None)
    replace_without_warn = getattr(zc, "replace_without_warn", None)

    # Fallbacks if the names are not already present in _zmat_class_core
    if constants is None:
        constants = importlib.import_module("chemcoord.constants")

    if replace_without_warn is None:
        common_methods = importlib.import_module("chemcoord.common_methods")
        replace_without_warn = getattr(common_methods, "replace_without_warn")

    if transformation is None:
        # Try a few likely private-module locations used across versions
        candidates = [
            "chemcoord.internal_coordinates._zmat_transformation",
            "chemcoord.internal_coordinates.zmat_transformation",
            "chemcoord._internal_coordinates._zmat_transformation",
            "chemcoord._internal_coordinates.zmat_transformation",
        ]
        for modname in candidates:
            try:
                transformation = importlib.import_module(modname)
                break
            except ModuleNotFoundError:
                pass

    if transformation is None:
        raise ModuleNotFoundError(
            "Could not locate chemcoord's internal zmat transformation module. "
            "Run `print(dir(importlib.import_module('chemcoord.internal_coordinates._zmat_class_core')))` "
            "to inspect available names."
        )

    # Resolve the Cartesian constructor without assuming exact file layout
    Cartesian_cls = getattr(cc, "Cartesian", None)
    if Cartesian_cls is None:
        cm = importlib.import_module(
            "chemcoord.cartesian_coordinates.cartesian_class_main"
        )
        Cartesian_cls = getattr(cm, "Cartesian")

    # --- Patch 1: make b/a/d object dtype, not strict string dtype ---
    def _build_zmat_patched(self, construction_table):
        c_table = construction_table.copy()

        for col in ["b", "a", "d"]:
            if col in c_table.columns:
                c_table[col] = c_table[col].astype(object)

        zmat_frame = pd.DataFrame(index=c_table.index)

        # atom labels should also avoid pandas string-arrow strictness here
        zmat_frame["atom"] = self.loc[c_table.index, "atom"].astype(object)

        for col in ["b", "a", "d"]:
            zmat_frame[col] = pd.Series(index=c_table.index, dtype=object)

        zmat_frame.loc[:, ["b", "a", "d"]] = (
            c_table.loc[:, ["b", "a", "d"]].astype(object)
        )

        zmat_values = np.asarray(
            self._calculate_zmat_values(c_table),
            dtype=np.float64,
        )
        zmat_frame["bond"] = zmat_values[:, 0]
        zmat_frame["angle"] = zmat_values[:, 1]
        zmat_frame["dihedral"] = zmat_values[:, 2]

        Zmat_cls = getattr(gz, "Zmat", None) or getattr(cc, "Zmat", None)
        if Zmat_cls is None:
            return zmat_frame
        return Zmat_cls(zmat_frame)

    gz.CartesianGetZmat._build_zmat = _build_zmat_patched

    # --- Patch 1b: construction table mixes magic strings (origin, e_z) with int
    # atom indices; pandas 2.2+ may infer StringDtype and reject ints (e.g. "got int64"). ---
    _orig_get_frag_constr_table = gz.CartesianGetZmat._get_frag_constr_table

    def _get_frag_constr_table_patched(self, *args, **kwargs):
        out = _orig_get_frag_constr_table(self, *args, **kwargs)
        out = out.copy()
        for col in ("b", "a", "d"):
            if col in out.columns:
                out[col] = out[col].astype(object)
        return out

    gz.CartesianGetZmat._get_frag_constr_table = _get_frag_constr_table_patched

    # --- Patch 2: force writable numeric arrays in get_cartesian ---
    def get_cartesian_patched(self):
        c_table = self.loc[:, ["b", "a", "d"]].copy()

        c_table = (
            replace_without_warn(c_table, constants.int_label)
            .astype("i8")
            .replace({k: v for v, k in enumerate(c_table.index)})
            .to_numpy(copy=True)
            .T
        )

        C = self.loc[:, ["bond", "angle", "dihedral"]].to_numpy(copy=True).T

        C[[1, 2], :] = np.radians(C[[1, 2], :])

        err, row, positions = transformation.get_X(C, c_table)
        positions = positions.T

        if err:
            raise ValueError(f"Error in row {row} with positions {positions}")

        cartesian = pd.DataFrame(
            positions,
            index=self.index,
            columns=["x", "y", "z"],
        )
        cartesian["atom"] = self["atom"].astype(object)

        return Cartesian_cls(cartesian)

    zc.ZmatCore.get_cartesian = get_cartesian_patched

    # --- Patch 3: pandas 3 now raises TypeError (not FutureWarning) on lossy setitem ---
    def _unsafe_setitem_patched(self, key, value):
        try:
            with warnings.catch_warnings():
                # Treat pandas/chemcoord future/dep warnings as errors so we
                # can handle them in a single place, instead of spamming output.
                warnings.simplefilter("error", category=FutureWarning)
                warnings.simplefilter("error", category=DeprecationWarning)

                indexer = getattr(self.molecule._frame, self.indexer)
                if isinstance(key, tuple):
                    indexer[key[0], key[1]] = value
                else:
                    indexer[key] = value
        except (FutureWarning, DeprecationWarning, TypeError):
            # Mixed assignment (e.g. symbol/string into float columns) now errors in pandas 3.
            # Cast addressed column(s) to object and retry once.
            if isinstance(key, tuple):
                if type(key[1]) is not str and idx.is_iterable(key[1]):
                    self.molecule._frame = self.molecule._frame.astype(
                        {k: "O" for k in key[1]}
                    )
                else:
                    self.molecule._frame = self.molecule._frame.astype({key[1]: "O"})
                indexer = getattr(self.molecule._frame, self.indexer)
                indexer[key[0], key[1]] = value
            else:
                raise TypeError("Assignment not supported.")

    idx._Unsafe_base.__setitem__ = _unsafe_setitem_patched

    # --- Patch 4: get_grad_zmat → apply_grad_zmat_tensor uses numpy str dtypes which
    # become pandas Arrow string columns; they reject mixed int/str construction refs.
    xyz_functions = importlib.import_module(
        "chemcoord.cartesian_coordinates.xyz_functions"
    )

    def apply_grad_zmat_tensor_patched(grad_C, construction_table, cart_dist):
        import sympy

        if (construction_table.index != cart_dist.index).any():
            message = "construction_table and cart_dist must use the same index"
            raise ValueError(message)
        from chemcoord.internal_coordinates.zmat_class_main import Zmat

        dtypes = [
            ("atom", object),
            ("b", object),
            ("bond", float),
            ("a", object),
            ("angle", float),
            ("d", object),
            ("dihedral", float),
        ]
        new = pd.DataFrame(
            np.empty(len(construction_table), dtype=dtypes),
            index=cart_dist.index,
        )

        X_dist = cart_dist.loc[:, ["x", "y", "z"]].values.T
        C_dist = np.tensordot(grad_C, X_dist, axes=([3, 2], [0, 1])).T
        if C_dist.dtype == np.dtype("i8"):
            C_dist = C_dist.astype("f8")
        try:
            C_dist[:, [1, 2]] = np.rad2deg(C_dist[:, [1, 2]])
        except (AttributeError, TypeError):
            C_dist[:, [1, 2]] = sympy.deg(C_dist[:, [1, 2]])
            new = new.astype({k: "O" for k in ["bond", "angle", "dihedral"]})

        ct = construction_table.loc[:, ["b", "a", "d"]].copy()
        for col in ("b", "a", "d"):
            ct[col] = ct[col].astype(object)
        new.loc[:, ["b", "a", "d"]] = ct
        new.loc[:, "atom"] = cart_dist.loc[:, "atom"].astype(object)
        new.loc[:, ["bond", "angle", "dihedral"]] = C_dist
        return Zmat(new, _metadata={"last_valid_cartesian": cart_dist})

    xyz_functions.apply_grad_zmat_tensor = apply_grad_zmat_tensor_patched


patch_chemcoord_for_pandas3()


def wrap_deg(x):
    return (x + 180.0) % 360.0 - 180.0

def interpolate_zmats(z1, z2, steps):
    base = z1.copy()

    c1 = z1.loc[:, ["bond", "angle", "dihedral"]].copy()
    c2 = z2.loc[:, ["bond", "angle", "dihedral"]].copy()

    delta = c2 - c1
    delta["angle"] = wrap_deg(delta["angle"])
    delta["dihedral"] = wrap_deg(delta["dihedral"])

    out = []
    for i in range(steps + 1):
        t = i / steps
        z = base.copy()
        vals = c1 + t * delta

        # avoid zero-length bonds during interpolation
        vals["bond"] = vals["bond"].clip(lower=1e-4)

        z.unsafe_loc[:, ["bond", "angle", "dihedral"]] = vals
        out.append(z)

    return out

def interpolate_xyzs_to_npz(xyz1: str, xyz2: str, steps: int = 1000, out_fn="interpolated.npz"):

    lala_zm = cc.Cartesian.read_xyz(xyz1).get_zmat()
    rala_cc = cc.Cartesian.read_xyz(xyz2)
    c_table = lala_zm.loc[:, ["b", "a", "d"]].copy()
    for col in ("b", "a", "d"):
        c_table[col] = c_table[col].astype(object)
    rala_zm = rala_cc.get_zmat(c_table)
    # interpolate between structures
    mixed = interpolate_zmats(lala_zm, rala_zm, steps)
    mixed_ccs = [_.get_cartesian() for _ in  mixed]
    ase_atoms_list = [
        ase.Atoms(mixed_ccs[i]["atom"], mixed_ccs[i][["x", "y", "z"]])
        for i in range(len(mixed_ccs))
    ]

    # write
    out_dict = {
        "R": [_.get_positions() for _ in ase_atoms_list],
        "Z": [_.get_chemical_symbols() for _ in ase_atoms_list],
        "N": [len(_) for _ in ase_atoms_list],
    }
    np.savez_compressed(out_fn, **out_dict)


import time
import ase
from io import StringIO
import pandas as pd


def sym_to_ase(eq):
    test = ase.io.extxyz.read_xyz(StringIO(eq["sym_mol"].to_xyz()))
    test = next(test)
    return test

def to_chemcord(Z, R):
    x,y,z = R.T
    df = pd.DataFrame({"atom": Z, "x": x.flatten(), "y":y.flatten(), "z": z.flatten()})
    cart = cc.Cartesian(df)
    zmat = cart.get_zmat()
    return cart, zmat


def ase_to_chemcord(atoms):
    return to_chemcord(atoms.get_chemical_symbols(), atoms.get_positions())



def chemcoord_to_ase(cart: cc.Cartesian):
    return ase.Atoms(cart["atom"], cart[["x", "y", "z"]])
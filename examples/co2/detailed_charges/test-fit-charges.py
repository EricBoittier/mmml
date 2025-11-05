#!/usr/bin/env python3
"""
K-fold cross-validated JAX/Flax MLP regressor for grouped (scheme, level, atom_index) data.

- Features: r1, r2, ang
- Target: value
- Train/evaluate separately per (scheme, level, atom_index)
- Uses sklearn KFold, StandardScaler, and metrics
- Minimal JAX/Flax training loop (plain SGD; no optax dependency)

Example usage:
    python cv_fit_jax.py --data data.csv --n-splits 5 --epochs 800 --lr 1e-3

The input CSV/TSV must contain columns:
    r1, r2, ang, level, atom_index, atom, scheme, value
"""

import argparse
import logging
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax import linen as nn
from jax import random
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


logger = logging.getLogger(__name__)


# ---------------------------
# Flax MLP definition
# ---------------------------
class MLP(nn.Module):
    hidden_sizes: tuple = (64, 64)
    @nn.compact
    def __call__(self, x):
        for h in self.hidden_sizes:
            x = nn.relu(nn.Dense(h)(x))
        x = nn.Dense(1)(x)
        return x.squeeze(-1)




def make_update(apply_fn):
    """Create a jitted SGD update that closes over apply_fn (no non-array args to jit)."""
    @jax.jit
    def update(params, X, y, lr):
        def loss_fn(prms, Xb, yb):
            preds = apply_fn(prms, Xb)
            return jnp.mean((preds - yb) ** 2)
        loss_grad = jax.grad(loss_fn)(params, X, y)
        new_params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, loss_grad)
        return new_params
    return update

def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    hidden,
    epochs,
    lr,
    batch_size,
    seed,
    *,
    log_interval=None,
):
    key = random.PRNGKey(seed)
    model = MLP(hidden_sizes=hidden)
    params = model.init(key, jnp.asarray(X_train))

    def apply_fn(prms, X):
        # no extra squeeze here — MLP already returns shape (n,)
        return model.apply(prms, X)

    update = make_update(apply_fn)

    n = X_train.shape[0]
    indices = np.arange(n)
    use_minibatch = batch_size is not None and 0 < batch_size < n

    best = {"params": params, "val_rmse": np.inf}
    log_interval = max(1, log_interval or max(1, epochs // 10))
    loop_start = time.perf_counter()
    last_log = loop_start

    logger.info(
        "Starting training: epochs=%d lr=%.3g batch_size=%s hidden=%s",
        epochs,
        lr,
        batch_size if use_minibatch else "full",
        hidden,
    )

    for epoch in range(1, epochs + 1):
        if use_minibatch:
            np.random.shuffle(indices)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                Xb = jnp.asarray(X_train[indices[start:end]])
                yb = jnp.asarray(y_train[indices[start:end]])
                params = update(params, Xb, yb, lr)
        else:
            params = update(params, jnp.asarray(X_train), jnp.asarray(y_train), lr)

        # track best on val
        preds_val = np.array(apply_fn(params, jnp.asarray(X_val)))
        rmse = float(np.sqrt(mean_squared_error(y_val, preds_val)))
        if rmse < best["val_rmse"]:
            best["val_rmse"] = rmse
            best["params"] = params

        if epoch % log_interval == 0 or epoch == 1 or epoch == epochs:
            preds_train = np.array(apply_fn(params, jnp.asarray(X_train)))
            train_rmse = float(np.sqrt(mean_squared_error(y_train, preds_train)))
            now = time.perf_counter()
            logger.info(
                "Epoch %d/%d: train_rmse=%.4f val_rmse=%.4f (+%.2fs, total %.2fs)",
                epoch,
                epochs,
                train_rmse,
                rmse,
                now - last_log,
                now - loop_start,
            )
            last_log = now

    params = best["params"]
    preds_train = np.array(apply_fn(params, jnp.asarray(X_train)))
    preds_val = np.array(apply_fn(params, jnp.asarray(X_val)))

    metrics = {
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, preds_train))),
        "train_mae": float(mean_absolute_error(y_train, preds_train)),
        "train_r2": float(r2_score(y_train, preds_train)) if len(np.unique(y_train)) > 1 else float("nan"),
        "val_rmse": float(np.sqrt(mean_squared_error(y_val, preds_val))),
        "val_mae": float(mean_absolute_error(y_val, preds_val)),
        "val_r2": float(r2_score(y_val, preds_val)) if len(np.unique(y_val)) > 1 else float("nan"),
    }
    logger.info(
        "Finished training: best_val_rmse=%.4f total_time=%.2fs",
        metrics["val_rmse"],
        time.perf_counter() - loop_start,
    )
    return metrics, params

def run_group_cv(df_group, n_splits, epochs, lr, batch_size, hidden, seed):
    # Extract features/target
    X = df_group[["r1", "r2", "ang"]].to_numpy(dtype=np.float64)
    y = df_group["value"].to_numpy(dtype=np.float64)

    # Decide CV strategy
    if len(df_group) < 2:
        return None  # Not enough data to validate anything
    if n_splits is None:
        # Auto: KFold with min(n, 5) if allowed, else LOO
        if len(df_group) >= 5:
            kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        elif len(df_group) >= 3:
            kf = KFold(n_splits=len(df_group), shuffle=True, random_state=seed)  # LOOCV-equivalent
        else:
            kf = LeaveOneOut()
    else:
        if len(df_group) >= n_splits:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        else:
            # fallback to LeaveOneOut when not enough samples
            kf = LeaveOneOut()

    fold_metrics = []
    # Cross-validation
    for fold, (tr, va) in enumerate(kf.split(X), start=1):
        X_tr, X_va = X[tr], X[va]
        y_tr, y_va = y[tr], y[va]

        # Scale features per-fold using train only
        xs = StandardScaler().fit(X_tr)
        X_trs = xs.transform(X_tr).astype(np.float32)
        X_vas = xs.transform(X_va).astype(np.float32)

        # (Optional) you can also standardize y; here we keep y as-is
        metrics, _ = train_model(
            X_trs, y_tr, X_vas, y_va,
            hidden=hidden, epochs=epochs, lr=lr, batch_size=batch_size, seed=seed + fold
        )
        fold_metrics.append(metrics)

    # Aggregate
    agg = {
        "val_rmse_mean": float(np.mean([m["val_rmse"] for m in fold_metrics])),
        "val_rmse_std": float(np.std([m["val_rmse"] for m in fold_metrics], ddof=1)) if len(fold_metrics) > 1 else 0.0,
        "val_mae_mean": float(np.mean([m["val_mae"] for m in fold_metrics])),
        "val_r2_mean": float(np.nanmean([m["val_r2"] for m in fold_metrics])),
        "train_rmse_mean": float(np.mean([m["train_rmse"] for m in fold_metrics])),
        "train_mae_mean": float(np.mean([m["train_mae"] for m in fold_metrics])),
        "train_r2_mean": float(np.nanmean([m["train_r2"] for m in fold_metrics])),
        "n_folds": len(fold_metrics),
        "n_samples": len(df_group),
    }
    return agg


def run_sklearn_models_cv(df_group, n_splits, seed):
    X = df_group[["r1", "r2", "ang"]].to_numpy(dtype=float)
    y = df_group["value"].to_numpy(dtype=float)

    if len(df_group) < 2:
        return None

    # Cross-validation strategy
    if n_splits is None:
        if len(df_group) >= 5:
            kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        else:
            kf = LeaveOneOut()
    else:
        if len(df_group) >= n_splits:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        else:
            kf = LeaveOneOut()

    # Models to benchmark
    models = {
        "KNN": KNeighborsRegressor(n_neighbors=3),
        "SVR": SVR(kernel="rbf", C=1.0, epsilon=0.1),
        "DecisionTree": DecisionTreeRegressor(random_state=seed),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=seed),
        "GradientBoosting": GradientBoostingRegressor(random_state=seed),
    }

    results = []

    for name, model in models.items():
        fold_metrics = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Standardize inputs
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_val = scaler.transform(X_val)

            model.fit(X_train, y_train)
            pred = model.predict(X_val)

            fold_metrics.append({
                "val_rmse": np.sqrt(mean_squared_error(y_val, pred)),
                "val_mae": mean_absolute_error(y_val, pred),
                "val_r2": r2_score(y_val, pred) if len(np.unique(y_val)) > 1 else np.nan,
            })

        agg = {
            "model": name,
            "val_rmse_mean": float(np.mean([m["val_rmse"] for m in fold_metrics])),
            "val_rmse_std": float(np.std([m["val_rmse"] for m in fold_metrics], ddof=1))
                              if len(fold_metrics) > 1 else 0.0,
            "val_mae_mean": float(np.mean([m["val_mae"] for m in fold_metrics])),
            "val_r2_mean": float(np.mean([m["val_r2"] for m in fold_metrics])),
            "n_samples": len(df_group),
            "n_folds": len(fold_metrics),
        }
        results.append(agg)

    return results



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to CSV/TSV file with columns r1,r2,ang,level,atom_index,atom,scheme,value")
    ap.add_argument("--sep", type=str, default=None, help="Pandas separator (auto by extension if None). Common: ',' or '\\t'")
    ap.add_argument("--n-splits", type=int, default=5, help="K for KFold (fallback to LOO if not enough samples). Use <=1 to auto-select.")
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=0, help="Minibatch size; <=0 means full-batch")
    ap.add_argument("--hidden", type=str, default="64,64", help="Comma-separated hidden sizes, e.g., '64,64' or '128,64,32'")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Load data
    if args.sep is None:
        # try to infer by extension
        if args.data.endswith(".tsv") or args.data.endswith(".tab"):
            sep = "\t"
        else:
            sep = ","
    else:
        sep = args.sep

    try:
        df = pd.read_csv(args.data, sep=sep)
    except Exception as e:
        print(f"Failed to read {args.data}: {e}", file=sys.stderr)
        sys.exit(1)

    required_cols = {"r1", "r2", "ang", "level", "atom_index", "atom", "scheme", "value"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)

    # Normalize dtypes
    df["atom_index"] = df["atom_index"].astype(int)
    # Ensure correct numeric types
    for c in ["r1", "r2", "ang", "value"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["r1", "r2", "ang", "value", "scheme", "level", "atom_index"])
    if len(df) == 0:
        print("No valid rows after cleaning.", file=sys.stderr)
        sys.exit(1)

    hidden = tuple(int(x) for x in args.hidden.split(",") if x.strip())

    # Group by (scheme, level, atom_index)
    results = []
    grouped = df.groupby(["scheme", "level", "atom_index"], sort=True)
    for (scheme, level, atom_idx), df_g in grouped:
        # Original JAX model metrics
        agg = run_group_cv(
            df_g,
            n_splits=(args.n_splits if args.n_splits > 1 else None),
            epochs=args.epochs,
            lr=args.lr,
            batch_size=(args.batch_size if args.batch_size and args.batch_size > 0 else None),
            hidden=hidden,
            seed=args.seed,
        )

        # Classical ML results
        sk_results = run_sklearn_models_cv(
            df_g,
            n_splits=(args.n_splits if args.n_splits > 1 else None),
            seed=args.seed,
        )

        if agg is not None:
            agg["model"] = "FlaxMLP"
            agg["scheme"] = scheme
            agg["level"] = level
            agg["atom_index"] = atom_idx
            results.append(agg)

        if sk_results is not None:
            for r in sk_results:
                r["scheme"] = scheme
                r["level"] = level
                r["atom_index"] = atom_idx
                results.append(r)
        if agg is None:
            res = {
                "scheme": scheme,
                "level": level,
                "atom_index": int(atom_idx),
                "n_samples": len(df_g),
                "n_folds": 0,
                "val_rmse_mean": np.nan,
                "val_rmse_std": np.nan,
                "val_mae_mean": np.nan,
                "val_r2_mean": np.nan,
                "train_rmse_mean": np.nan,
                "train_mae_mean": np.nan,
                "train_r2_mean": np.nan,
                "note": "Not enough samples for CV",
            }
        else:
            res = {
                "scheme": scheme,
                "level": level,
                "atom_index": int(atom_idx),
                **agg,
                "note": "",
            }
        results.append(res)

    out = pd.DataFrame(results).sort_values(["scheme", "level", "atom_index"])
    # Pretty print
    with pd.option_context("display.max_columns", None):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()

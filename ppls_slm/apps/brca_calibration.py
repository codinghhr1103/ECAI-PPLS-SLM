"""Calibration of PPLS-SLM predictive credible intervals on BRCA TCGA data.

This script uses the same 5-fold split as `ppls_slm.apps.brca_prediction` and
computes empirical coverage of element-wise credible intervals for
alpha in {0.05,0.10,0.15,0.20,0.25}.

By default, it reads the best r (CV-MSE optimal) for PPLS-SLM from:
  results_prediction_brca/brca_prediction_summary.csv
and then runs calibration at that r.

Outputs
-------
- brca_calibration_table.csv: rows=alpha, columns=[Expected, Fold1..Fold5, Mean]
"""

from __future__ import annotations

import argparse
import io
import os
import zipfile
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ppls_slm.algorithms import InitialPointGenerator, ScalarLikelihoodMethod
from ppls_slm.apps.prediction import (
    compute_credible_intervals,
    empirical_coverage,
    predict_conditional_covariance,
    predict_conditional_mean,
)


def load_brca_combined_raw(path: str) -> Tuple[np.ndarray, np.ndarray]:
    lower = path.lower()

    if lower.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            if not names:
                raise ValueError(f"Empty zip file: {path}")
            data = z.read(names[0])
            df = pd.read_csv(io.BytesIO(data))
    else:
        df = pd.read_csv(path)

    rs_cols = [c for c in df.columns if str(c).startswith("rs_")]
    pp_cols = [c for c in df.columns if str(c).startswith("pp_")]
    if not rs_cols or not pp_cols:
        raise ValueError(
            "BRCA combined dataset must contain `rs_` (genes) and `pp_` (proteins) columns. "
            f"Found rs_={len(rs_cols)}, pp_={len(pp_cols)}"
        )

    X = df[rs_cols].to_numpy(dtype=float)
    Y = df[pp_cols].to_numpy(dtype=float)

    good_rows = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    X = X[good_rows]
    Y = Y[good_rows]

    good_x_cols = np.isfinite(X).all(axis=0)
    good_y_cols = np.isfinite(Y).all(axis=0)
    X = X[:, good_x_cols]
    Y = Y[:, good_y_cols]

    return X, Y


def _standardize_train_test(X_train, Y_train, X_test, Y_test):
    from sklearn.preprocessing import StandardScaler

    sx = StandardScaler()
    sy = StandardScaler()

    X_train_s = sx.fit_transform(X_train)
    Y_train_s = sy.fit_transform(Y_train)
    X_test_s = sx.transform(X_test)
    Y_test_s = sy.transform(Y_test)

    return X_train_s, Y_train_s, X_test_s, Y_test_s, sx, sy


def _unstandardize_y(y_s, sy):
    return sy.inverse_transform(y_s)


def _unstandardize_cov(Cov_s, sy):
    scale = getattr(sy, "scale_", None)
    if scale is None:
        return Cov_s
    D = np.diag(np.asarray(scale, dtype=float))
    return D @ Cov_s @ D


def _fit_slm(X_train_s, Y_train_s, *, r: int, n_starts: int, seed: int, max_iter: int) -> Dict:
    p, q = X_train_s.shape[1], Y_train_s.shape[1]

    init_gen = InitialPointGenerator(p=p, q=q, r=r, n_starts=n_starts, random_seed=seed)
    starting_points = init_gen.generate_starting_points()

    slm = ScalarLikelihoodMethod(p=p, q=q, r=r, max_iter=int(max_iter), use_noise_preestimation=True)
    res = slm.fit(X_train_s, Y_train_s, starting_points)

    return {
        "W": res["W"],
        "C": res["C"],
        "B": res["B"],
        "Sigma_t": res["Sigma_t"],
        "sigma_e2": res["sigma_e2"],
        "sigma_f2": res["sigma_f2"],
        "sigma_h2": res["sigma_h2"],
    }


def _best_r_from_summary(path: str) -> int:
    df = pd.read_csv(path)
    sub = df[df["method"].astype(str) == "PPLS-SLM"]
    if sub.empty:
        raise ValueError(f"PPLS-SLM row not found in: {path}")
    r = sub.iloc[0]["r"]
    return int(r)


def run_calibration(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    r: int,
    n_folds: int,
    seed: int,
    n_starts: int,
    max_iter: int,
    alphas: List[float],
) -> pd.DataFrame:
    N = X.shape[0]
    rng = np.random.RandomState(seed)
    indices = rng.permutation(N)
    folds = np.array_split(indices, int(n_folds))

    # Table with rows alpha, cols expected + fold1..foldK + mean
    rows = []
    for a in alphas:
        row = {
            "Alpha": float(a),
            "Expected Coverage": f"{100.0 * (1.0 - float(a)):.2f}%",
        }
        covs = []

        for fold_idx, test_idx in enumerate(folds):
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_idx])
            X_train, Y_train = X[train_idx], Y[train_idx]
            X_test, Y_test = X[test_idx], Y[test_idx]

            X_train_s, Y_train_s, X_test_s, _Y_test_s, _sx, sy = _standardize_train_test(X_train, Y_train, X_test, Y_test)
            params = _fit_slm(X_train_s, Y_train_s, r=r, n_starts=n_starts, seed=seed + fold_idx, max_iter=max_iter)

            y_pred_s = predict_conditional_mean(X_test_s, params)
            Cov_s = predict_conditional_covariance(params)

            y_pred = _unstandardize_y(y_pred_s, sy)
            Cov = _unstandardize_cov(Cov_s, sy)

            lower, upper = compute_credible_intervals(y_pred, Cov, alpha=float(a))
            cov_pct = 100.0 * empirical_coverage(Y_test, lower, upper)
            covs.append(float(cov_pct))
            row[f"Fold {fold_idx + 1}"] = f"{cov_pct:.2f}%"

        row["Mean"] = f"{np.mean(covs):.2f}%"
        rows.append(row)

    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser(description="BRCA calibration (PPLS-SLM)")
    p.add_argument("--config", type=str, required=True, help="Path to config JSON (single source of truth)")
    return p.parse_args()


def main():
    args = parse_args()

    from ppls_slm.experiment_config import (
        coerce_float,
        coerce_int,
        get_experiment_cfg,
        load_config,
        require_keys,
    )

    cfg = load_config(args.config)
    calib_cfg = get_experiment_cfg(cfg, "calibration_brca")

    require_keys(
        calib_cfg,
        [
            "thread_limit",
            "brca_data",
            "output_dir",
            "prediction_summary",
            "seed",
            "n_folds",
            "r",
            "n_starts",
            "max_iter",
        ],
        ctx="experiments.calibration_brca",
    )

    for k in ("thread_limit", "seed", "n_folds", "n_starts", "max_iter"):
        coerce_int(calib_cfg, k, ctx="experiments.calibration_brca")

    output_dir = str(calib_cfg["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    # Runtime thread limiting
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=int(calib_cfg["thread_limit"]))
    except Exception:
        pass

    # r=null means "read best r from summary"
    r_val = calib_cfg.get("r")
    if r_val is None:
        r = _best_r_from_summary(str(calib_cfg["prediction_summary"]))
    else:
        r = int(r_val)

    X, Y = load_brca_combined_raw(str(calib_cfg["brca_data"]))
    print(f"Loaded BRCA: X={X.shape}, Y={Y.shape}")
    print(f"Using r={r} for calibration")

    alphas = [0.05, 0.10, 0.15, 0.20, 0.25]
    table = run_calibration(
        X,
        Y,
        r=r,
        n_folds=int(calib_cfg["n_folds"]),
        seed=int(calib_cfg["seed"]),
        n_starts=int(calib_cfg["n_starts"]),
        max_iter=int(calib_cfg["max_iter"]),
        alphas=alphas,
    )

    out_csv = os.path.join(output_dir, "brca_calibration_table.csv")
    table.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


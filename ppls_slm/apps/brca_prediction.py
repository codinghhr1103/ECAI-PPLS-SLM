"""Real-data prediction benchmark on TCGA-BRCA (Section 8.3 extension).

This script evaluates predictive accuracy on the BRCA TCGA paired multi-omics dataset:
- X: gene expression (N=705, p=604)
- Y: protein expression (N=705, q=223)

Protocol
--------
- 5-fold CV with a fixed random seed
- Per-fold standardisation (fit on train, apply to test)
- Methods:
    * PPLS-SLM (multi-start, spectral noise pre-estimation)
    * PPLS-EM
    * PLS regression (PLSR)
    * Ridge regression (RidgeCV)
- Latent dimension grid: r in {3,5,8,10} for PPLS-SLM/EM/PLSR (Ridge uses '-')

Outputs
-------
Writes CSVs under --output_dir (default: results_prediction_brca):
- brca_prediction_per_fold.csv   : fold-level metrics for all methods and r values
- brca_prediction_by_r.csv       : aggregated mean/std for each (method,r)
- brca_prediction_summary.csv    : best-r summary per method (r* minimising CV-MSE)
"""

from __future__ import annotations

import argparse
import io
import os
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ppls_slm.algorithms import EMAlgorithm, InitialPointGenerator, ScalarLikelihoodMethod
from ppls_slm.apps.prediction_baselines import compute_regression_metrics, run_plsr_prediction, run_ridge_prediction
from ppls_slm.apps.prediction import (
    compute_credible_intervals,
    empirical_coverage,
    predict_conditional_covariance,
    predict_conditional_mean,
)


def load_brca_combined_raw(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load the bundled BRCA combined dataset without global standardisation."""
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

    # Drop rows/cols with NaN (defensive; bundled data is usually clean)
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


def _fit_ppls_slm(X_train_s, Y_train_s, *, r: int, n_starts: int, seed: int, max_iter: int) -> Dict:
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


def _fit_ppls_em(X_train_s, Y_train_s, *, r: int, n_starts: int, seed: int, max_iter: int, tol: float) -> Dict:
    p, q = X_train_s.shape[1], Y_train_s.shape[1]

    init_gen = InitialPointGenerator(p=p, q=q, r=r, n_starts=n_starts, random_seed=seed)
    starting_points = init_gen.generate_starting_points()

    em = EMAlgorithm(p=p, q=q, r=r, max_iter=int(max_iter), tolerance=float(tol))
    res = em.fit(X_train_s, Y_train_s, starting_points)

    return {
        "W": res["W"],
        "C": res["C"],
        "B": res["B"],
        "Sigma_t": res["Sigma_t"],
        "sigma_e2": res["sigma_e2"],
        "sigma_f2": res["sigma_f2"],
        "sigma_h2": res["sigma_h2"],
    }


def _predict_ppls(X_test_s, params: Dict):
    y_pred_s = predict_conditional_mean(X_test_s, params)
    Cov_s = predict_conditional_covariance(params)
    return y_pred_s, Cov_s


def run_brca_prediction(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    r_grid: List[int],
    n_folds: int,
    seed: int,
    slm_n_starts: int,
    slm_max_iter: int,
    em_n_starts: int,
    em_max_iter: int,
    em_tol: float,
) -> pd.DataFrame:
    N = X.shape[0]
    rng = np.random.RandomState(seed)
    indices = rng.permutation(N)
    folds = np.array_split(indices, int(n_folds))

    rows: List[Dict] = []

    # Ridge does not depend on r
    for fold_idx, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_idx])
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_test, Y_test = X[test_idx], Y[test_idx]

        ridge = run_ridge_prediction(X_train, Y_train, X_test, Y_test)
        m = ridge["metrics"]
        rows.append({"method": "Ridge", "r": "-", "fold": fold_idx + 1, "mse": m.mse, "mae": m.mae, "r2": m.r2_mean})

    # Methods that depend on r
    for r in r_grid:
        for fold_idx, test_idx in enumerate(folds):
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_idx])
            X_train, Y_train = X[train_idx], Y[train_idx]
            X_test, Y_test = X[test_idx], Y[test_idx]

            # --- PPLS-SLM ---
            X_train_s, Y_train_s, X_test_s, _Y_test_s, _sx, sy = _standardize_train_test(X_train, Y_train, X_test, Y_test)
            slm_params = _fit_ppls_slm(X_train_s, Y_train_s, r=r, n_starts=slm_n_starts, seed=seed + fold_idx, max_iter=slm_max_iter)
            y_pred_s, _Cov_s = _predict_ppls(X_test_s, slm_params)
            y_pred = _unstandardize_y(y_pred_s, sy)
            m = compute_regression_metrics(Y_test, y_pred)
            rows.append({"method": "PPLS-SLM", "r": int(r), "fold": fold_idx + 1, "mse": m.mse, "mae": m.mae, "r2": m.r2_mean})

            # --- PPLS-EM ---
            em_params = _fit_ppls_em(X_train_s, Y_train_s, r=r, n_starts=em_n_starts, seed=seed + fold_idx, max_iter=em_max_iter, tol=em_tol)
            y_pred_s, _Cov_s = _predict_ppls(X_test_s, em_params)
            y_pred = _unstandardize_y(y_pred_s, sy)
            m = compute_regression_metrics(Y_test, y_pred)
            rows.append({"method": "PPLS-EM", "r": int(r), "fold": fold_idx + 1, "mse": m.mse, "mae": m.mae, "r2": m.r2_mean})

            # --- PLSR ---
            plsr = run_plsr_prediction(X_train, Y_train, X_test, Y_test, n_components=r)
            m = plsr["metrics"]
            rows.append({"method": "PLSR", "r": int(r), "fold": fold_idx + 1, "mse": m.mse, "mae": m.mae, "r2": m.r2_mean})

    return pd.DataFrame(rows)


def _aggregate_by_r(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (method, r), sub in df.groupby(["method", "r"], sort=False):
        out.append(
            {
                "method": method,
                "r": r,
                "mse_mean": float(sub["mse"].mean()),
                "mse_std": float(sub["mse"].std(ddof=1)),
                "mae_mean": float(sub["mae"].mean()),
                "mae_std": float(sub["mae"].std(ddof=1)),
                "r2_mean": float(sub["r2"].mean()),
                "r2_std": float(sub["r2"].std(ddof=1)),
            }
        )
    return pd.DataFrame(out)


def _select_best_r(df_by_r: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, sub in df_by_r.groupby("method", sort=False):
        if method == "Ridge":
            best = sub.iloc[0]
        else:
            # r stored as int for these methods
            sub2 = sub.copy()
            sub2["r_int"] = sub2["r"].astype(int)
            best = sub2.sort_values(["mse_mean", "r_int"], ascending=[True, True]).iloc[0]
        rows.append(best.drop(labels=[c for c in ("r_int",) if c in best.index]))
    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser(description="BRCA prediction benchmark (5-fold CV)")
    p.add_argument("--brca_data", type=str, default="application/brca_data_w_subtypes.csv.zip")
    p.add_argument("--output_dir", type=str, default="results_prediction_brca")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--r_grid", type=str, default="3,5,8,10")

    p.add_argument("--slm_n_starts", type=int, default=8)
    p.add_argument("--slm_max_iter", type=int, default=50)

    p.add_argument("--em_n_starts", type=int, default=8)
    p.add_argument("--em_max_iter", type=int, default=200)
    p.add_argument("--em_tol", type=float, default=1e-4)

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    r_grid = [int(x.strip()) for x in str(args.r_grid).split(",") if x.strip()]

    X, Y = load_brca_combined_raw(args.brca_data)
    print(f"Loaded BRCA: X={X.shape}, Y={Y.shape}")

    df = run_brca_prediction(
        X,
        Y,
        r_grid=r_grid,
        n_folds=args.n_folds,
        seed=args.seed,
        slm_n_starts=args.slm_n_starts,
        slm_max_iter=args.slm_max_iter,
        em_n_starts=args.em_n_starts,
        em_max_iter=args.em_max_iter,
        em_tol=args.em_tol,
    )

    df.to_csv(os.path.join(args.output_dir, "brca_prediction_per_fold.csv"), index=False)

    df_by_r = _aggregate_by_r(df)
    df_by_r.to_csv(os.path.join(args.output_dir, "brca_prediction_by_r.csv"), index=False)

    df_best = _select_best_r(df_by_r)
    df_best.to_csv(os.path.join(args.output_dir, "brca_prediction_summary.csv"), index=False)

    print("\nSaved:")
    print(f"  {args.output_dir}/brca_prediction_per_fold.csv")
    print(f"  {args.output_dir}/brca_prediction_by_r.csv")
    print(f"  {args.output_dir}/brca_prediction_summary.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

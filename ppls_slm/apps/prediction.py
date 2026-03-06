"""Application 2: Prediction with Uncertainty Quantification (Section 8.3)

This module reproduces and extends the prediction experiments from the paper
"Scalar Likelihood Method for Probabilistic Partial Least Squares Model with Rank n Update".

Two experiment tracks are supported:

A) Synthetic PPLS data (paper's main prediction sandbox)
   - p=q=100, r=20, N=120 (defaults; reduced for faster reproducible runs)
   - 5-fold CV
   - Compare predictive accuracy across four methods:

        * PPLS-SLM (fitted per fold)
        * PPLS-EM  (fitted per fold)
        * Classical PLS regression (PLSR)
        * Ridge regression (RidgeCV)
   - For PPLS-SLM and PPLS-EM, additionally evaluate calibration of credible intervals.

B) (Implemented in separate scripts) Real BRCA TCGA data prediction and calibration.

Evaluation protocol
-------------------
- All methods use the same CV folds (fixed RNG seed) and the same standardisation flow:
  per fold, fit a z-score transform on the training split and apply it to the test split.
- Accuracy metrics are evaluated on the original Y scale:
    MSE, MAE, and mean R2 across output dimensions.
- PPLS credible intervals are constructed element-wise using the predictive covariance.

Usage
-----
    python -m ppls_slm.apps.prediction --config config.json

Configuration
-------------
All hyperparameters and output paths are read from a single JSON config file.
See `config.json` under `experiments.prediction`.

"""

from __future__ import annotations

import argparse
import os
import warnings
from typing import Dict, List, Optional, Tuple

# NOTE (Windows stability): some BLAS/LAPACK builds may hang or become extremely slow
# on QR/SVD due to oversubscription or thread deadlocks. Defaulting to 1 thread makes
# the synthetic-data generation and optimisation steps deterministic and robust.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy import stats


from ppls_slm.algorithms import EMAlgorithm, InitialPointGenerator, ScalarLikelihoodMethod
from ppls_slm.apps.prediction_baselines import compute_regression_metrics, run_plsr_prediction, run_ridge_prediction
from ppls_slm.ppls_model import PPLSModel


# ─────────────────────────────────────────────────────────────────────────────
#  Data generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_ppls_data(
    p: int = 200,
    q: int = 200,
    r: int = 50,
    n_samples: int = 100,
    sigma_e2: float = 0.1,
    sigma_f2: float = 0.1,
    sigma_h2: float = 0.05,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Generate synthetic PPLS data for the prediction experiment."""
    rng = np.random.RandomState(seed)

    # Random orthonormal loading matrices
    W, _ = np.linalg.qr(rng.randn(p, r))
    C, _ = np.linalg.qr(rng.randn(q, r))

    # Decreasing signals so identifiability holds: theta_t2[i] * b[i] decreasing
    theta_t2 = np.linspace(1.5, 0.3, r)
    b = np.linspace(2.0, 0.5, r)
    assert np.all(np.diff(theta_t2 * b) < 0), "Identifiability violated."

    B = np.diag(b)
    Sigma_t = np.diag(theta_t2)

    model = PPLSModel(p, q, r)
    np.random.seed(seed + 1)  # reproducible sampling
    X, Y = model.sample(n_samples, W, C, B, Sigma_t, sigma_e2, sigma_f2, sigma_h2)

    true_params = {
        "W": W,
        "C": C,
        "B": B,
        "Sigma_t": Sigma_t,
        "sigma_e2": sigma_e2,
        "sigma_f2": sigma_f2,
        "sigma_h2": sigma_h2,
    }
    return X, Y, true_params


# ─────────────────────────────────────────────────────────────────────────────
#  Prediction helpers (Property 1 / Algorithm 3)
# ─────────────────────────────────────────────────────────────────────────────

def predict_conditional_mean(x_new: np.ndarray, params: Dict) -> np.ndarray:
    """Compute E[y_new | x_new, params] via the conditional Gaussian formula."""
    W, C, B = params["W"], params["C"], params["B"]
    Sigma_t = params["Sigma_t"]
    sigma_e2 = float(params["sigma_e2"])

    Sigma_xx = W @ Sigma_t @ W.T + sigma_e2 * np.eye(W.shape[0])

    L = np.linalg.cholesky(Sigma_xx + 1e-9 * np.eye(Sigma_xx.shape[0]))
    tmp = np.linalg.solve(L, W)
    tmp = np.linalg.solve(L.T, tmp)  # Sigma_xx^{-1} W

    A = C @ B @ Sigma_t @ tmp.T  # (q, p)

    if x_new.ndim == 1:
        return A @ x_new
    return x_new @ A.T


def predict_conditional_covariance(params: Dict) -> np.ndarray:
    """Compute Cov[y_new | x_new, params]. The covariance is x-independent."""
    W, C, B = params["W"], params["C"], params["B"]
    Sigma_t = params["Sigma_t"]
    sigma_e2 = float(params["sigma_e2"])
    sigma_f2 = float(params["sigma_f2"])
    sigma_h2 = float(params["sigma_h2"])

    r = W.shape[1]
    q = C.shape[0]

    b = np.diag(B)
    theta_t2 = np.diag(Sigma_t)

    Sigma_xx = W @ Sigma_t @ W.T + sigma_e2 * np.eye(W.shape[0])
    L = np.linalg.cholesky(Sigma_xx + 1e-9 * np.eye(Sigma_xx.shape[0]))
    WtSig_inv = np.linalg.solve(L.T, np.linalg.solve(L, W))  # Sigma_xx^{-1} W (p,r)

    B2Sigma_t = np.diag(b**2 * theta_t2)
    term1 = C @ (B2Sigma_t + sigma_h2 * np.eye(r)) @ C.T + sigma_f2 * np.eye(q)

    K = W.T @ WtSig_inv
    K = (K + K.T) / 2
    M = C @ B @ Sigma_t @ K @ Sigma_t @ B @ C.T

    Cov = term1 - M
    Cov = (Cov + Cov.T) / 2 + 1e-9 * np.eye(q)
    return Cov


def compute_credible_intervals(
    y_pred: np.ndarray,
    Cov_yx: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Element-wise symmetric credible intervals: mean ± z * sqrt(diag(Cov))."""
    z = stats.norm.ppf(1 - alpha / 2)
    std = np.sqrt(np.diag(Cov_yx))
    lower = y_pred - z * std[np.newaxis, :]
    upper = y_pred + z * std[np.newaxis, :]
    return lower, upper


def empirical_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of (test-sample, feature) entries within [lower, upper]."""
    within = (y_true >= lower) & (y_true <= upper)
    return float(within.mean())


# ─────────────────────────────────────────────────────────────────────────────
#  Fold-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _standardize_train_test(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
):
    from sklearn.preprocessing import StandardScaler

    sx = StandardScaler()
    sy = StandardScaler()

    X_train_s = sx.fit_transform(X_train)
    Y_train_s = sy.fit_transform(Y_train)
    X_test_s = sx.transform(X_test)
    Y_test_s = sy.transform(Y_test)

    return X_train_s, Y_train_s, X_test_s, Y_test_s, sx, sy


def _unstandardize_y_pred(y_pred_s: np.ndarray, sy) -> np.ndarray:
    return sy.inverse_transform(y_pred_s)


def _unstandardize_cov_y(Cov_s: np.ndarray, sy) -> np.ndarray:
    # If y_s = (y - mean)/scale, then Cov(y) = D * Cov(y_s) * D, D = diag(scale)
    scale = getattr(sy, "scale_", None)
    if scale is None:
        return Cov_s
    D = np.diag(np.asarray(scale, dtype=float))
    return D @ Cov_s @ D


def _fit_ppls_params_slm(
    X_train_s: np.ndarray,
    Y_train_s: np.ndarray,
    *,
    r: int,
    n_starts: int,
    seed: int,
    slm_max_iter: int,
) -> Dict:
    p, q = X_train_s.shape[1], Y_train_s.shape[1]

    init_gen = InitialPointGenerator(p=p, q=q, r=r, n_starts=n_starts, random_seed=seed)
    starting_points = init_gen.generate_starting_points()

    slm = ScalarLikelihoodMethod(
        p=p,
        q=q,
        r=r,
        max_iter=int(slm_max_iter),
        use_noise_preestimation=True,
    )
    res = slm.fit(X_train_s, Y_train_s, starting_points)

    return {
        "W": res["W"],
        "C": res["C"],
        "B": res["B"],
        "Sigma_t": res["Sigma_t"],
        "sigma_e2": res["sigma_e2"],
        "sigma_f2": res["sigma_f2"],
        "sigma_h2": res["sigma_h2"],
        "_meta": {"n_iterations": res.get("n_iterations"), "success": res.get("success")},
    }


def _fit_ppls_params_em(
    X_train_s: np.ndarray,
    Y_train_s: np.ndarray,
    *,
    r: int,
    n_starts: int,
    seed: int,
    em_max_iter: int,
    em_tol: float,
) -> Dict:
    p, q = X_train_s.shape[1], Y_train_s.shape[1]

    init_gen = InitialPointGenerator(p=p, q=q, r=r, n_starts=n_starts, random_seed=seed)
    starting_points = init_gen.generate_starting_points()

    em = EMAlgorithm(p=p, q=q, r=r, max_iter=int(em_max_iter), tolerance=float(em_tol))
    res = em.fit(X_train_s, Y_train_s, starting_points)

    return {
        "W": res["W"],
        "C": res["C"],
        "B": res["B"],
        "Sigma_t": res["Sigma_t"],
        "sigma_e2": res["sigma_e2"],
        "sigma_f2": res["sigma_f2"],
        "sigma_h2": res["sigma_h2"],
        "_meta": {"n_iterations": res.get("n_iterations"), "log_likelihood": res.get("log_likelihood")},
    }


def _predict_ppls(
    X_test_s: np.ndarray,
    *,
    params: Dict,
):
    y_pred_s = predict_conditional_mean(X_test_s, params)
    Cov_s = predict_conditional_covariance(params)
    return y_pred_s, Cov_s


# ─────────────────────────────────────────────────────────────────────────────
#  k-fold benchmark (Synthetic)
# ─────────────────────────────────────────────────────────────────────────────

def kfold_prediction_benchmark(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    r: int,
    n_folds: int,
    n_starts: int,
    seed: int,
    slm_max_iter: int,
    em_max_iter: int,
    em_tol: float,
    include_baselines: bool,
    alphas: Optional[List[float]] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import time

    """Run 5-fold CV and return:

    - metrics_per_fold: long table (fold, method, MSE/MAE/R2)
    - metrics_summary : per method mean±std
    - calib_summary   : per alpha expected + (SLM/EM) mean±std (percent)
    """
    if alphas is None:
        alphas = [0.05, 0.10, 0.15, 0.20, 0.25]

    N = X.shape[0]
    rng = np.random.RandomState(seed)
    indices = rng.permutation(N)
    folds = np.array_split(indices, int(n_folds))

    metrics_rows: List[Dict] = []
    calib_rows: List[Dict] = []

    for fold_idx, test_idx in enumerate(folds):
        if n_folds <= 1:
            train_idx = indices
        else:
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_idx])

        X_train, Y_train = X[train_idx], Y[train_idx]
        X_test, Y_test = X[test_idx], Y[test_idx]

        if verbose:
            print(
                f"  Fold {fold_idx+1}/{n_folds}  (train={len(train_idx)}, test={len(test_idx)})",
                flush=True,
            )

        # Shared fold standardisation for PPLS models
        X_train_s, Y_train_s, X_test_s, _Y_test_s, _sx, sy = _standardize_train_test(
            X_train, Y_train, X_test, Y_test
        )

        # --- PPLS-SLM ---
        if verbose:
            print(f"    Fitting PPLS-SLM (starts={n_starts}, max_iter={slm_max_iter})...", flush=True)
        _t_fit = time.time()
        slm_params = _fit_ppls_params_slm(
            X_train_s,
            Y_train_s,
            r=r,
            n_starts=n_starts,
            seed=seed + fold_idx,
            slm_max_iter=slm_max_iter,
        )
        if verbose:
            meta = slm_params.get("_meta", {}) if isinstance(slm_params, dict) else {}
            iters = meta.get("n_iterations")
            iters_s = f", iters={iters}" if iters is not None else ""
            print(f"    Done PPLS-SLM in {time.time()-_t_fit:.1f}s{iters_s}.", flush=True)

        y_pred_slm_s, Cov_slm_s = _predict_ppls(X_test_s, params=slm_params)

        y_pred_slm = _unstandardize_y_pred(y_pred_slm_s, sy)

        m_slm = compute_regression_metrics(Y_test, y_pred_slm)
        metrics_rows.append(
            {
                "fold": fold_idx + 1,
                "method": "PPLS-SLM",
                "mse": m_slm.mse,
                "mae": m_slm.mae,
                "r2": m_slm.r2_mean,
            }
        )

        # Calibration (percent)
        Cov_slm = _unstandardize_cov_y(Cov_slm_s, sy)
        for a in alphas:
            lower, upper = compute_credible_intervals(y_pred_slm, Cov_slm, alpha=float(a))
            cov = 100.0 * empirical_coverage(Y_test, lower, upper)
            calib_rows.append({"fold": fold_idx + 1, "method": "PPLS-SLM", "alpha": float(a), "coverage": cov})

        # --- PPLS-EM ---
        if verbose:
            print(f"    Fitting PPLS-EM  (starts={n_starts}, max_iter={em_max_iter}, tol={em_tol})...", flush=True)
        _t_fit = time.time()
        em_params = _fit_ppls_params_em(
            X_train_s,
            Y_train_s,
            r=r,
            n_starts=n_starts,
            seed=seed + fold_idx,
            em_max_iter=em_max_iter,
            em_tol=em_tol,
        )
        if verbose:
            meta = em_params.get("_meta", {}) if isinstance(em_params, dict) else {}
            iters = meta.get("n_iterations")
            iters_s = f", iters={iters}" if iters is not None else ""
            print(f"    Done PPLS-EM in {time.time()-_t_fit:.1f}s{iters_s}.", flush=True)

        y_pred_em_s, Cov_em_s = _predict_ppls(X_test_s, params=em_params)

        y_pred_em = _unstandardize_y_pred(y_pred_em_s, sy)

        m_em = compute_regression_metrics(Y_test, y_pred_em)
        metrics_rows.append(
            {
                "fold": fold_idx + 1,
                "method": "PPLS-EM",
                "mse": m_em.mse,
                "mae": m_em.mae,
                "r2": m_em.r2_mean,
            }
        )

        Cov_em = _unstandardize_cov_y(Cov_em_s, sy)
        for a in alphas:
            lower, upper = compute_credible_intervals(y_pred_em, Cov_em, alpha=float(a))
            cov = 100.0 * empirical_coverage(Y_test, lower, upper)
            calib_rows.append({"fold": fold_idx + 1, "method": "PPLS-EM", "alpha": float(a), "coverage": cov})

        # --- Baselines ---
        if include_baselines:
            if verbose:
                print(f"    Fitting PLSR (n_components={r})...", flush=True)
            _t_fit = time.time()
            plsr = run_plsr_prediction(X_train, Y_train, X_test, Y_test, n_components=r)
            if verbose:
                print(f"    Done PLSR in {time.time()-_t_fit:.1f}s.", flush=True)

            m_plsr = plsr["metrics"]
            metrics_rows.append(
                {
                    "fold": fold_idx + 1,
                    "method": "PLSR",
                    "mse": m_plsr.mse,
                    "mae": m_plsr.mae,
                    "r2": m_plsr.r2_mean,
                }
            )

            if verbose:
                print("    Fitting RidgeCV...", flush=True)
            _t_fit = time.time()
            ridge = run_ridge_prediction(X_train, Y_train, X_test, Y_test)
            if verbose:
                print(f"    Done RidgeCV in {time.time()-_t_fit:.1f}s.", flush=True)

            m_ridge = ridge["metrics"]
            metrics_rows.append(
                {
                    "fold": fold_idx + 1,
                    "method": "Ridge",
                    "mse": m_ridge.mse,
                    "mae": m_ridge.mae,
                    "r2": m_ridge.r2_mean,
                }
            )


    metrics_per_fold = pd.DataFrame(metrics_rows)

    def _summarise(df: pd.DataFrame) -> pd.DataFrame:
        out = []
        for method, sub in df.groupby("method", sort=False):
            out.append(
                {
                    "method": method,
                    "mse_mean": float(sub["mse"].mean()),
                    "mse_std": float(sub["mse"].std(ddof=1)),
                    "mae_mean": float(sub["mae"].mean()),
                    "mae_std": float(sub["mae"].std(ddof=1)),
                    "r2_mean": float(sub["r2"].mean()),
                    "r2_std": float(sub["r2"].std(ddof=1)),
                }
            )
        return pd.DataFrame(out)

    metrics_summary = _summarise(metrics_per_fold)

    calib_long = pd.DataFrame(calib_rows)

    # Summary by alpha for SLM/EM
    calib_summary_rows = []
    for a, suba in calib_long.groupby("alpha", sort=True):
        row = {"alpha": float(a), "expected_coverage": 100.0 * (1.0 - float(a))}
        for method in ("PPLS-SLM", "PPLS-EM"):
            subm = suba[suba["method"] == method]
            row[f"{method}_mean"] = float(subm["coverage"].mean())
            row[f"{method}_std"] = float(subm["coverage"].std(ddof=1))
        calib_summary_rows.append(row)

    calib_summary = pd.DataFrame(calib_summary_rows)

    return metrics_per_fold, metrics_summary, calib_summary


# ─────────────────────────────────────────────────────────────────────────────
#  Visualisation (optional)
# ─────────────────────────────────────────────────────────────────────────────

def plot_calibration(calib_summary: pd.DataFrame, output_dir: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not found – skipping calibration plot.")
        return

    x = calib_summary["expected_coverage"].to_numpy()
    y_slm = calib_summary["PPLS-SLM_mean"].to_numpy()
    y_em = calib_summary["PPLS-EM_mean"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(x, y_slm, "o-", label="PPLS-SLM")
    ax.plot(x, y_em, "s-", label="PPLS-EM")
    ax.plot([x.min(), x.max()], [x.min(), x.max()], "k--", linewidth=1.2, label="Perfect")

    ax.set_xlabel("Nominal coverage (%)")
    ax.set_ylabel("Empirical coverage (%)")
    ax.set_title("Calibration of predictive credible intervals")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(output_dir, "calibration_plot.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="PPLS prediction experiment (synthetic)")
    p.add_argument("--config", type=str, required=True, help="Path to config JSON (single source of truth)")
    return p.parse_args()


def main():
    import time

    args = parse_args()

    from ppls_slm.experiment_config import (
        coerce_bool,
        coerce_float,
        coerce_int,
        get_experiment_cfg,
        load_config,
        require_keys,
    )

    cfg = load_config(args.config)
    pred_cfg = get_experiment_cfg(cfg, "prediction")

    require_keys(
        pred_cfg,
        [
            "thread_limit",
            "output_dir",
            "p",
            "q",
            "r",
            "n_samples",
            "n_folds",
            "n_starts",
            "seed",
            "sigma_e2",
            "sigma_f2",
            "sigma_h2",
            "max_iter",
            "em_tol",
            "plot",
            "no_baselines",
        ],
        ctx="experiments.prediction",
    )

    # Coerce basic types
    for k in ("thread_limit", "p", "q", "r", "n_samples", "n_folds", "n_starts", "seed", "max_iter"):
        coerce_int(pred_cfg, k, ctx="experiments.prediction")
    for k in ("sigma_e2", "sigma_f2", "sigma_h2", "em_tol"):
        coerce_float(pred_cfg, k, ctx="experiments.prediction")
    for k in ("plot", "no_baselines"):
        coerce_bool(pred_cfg, k, ctx="experiments.prediction")

    output_dir = str(pred_cfg["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    # Runtime thread limiting (more reliable than env vars if NumPy was imported early).
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=int(pred_cfg["thread_limit"]))
    except Exception:
        pass

    print(f"[prediction] using: {__file__}", flush=True)
    print(f"[prediction] config: {args.config}", flush=True)

    print("=" * 60, flush=True)
    print("Prediction experiment (synthetic)", flush=True)
    print("=" * 60, flush=True)

    p_dim = int(pred_cfg["p"])
    q_dim = int(pred_cfg["q"])
    r = int(pred_cfg["r"])
    n_samples = int(pred_cfg["n_samples"])
    n_folds = int(pred_cfg["n_folds"])
    n_starts = int(pred_cfg["n_starts"])
    seed = int(pred_cfg["seed"])

    sigma_e2 = float(pred_cfg["sigma_e2"])
    sigma_f2 = float(pred_cfg["sigma_f2"])
    sigma_h2 = float(pred_cfg["sigma_h2"])

    max_iter = int(pred_cfg["max_iter"])
    em_tol = float(pred_cfg["em_tol"])

    print(f"  p={p_dim}, q={q_dim}, r={r}, N={n_samples}")
    print(f"  folds={n_folds}, starts={n_starts}")
    print(f"  SLM/EM: max_iter={max_iter} (spectral noise pre-estimation), tol={em_tol}")
    print("=" * 60)

    print("\nGenerating synthetic PPLS data...", flush=True)
    print("  - generating orthonormal loadings + sampling latent variables", flush=True)
    t0 = time.time()
    X, Y, _true_params = generate_ppls_data(
        p=p_dim,
        q=q_dim,
        r=r,
        n_samples=n_samples,
        sigma_e2=sigma_e2,
        sigma_f2=sigma_f2,
        sigma_h2=sigma_h2,
        seed=seed,
    )
    print(f"  X: {X.shape}, Y: {Y.shape}", flush=True)
    print(f"  data generation took {time.time()-t0:.3f}s", flush=True)

    alphas = [0.05, 0.10, 0.15, 0.20, 0.25]

    print(f"\nRunning {n_folds}-fold CV benchmark...", flush=True)
    metrics_per_fold, metrics_summary, calib_summary = kfold_prediction_benchmark(
        X,
        Y,
        r=r,
        n_folds=n_folds,
        n_starts=n_starts,
        seed=seed,
        slm_max_iter=max_iter,
        em_max_iter=max_iter,
        em_tol=em_tol,
        include_baselines=(not bool(pred_cfg["no_baselines"])),
        alphas=alphas,
        verbose=True,
    )

    # Save
    metrics_per_fold.to_csv(os.path.join(output_dir, "prediction_metrics_per_fold.csv"), index=False)
    metrics_summary.to_csv(os.path.join(output_dir, "prediction_metrics_summary.csv"), index=False)
    calib_summary.to_csv(os.path.join(output_dir, "calibration_comparison.csv"), index=False)

    print("\n── Prediction metrics (mean ± std across folds) ──")
    disp = metrics_summary.copy()
    for k in ("mse", "mae", "r2"):
        disp[f"{k}"] = disp[f"{k}_mean"].map(lambda x: f"{x:.4g}") + " ± " + disp[f"{k}_std"].map(lambda x: f"{x:.4g}")
    print(disp[["method", "mse", "mae", "r2"]].to_string(index=False))

    print("\n── Calibration summary (mean ± std, %) ──")
    disp_c = calib_summary.copy()
    disp_c["PPLS-SLM"] = disp_c["PPLS-SLM_mean"].map(lambda x: f"{x:.2f}") + " ± " + disp_c["PPLS-SLM_std"].map(lambda x: f"{x:.2f}")
    disp_c["PPLS-EM"] = disp_c["PPLS-EM_mean"].map(lambda x: f"{x:.2f}") + " ± " + disp_c["PPLS-EM_std"].map(lambda x: f"{x:.2f}")
    print(disp_c[["alpha", "expected_coverage", "PPLS-SLM", "PPLS-EM"]].to_string(index=False))

    if bool(pred_cfg["plot"]):
        plot_calibration(calib_summary, output_dir)
        print(f"\nSaved plot: {output_dir}/calibration_plot.png")

    print(f"\nResults saved to: {output_dir}/")
    print("Prediction experiment complete.")

    return metrics_summary


if __name__ == "__main__":
    main()


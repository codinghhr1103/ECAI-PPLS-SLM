"""Application 2: Prediction with Uncertainty Quantification (Section 8.3)

This module reproduces and extends the prediction experiments from the paper
"Scalar Likelihood Method for Probabilistic Partial Least Squares Model with Rank n Update".

Two experiment tracks are supported:

A) Synthetic PPLS data (paper's main prediction sandbox)
   - p=q=200, r=50, N=100 (defaults)
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
    python -m ppls_slm.apps.prediction --output_dir results_prediction

Common options
--------------
  --p INT         Dimension of x            (default: 200)
  --q INT         Dimension of y            (default: 200)
  --r INT         Latent dimension          (default: 50)
  --n_samples INT Sample size               (default: 100)
  --n_folds INT   CV folds                  (default: 5)
  --n_starts INT  Multi-start per fold      (default: 16)
  --seed INT      Random seed               (default: 42)
  --output_dir    Directory for results     (default: results_prediction)
  --plot          Generate calibration plot (flag)

PPLS algorithm options
----------------------
  --slm_max_iter INT    SLM max iterations (default: 100)
  --em_max_iter INT     EM max iterations  (default: 200)
  --em_tol FLOAT        EM relative tolerance (default: 1e-4)

Baseline options
----------------
  --no_baselines         Skip PLSR/Ridge baselines
"""

from __future__ import annotations

import argparse
import os
import warnings
from typing import Dict, List, Optional, Tuple

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
        slm_params = _fit_ppls_params_slm(
            X_train_s,
            Y_train_s,
            r=r,
            n_starts=n_starts,
            seed=seed + fold_idx,
            slm_max_iter=slm_max_iter,
        )
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
        em_params = _fit_ppls_params_em(
            X_train_s,
            Y_train_s,
            r=r,
            n_starts=n_starts,
            seed=seed + fold_idx,
            em_max_iter=em_max_iter,
            em_tol=em_tol,
        )
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
            plsr = run_plsr_prediction(X_train, Y_train, X_test, Y_test, n_components=r)
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

            ridge = run_ridge_prediction(X_train, Y_train, X_test, Y_test)
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

    p.add_argument("--p", type=int, default=200)
    p.add_argument("--q", type=int, default=200)
    p.add_argument("--r", type=int, default=50)
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--n_starts", type=int, default=16)

    p.add_argument("--sigma_e2", type=float, default=0.1)
    p.add_argument("--sigma_f2", type=float, default=0.1)
    p.add_argument("--sigma_h2", type=float, default=0.05)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="results_prediction")

    p.add_argument("--slm_max_iter", type=int, default=100)
    p.add_argument("--em_max_iter", type=int, default=200)
    p.add_argument("--em_tol", type=float, default=1e-4)

    p.add_argument("--no_baselines", action="store_true", help="Skip PLSR and Ridge baselines")
    p.add_argument("--plot", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Prediction experiment (synthetic)")
    print("=" * 60)
    print(f"  p={args.p}, q={args.q}, r={args.r}, N={args.n_samples}")
    print(f"  folds={args.n_folds}, starts={args.n_starts}")
    print(f"  SLM: max_iter={args.slm_max_iter} (spectral noise pre-estimation)")
    print(f"  EM : max_iter={args.em_max_iter}, tol={args.em_tol}")
    print("=" * 60)

    print("\nGenerating synthetic PPLS data...")
    X, Y, _true_params = generate_ppls_data(
        p=args.p,
        q=args.q,
        r=args.r,
        n_samples=args.n_samples,
        sigma_e2=args.sigma_e2,
        sigma_f2=args.sigma_f2,
        sigma_h2=args.sigma_h2,
        seed=args.seed,
    )
    print(f"  X: {X.shape}, Y: {Y.shape}")

    alphas = [0.05, 0.10, 0.15, 0.20, 0.25]

    print(f"\nRunning {args.n_folds}-fold CV benchmark...")
    metrics_per_fold, metrics_summary, calib_summary = kfold_prediction_benchmark(
        X,
        Y,
        r=args.r,
        n_folds=args.n_folds,
        n_starts=args.n_starts,
        seed=args.seed,
        slm_max_iter=args.slm_max_iter,
        em_max_iter=args.em_max_iter,
        em_tol=args.em_tol,
        include_baselines=(not args.no_baselines),
        alphas=alphas,
        verbose=True,
    )

    # Save
    metrics_per_fold.to_csv(os.path.join(args.output_dir, "prediction_metrics_per_fold.csv"), index=False)
    metrics_summary.to_csv(os.path.join(args.output_dir, "prediction_metrics_summary.csv"), index=False)
    calib_summary.to_csv(os.path.join(args.output_dir, "calibration_comparison.csv"), index=False)

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

    if args.plot:
        plot_calibration(calib_summary, args.output_dir)
        print(f"\nSaved plot: {args.output_dir}/calibration_plot.png")

    print(f"\nResults saved to: {args.output_dir}/")
    print("Prediction experiment complete.")

    return metrics_summary


if __name__ == "__main__":
    main()

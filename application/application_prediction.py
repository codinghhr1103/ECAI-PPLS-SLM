"""
Application 2: Prediction with Uncertainty Quantification
==========================================================

Reproduces the prediction experiment from Section 8.3 of the paper
"Scalar Likelihood Method for Probabilistic Partial Least Squares Model with Rank n Update".

This application:
1. Generates synthetic PPLS data  (p = q = 200, r = 50, N = 100).
2. Runs 5-fold cross-validation.
3. On each test fold, predicts Y_new | X_new using the conditional Gaussian
   distribution (Property 1 in the paper):
       E[y | x] = C B Σ_t W' (W Σ_t W' + σ_e² I)^{-1} x
       Cov[y|x] = C(B² Σ_t + σ_h² I)C' + σ_f² I
                  - C B Σ_t W' (W Σ_t W' + σ_e² I)^{-1} W Σ_t B C'
4. Constructs symmetric credible intervals at levels α ∈ {0.05, 0.10, 0.15, 0.20, 0.25}
   (Algorithm 3 in the paper).
5. Reports the empirical coverage rates and compares them with the nominal
   (1 - α) targets, verifying well-calibrated uncertainty.

Usage
-----
    python application_prediction.py [--options]

Main options
------------
  --p INT         Dimension of x            (default: 200)
  --q INT         Dimension of y            (default: 200)
  --r INT         Latent dimension          (default: 50)
  --n_samples INT Sample size               (default: 100)
  --n_folds INT   CV folds                  (default: 5)
  --n_starts INT  Multi-start per fold      (default: 16)
  --seed INT      Random seed               (default: 42)
  --output_dir    Directory for results     (default: results_prediction)
  --plot          Generate calibration plot (flag)

Dependencies
------------
Requires ppls_model.py and algorithms.py in the same directory.
"""

import argparse
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# ── project imports ──────────────────────────────────────────────────────────
from algorithms import InitialPointGenerator, ScalarLikelihoodMethod
from ppls_model import PPLSModel


# ─────────────────────────────────────────────────────────────────────────────
#  Data generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_ppls_data(p: int = 200,
                       q: int = 200,
                       r: int = 50,
                       n_samples: int = 100,
                       sigma_e2: float = 0.1,
                       sigma_f2: float = 0.1,
                       sigma_h2: float = 0.05,
                       seed: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate synthetic PPLS data for the prediction experiment.

    Returns
    -------
    X : ndarray (N, p)
    Y : ndarray (N, q)
    true_params : dict with W, C, B, Sigma_t, sigma_e2, sigma_f2, sigma_h2
    """
    rng = np.random.RandomState(seed)

    # Random orthonormal loading matrices
    W, _ = np.linalg.qr(rng.randn(p, r))
    C, _ = np.linalg.qr(rng.randn(q, r))

    # Decreasing signals so identifiability holds: theta_t2[i] * b[i] decreasing
    theta_t2 = np.linspace(1.5, 0.3, r)
    b        = np.linspace(2.0, 0.5, r)
    # Verify: sort by product descending, already done via linspace
    assert np.all(np.diff(theta_t2 * b) < 0), "Identifiability violated."

    B       = np.diag(b)
    Sigma_t = np.diag(theta_t2)

    model = PPLSModel(p, q, r)
    np.random.seed(seed + 1)   # reproducible sampling
    X, Y = model.sample(n_samples, W, C, B, Sigma_t, sigma_e2, sigma_f2, sigma_h2)

    true_params = {
        "W": W, "C": C, "B": B, "Sigma_t": Sigma_t,
        "sigma_e2": sigma_e2, "sigma_f2": sigma_f2, "sigma_h2": sigma_h2,
    }
    return X, Y, true_params


# ─────────────────────────────────────────────────────────────────────────────
#  Prediction helpers  (Property 1 / Algorithm 3 in the paper)
# ─────────────────────────────────────────────────────────────────────────────

def predict_conditional_mean(x_new: np.ndarray,
                              params: Dict) -> np.ndarray:
    """
    Compute E[y_new | x_new, params] via the conditional Gaussian formula:

        E[y | x] = C B Σ_t W' Σ_xx^{-1} x
    where  Σ_xx = W Σ_t W' + σ_e² I.

    Parameters
    ----------
    x_new    : (q,) or (N_test, p) row vector(s) of new observations
    params   : dict with W, C, B, Sigma_t, sigma_e2, sigma_f2, sigma_h2

    Returns
    -------
    y_pred   : (..., q) predicted means
    """
    W, C, B    = params["W"], params["C"], params["B"]
    Sigma_t    = params["Sigma_t"]
    sigma_e2   = params["sigma_e2"]

    # Σ_xx = W Σ_t W' + σ_e² I
    Sigma_xx = W @ Sigma_t @ W.T + sigma_e2 * np.eye(W.shape[0])

    # Regression coefficient  A = C B Σ_t W' Σ_xx^{-1}
    # Computed via Cholesky for numerical stability
    L = np.linalg.cholesky(Sigma_xx + 1e-9 * np.eye(Sigma_xx.shape[0]))
    # Solve Σ_xx^{-1} W  → shape (p, r)
    tmp = np.linalg.solve(L, W)                   # L^{-1} W
    tmp = np.linalg.solve(L.T, tmp)               # Σ_xx^{-1} W

    A = C @ B @ Sigma_t @ tmp.T                   # (q, p)

    if x_new.ndim == 1:
        return A @ x_new
    return x_new @ A.T                            # (N_test, q)


def predict_conditional_covariance(params: Dict) -> np.ndarray:
    """
    Compute Cov[y_new | x_new, params].

    Cov[y|x] = C(B² Σ_t + σ_h² I)C' + σ_f² I
               - C B Σ_t W' Σ_xx^{-1} W Σ_t B C'

    The conditional covariance is the same for every x_new.

    Returns
    -------
    Cov_yx : (q, q) ndarray
    """
    W, C, B    = params["W"], params["C"], params["B"]
    Sigma_t    = params["Sigma_t"]
    sigma_e2   = params["sigma_e2"]
    sigma_f2   = params["sigma_f2"]
    sigma_h2   = params["sigma_h2"]

    r = W.shape[1]
    q = C.shape[0]

    b        = np.diag(B)
    theta_t2 = np.diag(Sigma_t)

    Sigma_xx = W @ Sigma_t @ W.T + sigma_e2 * np.eye(W.shape[0])
    L = np.linalg.cholesky(Sigma_xx + 1e-9 * np.eye(Sigma_xx.shape[0]))
    WtSig_inv = np.linalg.solve(L.T, np.linalg.solve(L, W))  # Σ_xx^{-1} W (p,r)

    # C(B² Σ_t + σ_h² I)C'
    B2Sigma_t = np.diag(b ** 2 * theta_t2)
    term1 = C @ (B2Sigma_t + sigma_h2 * np.eye(r)) @ C.T + sigma_f2 * np.eye(q)

    # C B Σ_t W' Σ_xx^{-1} W Σ_t B C'
    M = C @ B @ Sigma_t @ WtSig_inv.T @ Sigma_t @ B @ C.T   # (q,q)

    Cov = term1 - M
    # Symmetrise and regularise
    Cov = (Cov + Cov.T) / 2 + 1e-9 * np.eye(q)
    return Cov


def compute_credible_intervals(y_pred: np.ndarray,
                                Cov_yx: np.ndarray,
                                alpha: float = 0.05
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute element-wise symmetric credible intervals.

    CI_j = mean_j ± z_{α/2} * sqrt(Cov[j,j])

    Parameters
    ----------
    y_pred : (N_test, q)  predicted means
    Cov_yx : (q, q)       conditional covariance (same for all x_new)
    alpha  : significance level (e.g. 0.05 for 95% CI)

    Returns
    -------
    lower : (N_test, q)
    upper : (N_test, q)
    """
    z = stats.norm.ppf(1 - alpha / 2)
    std = np.sqrt(np.diag(Cov_yx))       # (q,)

    lower = y_pred - z * std[np.newaxis, :]
    upper = y_pred + z * std[np.newaxis, :]
    return lower, upper


def empirical_coverage(y_true: np.ndarray,
                        lower: np.ndarray,
                        upper: np.ndarray) -> float:
    """
    Fraction of (test-sample, feature) entries whose true value falls
    within [lower, upper].
    """
    within = (y_true >= lower) & (y_true <= upper)
    return within.mean()


# ─────────────────────────────────────────────────────────────────────────────
#  Single-fold prediction
# ─────────────────────────────────────────────────────────────────────────────

def fit_and_predict_fold(X_train: np.ndarray,
                          Y_train: np.ndarray,
                          X_test:  np.ndarray,
                          Y_test:  np.ndarray,
                          r: int,
                          n_starts: int = 16,
                          algorithm_seed: int = 0,
                          use_true_params: bool = False,
                          true_params: Optional[Dict] = None,
                          alphas: Optional[List[float]] = None,
                          ) -> Dict:
    """
    Fit PPLS on training data and evaluate prediction coverage on test data.

    Parameters
    ----------
    use_true_params : bool
        If True, skip model fitting and use true_params directly
        (simulates perfectly calibrated model as in the paper).
    alphas : list of alpha values for credible intervals.

    Returns
    -------
    dict with keys 'coverage_{alpha}' for each alpha.
    """
    if alphas is None:
        alphas = [0.05, 0.10, 0.15, 0.20, 0.25]

    p, q = X_train.shape[1], Y_train.shape[1]

    # ── Fit model ─────────────────────────────────────────────────────────────
    if use_true_params:
        params = true_params
    else:
        init_gen = InitialPointGenerator(p=p, q=q, r=r,
                                         n_starts=n_starts,
                                         random_seed=algorithm_seed)
        starting_points = init_gen.generate_starting_points()

        slm = ScalarLikelihoodMethod(p=p, q=q, r=r, max_iter=100,
                                     use_noise_preestimation=True)
        res = slm.fit(X_train, Y_train, starting_points)

        params = {
            "W":        res["W"],
            "C":        res["C"],
            "B":        res["B"],
            "Sigma_t":  res["Sigma_t"],
            "sigma_e2": res["sigma_e2"],
            "sigma_f2": res["sigma_f2"],
            "sigma_h2": res["sigma_h2"],
        }

    # ── Predict on test set ───────────────────────────────────────────────────
    y_pred   = predict_conditional_mean(X_test, params)       # (N_test, q)
    Cov_yx   = predict_conditional_covariance(params)         # (q, q)

    # ── Compute coverage for each alpha ───────────────────────────────────────
    fold_results = {}
    for alpha in alphas:
        lower, upper = compute_credible_intervals(y_pred, Cov_yx, alpha=alpha)
        cov = empirical_coverage(Y_test, lower, upper) * 100  # percent
        fold_results[f"alpha_{alpha}"] = cov

    fold_results["y_pred"]  = y_pred
    fold_results["Cov_yx"]  = Cov_yx
    fold_results["params"]  = params
    return fold_results


# ─────────────────────────────────────────────────────────────────────────────
#  k-fold cross-validation
# ─────────────────────────────────────────────────────────────────────────────

def kfold_prediction_experiment(
    X: np.ndarray,
    Y: np.ndarray,
    r: int = 50,
    n_folds: int = 5,
    n_starts: int = 16,
    alphas: Optional[List[float]] = None,
    use_true_params: bool = True,
    true_params: Optional[Dict] = None,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run k-fold CV and collect empirical coverage at each alpha.

    Parameters
    ----------
    use_true_params : bool
        Paper's primary experiment: use the true parameters on the test set
        (simulates perfectly calibrated model).  Set to False to use SLM
        estimates from the training fold.

    Returns
    -------
    coverage_df : DataFrame  (rows = folds, columns = alpha levels)
    """
    if alphas is None:
        alphas = [0.05, 0.10, 0.15, 0.20, 0.25]

    N = X.shape[0]
    rng = np.random.RandomState(seed)
    indices = rng.permutation(N)
    folds = np.array_split(indices, n_folds)

    rows = []
    for fold_idx, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_idx])

        X_train, Y_train = X[train_idx], Y[train_idx]
        X_test,  Y_test  = X[test_idx],  Y[test_idx]

        if verbose:
            print(f"  Fold {fold_idx+1}/{n_folds}  "
                  f"(train={len(train_idx)}, test={len(test_idx)})", end="  ")

        fold_res = fit_and_predict_fold(
            X_train, Y_train, X_test, Y_test,
            r=r,
            n_starts=n_starts,
            algorithm_seed=seed + fold_idx,
            use_true_params=use_true_params,
            true_params=true_params,
            alphas=alphas,
        )

        row = {f"Alpha={a}": fold_res[f"alpha_{a}"] for a in alphas}
        rows.append(row)

        if verbose:
            # Print coverage at alpha=0.05 as a quick check
            cov_05 = fold_res["alpha_0.05"]
            print(f"Coverage@α=0.05: {cov_05:.2f}%")

    coverage_df = pd.DataFrame(rows)
    coverage_df.index = [f"Fold {i+1}" for i in range(n_folds)]
    return coverage_df


# ─────────────────────────────────────────────────────────────────────────────
#  Summary statistics
# ─────────────────────────────────────────────────────────────────────────────

def build_coverage_table(coverage_df: pd.DataFrame) -> pd.DataFrame:
    """
    Append a summary row (mean ± std across folds) and a 'Nominal' column.

    Returns a formatted DataFrame matching Table 4 of the paper.
    """
    alphas = [float(c.split("=")[1]) for c in coverage_df.columns]
    nominals = [(1 - a) * 100 for a in alphas]

    mean_row = coverage_df.mean()
    std_row  = coverage_df.std()

    display = coverage_df.copy()
    display.loc["Mean ± Std"] = [
        f"{mean_row[c]:.2f} ± {std_row[c]:.2f}" for c in coverage_df.columns
    ]
    display.loc["Nominal (%)"] = [f"{nom:.0f}" for nom in nominals]
    return display


def check_calibration(coverage_df: pd.DataFrame,
                       tol: float = 3.0) -> bool:
    """
    Verify that the empirical coverage at each alpha is within `tol` percentage
    points of the nominal (1-alpha)*100 level on average across folds.
    """
    all_ok = True
    print("\nCalibration check (mean coverage vs nominal):")
    for col in coverage_df.columns:
        alpha = float(col.split("=")[1])
        nominal = (1 - alpha) * 100
        empirical = coverage_df[col].mean()
        diff = abs(empirical - nominal)
        status = "✓" if diff <= tol else "✗"
        print(f"  {col:14s}  nominal={nominal:.0f}%  empirical={empirical:.2f}%  "
              f"diff={diff:.2f}%  {status}")
        if diff > tol:
            all_ok = False
    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
#  Visualisation (optional)
# ─────────────────────────────────────────────────────────────────────────────

def plot_calibration(coverage_df: pd.DataFrame,
                     output_dir: str = "results_prediction"):
    """
    Line plot: empirical coverage vs nominal level for each fold.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        warnings.warn("matplotlib not found – skipping calibration plot.")
        return

    alphas = [float(c.split("=")[1]) for c in coverage_df.columns]
    nominals = [(1 - a) * 100 for a in alphas]

    fig, ax = plt.subplots(figsize=(7, 5))

    # Individual folds
    fold_rows = coverage_df.values  # (n_folds, n_alphas)
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(fold_rows)))
    for i, (row, color) in enumerate(zip(fold_rows, colors)):
        ax.plot(nominals, row, "o--", color=color, alpha=0.7,
                linewidth=1.2, markersize=5, label=f"Fold {i+1}")

    # Mean across folds
    means = coverage_df.mean().values
    ax.plot(nominals, means, "k-o", linewidth=2.0, markersize=7, label="Mean", zorder=5)

    # Perfect calibration diagonal
    ax.plot([min(nominals), max(nominals)], [min(nominals), max(nominals)],
            "r--", linewidth=1.5, label="Perfect calibration")

    ax.set_xlabel("Nominal coverage (%)", fontsize=11)
    ax.set_ylabel("Empirical coverage (%)", fontsize=11)
    ax.set_title("Prediction interval calibration (5-fold CV)", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(min(nominals) - 2, max(nominals) + 2)
    ax.set_ylim(min(nominals) - 5, max(nominals) + 5)
    plt.tight_layout()

    fig.savefig(os.path.join(output_dir, "calibration_plot.png"), dpi=150)
    plt.close(fig)
    print(f"Calibration plot saved to {output_dir}/calibration_plot.png")


def plot_prediction_example(X_test: np.ndarray,
                             Y_test: np.ndarray,
                             y_pred: np.ndarray,
                             Cov_yx: np.ndarray,
                             feature_idx: int = 0,
                             alpha: float = 0.05,
                             output_dir: str = "results_prediction"):
    """
    Scatter plot: true vs predicted for a single output feature,
    with 95% credible intervals shown as error bars.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not found – skipping example prediction plot.")
        return

    z = stats.norm.ppf(1 - alpha / 2)
    std_j = np.sqrt(Cov_yx[feature_idx, feature_idx])
    err = z * std_j  # scalar (same for all test points)

    true_j = Y_test[:, feature_idx]
    pred_j = y_pred[:, feature_idx]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.errorbar(true_j, pred_j, yerr=err, fmt="o", color="steelblue",
                ecolor="lightblue", elinewidth=1.5, capsize=3,
                alpha=0.7, markersize=5, label=f"95% CI (α={alpha})")
    lims = [min(true_j.min(), pred_j.min()) - 0.2,
            max(true_j.max(), pred_j.max()) + 0.2]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="y = x")
    ax.set_xlabel(f"True Y (feature {feature_idx})", fontsize=11)
    ax.set_ylabel(f"Predicted Y (feature {feature_idx})", fontsize=11)
    ax.set_title("True vs Predicted with Credible Intervals", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    fig.savefig(os.path.join(output_dir, "prediction_example.png"), dpi=150)
    plt.close(fig)
    print(f"Prediction example plot saved to {output_dir}/prediction_example.png")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="PPLS Prediction with Uncertainty Quantification"
    )
    parser.add_argument("--p",          type=int, default=200,
                        help="Dimension of x (default: 200).")
    parser.add_argument("--q",          type=int, default=200,
                        help="Dimension of y (default: 200).")
    parser.add_argument("--r",          type=int, default=50,
                        help="Latent dimension (default: 50).")
    parser.add_argument("--n_samples",  type=int, default=100,
                        help="Total sample size N (default: 100).")
    parser.add_argument("--n_folds",    type=int, default=5,
                        help="Number of CV folds (default: 5).")
    parser.add_argument("--n_starts",   type=int, default=16,
                        help="Multi-start initializations per fold (default: 16).")
    parser.add_argument("--sigma_e2",   type=float, default=0.1)
    parser.add_argument("--sigma_f2",   type=float, default=0.1)
    parser.add_argument("--sigma_h2",   type=float, default=0.05)
    parser.add_argument("--use_true",   action="store_true",
                        help="Use true params for prediction (paper's setup).")
    parser.add_argument("--use_slm",    action="store_true",
                        help="Fit SLM on each training fold (slower).")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Random seed (default: 42).")
    parser.add_argument("--output_dir", type=str, default="results_prediction",
                        help="Directory to save results.")
    parser.add_argument("--plot",       action="store_true",
                        help="Generate calibration and example plots.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine prediction mode
    # Default (neither flag): paper's setup = use true parameters
    use_true_params = not args.use_slm  # True unless --use_slm is set

    os.makedirs(args.output_dir, exist_ok=True)
    alphas = [0.05, 0.10, 0.15, 0.20, 0.25]

    print("="*60)
    print("PPLS Prediction with Uncertainty Quantification")
    print("="*60)
    print(f"  Model:      p={args.p}, q={args.q}, r={args.r}")
    print(f"  Samples:    N={args.n_samples}")
    print(f"  CV folds:   {args.n_folds}")
    print(f"  Mode:       {'true parameters (paper setup)' if use_true_params else 'SLM-fitted parameters'}")
    print("="*60)

    # ── 1. Generate data ──────────────────────────────────────────────────────
    print("\nGenerating synthetic PPLS data...")
    X, Y, true_params = generate_ppls_data(
        p=args.p, q=args.q, r=args.r, n_samples=args.n_samples,
        sigma_e2=args.sigma_e2, sigma_f2=args.sigma_f2, sigma_h2=args.sigma_h2,
        seed=args.seed,
    )
    print(f"  X: {X.shape},  Y: {Y.shape}")

    # ── 2. Run k-fold CV ──────────────────────────────────────────────────────
    print(f"\nRunning {args.n_folds}-fold cross-validation...")
    coverage_df = kfold_prediction_experiment(
        X, Y,
        r=args.r,
        n_folds=args.n_folds,
        n_starts=args.n_starts,
        alphas=alphas,
        use_true_params=use_true_params,
        true_params=true_params if use_true_params else None,
        seed=args.seed,
        verbose=True,
    )

    # ── 3. Print summary table ────────────────────────────────────────────────
    print("\n── Coverage table (%, across folds) ──")
    display_df = build_coverage_table(coverage_df)
    print(display_df.to_string())

    # ── 4. Calibration check ──────────────────────────────────────────────────
    check_calibration(coverage_df)

    # ── 5. Save results ───────────────────────────────────────────────────────
    coverage_df.to_csv(os.path.join(args.output_dir, "coverage_results.csv"))
    display_df.to_csv(os.path.join(args.output_dir, "coverage_table.csv"))
    print(f"\nResults saved to: {args.output_dir}/")

    # ── 6. Plots ──────────────────────────────────────────────────────────────
    if args.plot:
        plot_calibration(coverage_df, args.output_dir)

        # Generate one example prediction (using true params for illustration)
        rng = np.random.RandomState(args.seed)
        test_idx = rng.choice(args.n_samples, max(10, args.n_samples // args.n_folds),
                               replace=False)
        X_test = X[test_idx]
        Y_test = Y[test_idx]

        y_pred  = predict_conditional_mean(X_test, true_params)
        Cov_yx  = predict_conditional_covariance(true_params)
        plot_prediction_example(X_test, Y_test, y_pred, Cov_yx,
                                 feature_idx=0, alpha=0.05,
                                 output_dir=args.output_dir)

    print("\nPrediction experiment complete.")
    return coverage_df


if __name__ == "__main__":
    main()

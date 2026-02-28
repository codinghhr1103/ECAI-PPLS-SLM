"""
Speed Comparison: Scalar vs Matrix Likelihood Evaluation
=========================================================

Benchmarks the wall-clock time for evaluating the PPLS log-likelihood
in two equivalent formulations:

  Matrix form  – builds the full (p+q)×(p+q) covariance matrix, then
                 computes its Cholesky decomposition and solves a linear
                 system to obtain  ln det(Σ) + tr(S Σ⁻¹).

  Scalar form  – avoids full matrix operations by computing the same
                 quantity component-wise through r scalar loops, using
                 the rank-n determinant / inverse identity for structured
                 matrices with orthonormal loading columns.

All configuration is in the CONFIG dict at the top of the file.
No command-line arguments are accepted.
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# ======================================================================
#  CONFIG  –  all tunable parameters live here
# ======================================================================
CONFIG = {
    # Grid of p and q values to sweep (p == q assumed in each run)
    "pq_values": [50, 100, 200, 500, 1000],

    # Latent dimensions to evaluate; each value produces one heatmap panel
    "r_values": [5, 10, 20],

    # Number of timed repetitions per (p, q, r) configuration
    "n_repetitions": 100,

    # Random seed for reproducibility
    "random_seed": 42,

    # Output file path for the combined heatmap figure
    "output_path": "speed_comparison.png",

    # Figure size (width, height) in inches
    "figure_size": (14, 4.5),

    # DPI for saved figure
    "figure_dpi": 300,

    # Matplotlib colour map for heatmaps
    "cmap": "YlOrRd",
}
# ======================================================================


# ──────────────────────────────────────────────────────────────────────
#  Helpers: random valid PPLS parameters
# ──────────────────────────────────────────────────────────────────────

def _random_orthonormal(m: int, k: int, rng: np.random.RandomState) -> np.ndarray:
    """Return an m×k matrix with orthonormal columns (via QR)."""
    A = rng.randn(m, k)
    Q, _ = np.linalg.qr(A)
    return Q[:, :k]


def generate_params(p: int, q: int, r: int,
                    rng: np.random.RandomState) -> dict:
    """
    Generate random PPLS parameters satisfying identifiability constraints:
      • W  (p×r), C  (q×r) – orthonormal columns
      • b  (r,)             – positive entries
      • theta_t2 (r,)       – positive entries; θ²·b decreasing
      • sigma_h2, sigma_e2, sigma_f2 – positive scalars
    """
    W = _random_orthonormal(p, r, rng)
    C = _random_orthonormal(q, r, rng)

    # Draw raw values and sort so θ²·b is strictly decreasing
    raw_t = np.exp(rng.uniform(-0.5, 1.0, r))
    raw_b = np.exp(rng.uniform(-0.5, 1.0, r))
    order    = np.argsort(raw_t * raw_b)[::-1]
    theta_t2 = raw_t[order]
    b        = raw_b[order]

    sigma_h2 = float(np.exp(rng.uniform(-3.0, -0.5)))
    sigma_e2 = float(np.exp(rng.uniform(-3.0, -0.5)))
    sigma_f2 = float(np.exp(rng.uniform(-3.0, -0.5)))

    return dict(W=W, C=C, b=b, theta_t2=theta_t2,
                sigma_h2=sigma_h2, sigma_e2=sigma_e2, sigma_f2=sigma_f2)


def random_sample_covariance(p: int, q: int,
                              rng: np.random.RandomState) -> np.ndarray:
    """Return a random positive-definite (p+q)×(p+q) sample covariance."""
    n = p + q
    A = rng.randn(n, n)
    return (A @ A.T) / n + np.eye(n) * 0.1


# ──────────────────────────────────────────────────────────────────────
#  Matrix-form likelihood
# ──────────────────────────────────────────────────────────────────────

def build_covariance(params: dict, p: int, q: int) -> np.ndarray:
    """Assemble the full (p+q)×(p+q) PPLS joint covariance matrix."""
    W, C  = params["W"], params["C"]
    b     = params["b"]
    th2   = params["theta_t2"]
    sh2   = params["sigma_h2"]
    se2   = params["sigma_e2"]
    sf2   = params["sigma_f2"]

    Sigma_t  = np.diag(th2)
    B        = np.diag(b)
    B2_Sig_t = np.diag(b**2 * th2)

    Sxx = W @ Sigma_t @ W.T + se2 * np.eye(p)
    Sxy = W @ Sigma_t @ B @ C.T
    Syy = C @ (B2_Sig_t + sh2 * np.eye(len(b))) @ C.T + sf2 * np.eye(q)

    return np.block([[Sxx, Sxy], [Sxy.T, Syy]])


def matrix_log_likelihood(S: np.ndarray, Sigma: np.ndarray) -> float:
    """
    Negative profile log-likelihood (matrix form):
        L = ln det(Σ) + tr(S Σ⁻¹)

    Uses Cholesky decomposition for numerical stability.
    """
    sign, logdet = np.linalg.slogdet(Sigma)
    if sign <= 0:
        return np.inf
    try:
        L     = np.linalg.cholesky(Sigma)
        Z     = np.linalg.solve(L, S)
        trace = np.trace(np.linalg.solve(L.T, Z))
    except np.linalg.LinAlgError:
        trace = np.trace(S @ np.linalg.inv(Sigma))
    return logdet + trace


# ──────────────────────────────────────────────────────────────────────
#  Scalar-form likelihood
# ──────────────────────────────────────────────────────────────────────

def scalar_log_likelihood(S: np.ndarray, params: dict,
                           p: int, q: int) -> float:
    """
    Negative profile log-likelihood (scalar form).

    Avoids explicit construction of Σ by computing ln det Σ and
    tr(S Σ⁻¹) component-wise over the r latent dimensions, using
    the rank-n determinant / inverse identities for matrices of the
    form  A D_n Aᵀ + k I  with orthonormal A.

    Scalar expansion:

        ln det Σ = (p-r) ln σ_e² + (q-r) ln σ_f²  + Σᵢ ln Dᵢ

        tr(S Σ⁻¹) = tr(S_xx)/σ_e² + tr(S_yy)/σ_f²
                    - Σᵢ [ Φ_x(i) Q_x(i) + Φ_y(i) Q_y(i)
                           + Φ_xy(i) Q_xy(i) ]

    where:
        Dᵢ     = (σ_f² + σ_h²)(θ_tᵢ² + σ_e²) + bᵢ² θ_tᵢ² σ_f²
        Φ_x(i) = (σ_f² + σ_h²) θ_tᵢ² / Dᵢ
        Φ_y(i) = [σ_h²(θ_tᵢ² + σ_e²) + bᵢ² θ_tᵢ² σ_e²] / Dᵢ
        Φ_xy(i)= bᵢ θ_tᵢ² / Dᵢ
        Q_x(i) = (wᵢᵀ S_xx wᵢ) / σ_e²
        Q_y(i) = (cᵢᵀ S_yy cᵢ) / σ_f²
        Q_xy(i)= 2 wᵢᵀ S_xy cᵢ
    """
    W, C  = params["W"], params["C"]
    b     = params["b"]
    th2   = params["theta_t2"]
    sh2   = params["sigma_h2"]
    se2   = params["sigma_e2"]
    sf2   = params["sigma_f2"]
    r     = len(b)

    S_xx = S[:p, :p]
    S_yy = S[p:, p:]
    S_xy = S[:p, p:]

    # ---- log-determinant ----
    D = (sf2 + sh2) * (th2 + se2) + b**2 * th2 * sf2
    if np.any(D <= 0):
        return np.inf
    ln_det = (p - r) * np.log(se2) + (q - r) * np.log(sf2) + np.sum(np.log(D))

    # ---- trace term ----
    trace = np.trace(S_xx) / se2 + np.trace(S_yy) / sf2

    Phi_x  = (sf2 + sh2) * th2 / D
    Phi_y  = (sh2 * (th2 + se2) + b**2 * th2 * se2) / D
    Phi_xy = b * th2 / D

    # Vectorised projected quadratic forms over all r components
    # WtSxxW[i,i] = wᵢᵀ S_xx wᵢ
    WtSxxW_diag = np.einsum("pi,pq,qi->i", W, S_xx, W)
    CtSyyC_diag = np.einsum("qi,qp,pi->i", C, S_yy, C)
    WtSxyC_diag = np.einsum("pi,pq,qi->i", W, S_xy, C)

    Q_x  = WtSxxW_diag / se2
    Q_y  = CtSyyC_diag / sf2
    Q_xy = 2.0 * WtSxyC_diag

    trace -= np.sum(Phi_x * Q_x + Phi_y * Q_y + Phi_xy * Q_xy)

    return ln_det + trace


# ──────────────────────────────────────────────────────────────────────
#  Timing harness
# ──────────────────────────────────────────────────────────────────────

def time_one_config(p: int, q: int, r: int,
                    n_reps: int,
                    rng: np.random.RandomState) -> tuple:
    """
    Return (mean_matrix_time, mean_scalar_time) in seconds over n_reps
    repetitions for a given (p, q, r) configuration.
    Each repetition uses freshly generated random parameters and S so
    that cache effects do not systematically favour either form.
    """
    mat_times = np.empty(n_reps)
    scl_times = np.empty(n_reps)

    for k in range(n_reps):
        params = generate_params(p, q, r, rng)
        S      = random_sample_covariance(p, q, rng)

        # matrix form
        t0 = time.perf_counter()
        Sigma = build_covariance(params, p, q)
        _     = matrix_log_likelihood(S, Sigma)
        mat_times[k] = time.perf_counter() - t0

        # scalar form
        t0 = time.perf_counter()
        _  = scalar_log_likelihood(S, params, p, q)
        scl_times[k] = time.perf_counter() - t0

    return float(np.mean(mat_times)), float(np.mean(scl_times))


# ──────────────────────────────────────────────────────────────────────
#  Main experiment loop
# ──────────────────────────────────────────────────────────────────────

def run_experiment(cfg: dict) -> dict:
    """
    Sweep the full (p, q, r) grid and return:
        results[r_val]  – 2-D NumPy array of speedup factors,
                          rows index p, columns index q.
    """
    rng     = np.random.RandomState(cfg["random_seed"])
    pq_vals = cfg["pq_values"]
    r_vals  = cfg["r_values"]
    n_reps  = cfg["n_repetitions"]
    n_pq    = len(pq_vals)

    results = {}
    total   = len(r_vals) * n_pq * n_pq
    done    = 0

    for r in r_vals:
        speedup = np.zeros((n_pq, n_pq))
        for i, p in enumerate(pq_vals):
            for j, q in enumerate(pq_vals):
                done += 1
                print(f"  [{done:3d}/{total}]  r={r:2d}  p={p:4d}  q={q:4d}",
                      end="", flush=True)
                t_mat, t_scl = time_one_config(p, q, r, n_reps, rng)
                su = t_mat / t_scl if t_scl > 0 else np.nan
                speedup[i, j] = su
                print(f"  →  {su:.3f}×")
        results[r] = speedup

    return results


# ──────────────────────────────────────────────────────────────────────
#  Visualisation
# ──────────────────────────────────────────────────────────────────────

def plot_results(results: dict, cfg: dict):
    """Draw three side-by-side heatmaps (one per r value) and save."""
    pq_vals  = cfg["pq_values"]
    r_vals   = cfg["r_values"]
    out_path = cfg["output_path"]
    cmap     = cfg["cmap"]

    fig, axes = plt.subplots(1, len(r_vals),
                              figsize=cfg["figure_size"],
                              constrained_layout=True)

    # Shared colour scale across all panels
    all_vals = np.concatenate([results[r].ravel() for r in r_vals])
    vmin = np.nanmin(all_vals)
    vmax = np.nanmax(all_vals)

    tick_labels = [str(v) for v in pq_vals]

    for ax, r in zip(axes, r_vals):
        data = results[r]
        im   = ax.imshow(data, aspect="auto", origin="upper",
                         cmap=cmap, vmin=vmin, vmax=vmax,
                         interpolation="nearest")

        # Annotate each cell
        threshold = vmin + 0.6 * (vmax - vmin)
        for i in range(len(pq_vals)):
            for j in range(len(pq_vals)):
                val = data[i, j]
                color = "white" if val > threshold else "black"
                ax.text(j, i, f"{val:.2f}",
                        ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

        ax.set_xticks(range(len(pq_vals)))
        ax.set_yticks(range(len(pq_vals)))
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_yticklabels(tick_labels, fontsize=9)
        ax.set_xlabel("q", fontsize=10)
        ax.set_ylabel("p", fontsize=10)
        ax.set_title(f"r = {r}", fontsize=11, fontweight="bold")

        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Speedup (×)", fontsize=9)
        cb.locator = ticker.MaxNLocator(nbins=5)
        cb.update_ticks()

    fig.suptitle(
        "Speedup Factor: Matrix Form vs Scalar Form Likelihood Evaluation",
        fontsize=12, fontweight="bold", y=1.02,
    )

    fig.savefig(out_path, dpi=cfg["figure_dpi"],
                bbox_inches="tight", facecolor="white")
    print(f"\nFigure saved → {os.path.abspath(out_path)}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Scalar vs Matrix Likelihood  –  Speed Experiment")
    print("=" * 60)
    print(f"  pq_values     : {CONFIG['pq_values']}")
    print(f"  r_values      : {CONFIG['r_values']}")
    print(f"  n_repetitions : {CONFIG['n_repetitions']}")
    print(f"  random_seed   : {CONFIG['random_seed']}")
    print(f"  output_path   : {CONFIG['output_path']}")
    print("=" * 60 + "\n")

    results = run_experiment(CONFIG)

    # Print diagonal summary (p == q)
    print("\nSpeedup summary for p == q (matrix_time / scalar_time):")
    header = "  r  |  " + "  ".join(f"p=q={v:4d}" for v in CONFIG["pq_values"])
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in CONFIG["r_values"]:
        diag = [results[r][i, i] for i in range(len(CONFIG["pq_values"]))]
        row  = "  ".join(f"{v:8.3f}x" for v in diag)
        print(f"  {r:2d} |  {row}")

    plot_results(results, CONFIG)
    print("\nDone.")


if __name__ == "__main__":
    main()

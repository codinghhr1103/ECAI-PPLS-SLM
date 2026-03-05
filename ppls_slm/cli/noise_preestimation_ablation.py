"""Noise-preestimation ablation experiments.

Implements two experiments requested in the paper-ablation discussion:

1) Pre-estimation accuracy validation across SNR and N, with Theorem bound overlay.
2) Ablation: pre-estimate+fix vs joint optimisation of (sigma_e^2, sigma_f^2).


Run:
  python -m ppls_slm.cli.noise_preestimation_ablation --output_dir results_noise_ablation

All outputs are written under output_dir.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ppls_slm.data_generator import SineDataGenerator
from ppls_slm.ppls_model import NoiseEstimator, PPLSModel
from ppls_slm.algorithms import InitialPointGenerator, ScalarLikelihoodMethod


@dataclass(frozen=True)
class NoiseLevel:
    name: str
    sigma_e2: float
    sigma_f2: float
    sigma_h2: float


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _log(msg: str) -> None:
    # Keep logs visible when run under scripts/run_all_experiments.py tee runner.
    print(msg, flush=True)


def _set_global_seed(seed: int) -> None:
    np.random.seed(int(seed))



def _generate_params_fixed_loadings(
    p: int,
    q: int,
    r: int,
    sigma_t_diag: float,
    noise: NoiseLevel,
    seed: int,
) -> Dict:
    """Generate true parameters with fixed W/C (sine-based) and controllable Sigma_t."""
    gen = SineDataGenerator(p=p, q=q, r=r, n_samples=1, random_seed=seed, output_dir=".")
    params = gen.generate_true_parameters(sigma_e2=noise.sigma_e2, sigma_f2=noise.sigma_f2, sigma_h2=noise.sigma_h2)
    params["Sigma_t"] = np.diag(np.full(r, float(sigma_t_diag)))
    return params


def _sample_ppls(
    params: Dict,
    n_samples: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample (X, Y, S_signal) where S_signal = T W^T (the x-space signal)."""
    p, q, r = int(params["p"]), int(params["q"]), int(params["r"])
    W, C, B, Sigma_t = params["W"], params["C"], params["B"], params["Sigma_t"]
    sigma_e2, sigma_f2, sigma_h2 = float(params["sigma_e2"]), float(params["sigma_f2"]), float(params["sigma_h2"])

    theta_t = np.sqrt(np.diag(Sigma_t))
    T = rng.standard_normal((n_samples, r)) @ np.diag(theta_t)
    H = np.sqrt(sigma_h2) * rng.standard_normal((n_samples, r))
    U = T @ B + H
    E = np.sqrt(sigma_e2) * rng.standard_normal((n_samples, p))
    F = np.sqrt(sigma_f2) * rng.standard_normal((n_samples, q))

    S_signal = T @ W.T
    X = S_signal + E
    Y = U @ C.T + F
    return X, Y, S_signal


def _sigma_hat_e2(X: np.ndarray) -> float:
    N, p = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    return float(np.sum(Xc ** 2) / (N * p))


def _theorem_noise_bound(
    *,
    epsilon: float,
    sigma_e2: float,
    N: int,
    p: int,
    delta: float,
) -> float:
    """Theorem bound (Eq. noise_error_bound) for |sigma_hat - sigma_e2|."""
    sigma_e = float(np.sqrt(sigma_e2))
    term1 = (epsilon ** 2) / p
    term2 = (2.0 * sigma_e * epsilon) / (np.sqrt(N) * p)
    term3 = sigma_e2 * np.sqrt((2.0 / (N * p)) * np.log(4.0 / delta))
    term4 = sigma_e2 / N
    return float(term1 + term2 + term3 + term4)


def _align_signs(W_est: np.ndarray, C_est: np.ndarray, B_est: np.ndarray,
                 W_true: np.ndarray, C_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    W = W_est.copy()
    C = C_est.copy()
    B = B_est.copy()
    r = W_true.shape[1]

    for i in range(r):
        wc = np.corrcoef(W[:, i], W_true[:, i])[0, 1]
        cc = np.corrcoef(C[:, i], C_true[:, i])[0, 1]
        if np.isfinite(wc) and wc < 0:
            W[:, i] *= -1
        if np.isfinite(cc) and cc < 0:
            C[:, i] *= -1

    b = np.diag(B)
    B = np.diag(np.abs(b))
    return W, C, B


def _compute_mse_all(params_est: Dict, params_true: Dict) -> Dict[str, float]:
    W, C, B = _align_signs(params_est["W"], params_est["C"], params_est["B"],
                           params_true["W"], params_true["C"])

    out = {
        "mse_W": float(np.mean((W - params_true["W"]) ** 2)),
        "mse_C": float(np.mean((C - params_true["C"]) ** 2)),
        "mse_B": float(np.mean((np.diag(B) - np.diag(params_true["B"])) ** 2)),
        "mse_Sigma_t": float(np.mean((np.diag(params_est["Sigma_t"]) - np.diag(params_true["Sigma_t"])) ** 2)),
        "mse_sigma_h2": float((float(params_est["sigma_h2"]) - float(params_true["sigma_h2"])) ** 2),
        "mse_sigma_e2": float((float(params_est.get("sigma_e2", np.nan)) - float(params_true["sigma_e2"])) ** 2),
        "mse_sigma_f2": float((float(params_est.get("sigma_f2", np.nan)) - float(params_true["sigma_f2"])) ** 2),
    }
    return out


def experiment_1_preestimate_accuracy(
    *,
    output_dir: str,
    seed: int,
    delta: float,
    M: int,
    p: int,
    q: int,
    r: int,
    Ns: List[int],
    snr_sigma_t_diags: Dict[str, float],
    sigma_e2: float,
    sigma_f2: float,
    sigma_h2: float,
) -> None:
    out_dir = _ensure_dir(os.path.join(output_dir, "exp1_preestimate_accuracy"))

    noise = NoiseLevel("fixed", sigma_e2=sigma_e2, sigma_f2=sigma_f2, sigma_h2=sigma_h2)

    rows = []
    for snr_idx, (snr_name, sigma_t_diag) in enumerate(snr_sigma_t_diags.items()):
        params = _generate_params_fixed_loadings(p, q, r, sigma_t_diag, noise, seed=seed)

        for N in Ns:
            rel_errors = []
            rel_bounds = []
            epsilons = []

            for m in range(M):
                rng = np.random.default_rng(seed + 10000 * snr_idx + 37 * N + m)
                X, _, S_signal = _sample_ppls(params, n_samples=N, rng=rng)

                sigma_hat = _sigma_hat_e2(X)
                rel_err = abs(sigma_hat - sigma_e2) / sigma_e2

                s_bar = S_signal.mean(axis=0, keepdims=True)
                epsilon = float(np.max(np.linalg.norm(S_signal - s_bar, axis=1)))
                bound = _theorem_noise_bound(epsilon=epsilon, sigma_e2=sigma_e2, N=N, p=p, delta=delta)
                rel_bound = bound / sigma_e2

                rel_errors.append(rel_err)
                rel_bounds.append(rel_bound)
                epsilons.append(epsilon)

            rows.append({
                "snr": snr_name,
                "sigma_t_diag": sigma_t_diag,
                "N": N,
                "M": M,
                "rel_error_mean": float(np.mean(rel_errors)),
                "rel_error_std": float(np.std(rel_errors)),
                "rel_bound_mean": float(np.mean(rel_bounds)),
                "rel_bound_std": float(np.std(rel_bounds)),
                "epsilon_mean": float(np.mean(epsilons)),
                "epsilon_std": float(np.std(epsilons)),
            })

    df = pd.DataFrame(rows).sort_values(["snr", "N"])
    df.to_csv(os.path.join(out_dir, "exp1_preestimate_accuracy.csv"), index=False)

    # Plot: relative error vs N, overlay theoretical bound (dashed)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for snr_name in snr_sigma_t_diags.keys():
        sub = df[df["snr"] == snr_name]
        ax.plot(sub["N"], sub["rel_error_mean"], marker="o", label=f"{snr_name}: empirical")
        ax.plot(sub["N"], sub["rel_bound_mean"], linestyle="--", marker=None, label=f"{snr_name}: Theorem bound")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel(r"Relative error $|\hat{\sigma}_e^2-\sigma_e^2|/\sigma_e^2$")
    ax.set_title(r"Noise pre-estimation accuracy vs $N$ (with Theorem bound)")
    ax.legend(ncol=1, fontsize=9)
    ax.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp1_preestimate_accuracy.png"), dpi=300)
    plt.close(fig)

    with open(os.path.join(out_dir, "exp1_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "seed": seed,
            "delta": delta,
            "M": M,
            "p": p,
            "q": q,
            "r": r,
            "Ns": Ns,
            "snr_sigma_t_diags": snr_sigma_t_diags,
            "sigma_e2": sigma_e2,
            "sigma_f2": sigma_f2,
            "sigma_h2": sigma_h2,
        }, f, indent=2)


def experiment_2_ablation_joint_vs_fixed(
    *,
    output_dir: str,
    seed: int,
    M: int,
    p: int,
    q: int,
    r: int,
    N: int,
    n_starts: int,
    slm_max_iter: int,
    optimizer: str,
    gtol: float,
    xtol: float,
    barrier_tol: float,
    noise_levels: List[NoiseLevel],
) -> None:

    out_dir = _ensure_dir(os.path.join(output_dir, "exp2_joint_vs_fixed"))

    _log(
        f"[exp2] start | p=q={p}, r={r}, N={N}, M={M}, n_starts={n_starts}, slm_max_iter={slm_max_iter}"
    )

    # Fixed true parameters (same W/C/B/Sigma_t across noise levels)
    base_params = _generate_params_fixed_loadings(p, q, r, sigma_t_diag=1.0, noise=noise_levels[0], seed=seed)
    base_params["sigma_e2"] = noise_levels[0].sigma_e2
    base_params["sigma_f2"] = noise_levels[0].sigma_f2
    base_params["sigma_h2"] = noise_levels[0].sigma_h2

    init_gen = InitialPointGenerator(p=p, q=q, r=r, n_starts=n_starts, random_seed=seed)
    base_starting_points = init_gen.generate_starting_points()

    model = PPLSModel(p, q, r)

    summary_rows = []

    for noise_idx, noise in enumerate(noise_levels):
        t_noise0 = time.perf_counter()
        _log(f"[exp2] noise={noise.name} start ({noise_idx+1}/{len(noise_levels)})")

        params_true = dict(base_params)
        params_true["sigma_e2"] = noise.sigma_e2
        params_true["sigma_f2"] = noise.sigma_f2
        params_true["sigma_h2"] = noise.sigma_h2

        raw_path = os.path.join(out_dir, f"exp2_raw_{noise.name}.csv")
        rows_A: List[Dict] = []
        rows_B: List[Dict] = []

        # Resume support: if raw CSV exists, continue from next rep.
        start_rep = 0
        if os.path.exists(raw_path):
            try:
                existing = pd.read_csv(raw_path)
                if "rep" in existing.columns and len(existing):
                    start_rep = int(existing["rep"].max()) + 1
                    _log(f"[exp2] resume detected: {raw_path} (start_rep={start_rep})")
            except Exception:
                pass

        rep_times: List[float] = []

        for m in range(start_rep, M):
            t_rep0 = time.perf_counter()
            _log(f"[exp2] noise={noise.name} rep {m+1}/{M} start")

            rng = np.random.default_rng(seed + 7777 + 1000 * noise_idx + 13 * m)
            X, Y, _ = _sample_ppls(params_true, n_samples=N, rng=rng)

            # --- Scheme A: pre-estimate and fix ---
            _log(f"[exp2]   A_fixed start")
            slm_A = ScalarLikelihoodMethod(
                p=p,
                q=q,
                r=r,
                max_iter=slm_max_iter,
                optimizer="trust-constr",
                use_noise_preestimation=True,
                optimize_noise_variances=False,
            )
            t0 = time.perf_counter()
            res_A = slm_A.fit(X, Y, base_starting_points)
            tA = time.perf_counter() - t0
            _log(
                f"[exp2]   A_fixed done | success={bool(res_A.get('success', False))} | "
                f"iters={int(res_A.get('n_iterations', 0))} | {tA:.1f}s"
            )

            # --- Scheme B: joint optimisation (sigma_e2/sigma_f2 included in theta) ---
            _log(f"[exp2]   B_joint start")
            slm_B = ScalarLikelihoodMethod(
                p=p,
                q=q,
                r=r,
                max_iter=slm_max_iter,
                optimizer=optimizer,
                use_noise_preestimation=True,
                optimize_noise_variances=True,
                gtol=gtol,
                xtol=xtol,
                barrier_tol=barrier_tol,
            )

            t0 = time.perf_counter()
            res_B = slm_B.fit(X, Y, base_starting_points)
            tB = time.perf_counter() - t0
            _log(
                f"[exp2]   B_joint done | success={bool(res_B.get('success', False))} | "
                f"iters={int(res_B.get('n_iterations', 0))} | {tB:.1f}s"
            )

            t_rep = time.perf_counter() - t_rep0
            rep_times.append(float(t_rep))
            mean_rep = float(np.mean(rep_times)) if rep_times else float('nan')
            remaining = (M - (m + 1))
            eta_min = (mean_rep * remaining) / 60.0 if np.isfinite(mean_rep) else float('nan')
            _log(f"[exp2] noise={noise.name} rep {m+1}/{M} done | rep={t_rep:.1f}s | ETA~{eta_min:.1f} min")


            # compute comparable final L via matrix form
            def _final_L(res: Dict) -> float:
                XY = np.hstack([X, Y])
                XY_c = XY - XY.mean(axis=0)
                S = (XY_c.T @ XY_c) / N
                Sigma = model.compute_covariance_matrix(
                    res["W"], res["C"], res["B"], res["Sigma_t"],
                    float(res["sigma_e2"]), float(res["sigma_f2"]), float(res["sigma_h2"])
                )
                return float(model.log_likelihood_matrix(S, Sigma))

            mse_A = _compute_mse_all(res_A, params_true)
            mse_B = _compute_mse_all(res_B, params_true)

            rows_A.append({
                "rep": m,
                "scheme": "A_fixed",
                "noise": noise.name,
                "success": int(bool(res_A.get("success", False))),
                "objective_value": float(res_A.get("objective_value", np.nan)),
                "final_L": _final_L(res_A),
                "n_iterations": int(res_A.get("n_iterations", 0)),
                "runtime_sec": float(tA),
                **mse_A,
            })

            rows_B.append({
                "rep": m,
                "scheme": "B_joint",
                "noise": noise.name,
                "success": int(bool(res_B.get("success", False))),
                "objective_value": float(res_B.get("objective_value", np.nan)),
                "final_L": _final_L(res_B),
                "n_iterations": int(res_B.get("n_iterations", 0)),
                "runtime_sec": float(tB),
                **mse_B,
            })

            # Incremental save so long runs are not lost.
            try:
                # Merge with existing file if present.
                if os.path.exists(raw_path):
                    prev = pd.read_csv(raw_path)
                    df_all = pd.concat([prev, pd.DataFrame([rows_A[-1], rows_B[-1]])], ignore_index=True)
                else:
                    df_all = pd.DataFrame(rows_A + rows_B)
                df_all.to_csv(raw_path, index=False)
            except Exception:
                pass

            # (No extra periodic log here; we already log start/done + ETA for every rep.)


        # Final load for aggregation
        df_all = pd.read_csv(raw_path) if os.path.exists(raw_path) else pd.DataFrame(rows_A + rows_B)
        dfA = df_all[df_all["scheme"] == "A_fixed"].copy()
        dfB = df_all[df_all["scheme"] == "B_joint"].copy()

        # Keep deterministic row order
        dfA = dfA.sort_values("rep")
        dfB = dfB.sort_values("rep")

        t_noise = time.perf_counter() - t_noise0
        _log(f"[exp2] noise={noise.name} finished in {t_noise/60:.1f} min")


        def _agg(df: pd.DataFrame) -> Dict[str, float]:
            # aggregate over successful runs only
            df_ok = df[df["success"] == 1]
            out = {"success_rate": float(df["success"].mean())}
            for col in [
                "final_L", "objective_value", "n_iterations", "runtime_sec",
                "mse_W", "mse_C", "mse_B", "mse_Sigma_t", "mse_sigma_h2", "mse_sigma_e2", "mse_sigma_f2",
            ]:
                out[f"{col}_mean"] = float(df_ok[col].mean()) if len(df_ok) else float("nan")
                out[f"{col}_std"] = float(df_ok[col].std(ddof=0)) if len(df_ok) else float("nan")
            return out

        aggA = _agg(dfA)
        aggB = _agg(dfB)

        summary_rows.append({"noise": noise.name, "scheme": "A_fixed", **aggA})
        summary_rows.append({"noise": noise.name, "scheme": "B_joint", **aggB})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(out_dir, "exp2_summary_table.csv"), index=False)

    # Plot a compact comparison figure for the paper.
    # We visualise (i) success rate and (ii) mean runtime (including failed runs), by noise level and scheme.
    try:
        dfs = []
        for noise in noise_levels:
            raw_path = os.path.join(out_dir, f"exp2_raw_{noise.name}.csv")
            if os.path.exists(raw_path):
                df = pd.read_csv(raw_path)
                dfs.append(df)
        if dfs:
            df_all = pd.concat(dfs, ignore_index=True)

            agg_all = df_all.groupby(["noise", "scheme"], as_index=False).agg(
                success_rate=("success", "mean"),
                runtime_sec_mean=("runtime_sec", "mean"),
            )

            fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))

            # success rate
            ax = axes[0]
            for scheme in ["A_fixed", "B_joint"]:
                sub = agg_all[agg_all["scheme"] == scheme].set_index("noise")
                xs = list(sub.index)
                ys = sub["success_rate"].values
                ax.plot(xs, ys, marker="o", label=scheme)
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel("Success rate")
            ax.set_title("Exp2: success rate")
            ax.grid(True, axis="y", alpha=0.25)
            ax.legend(fontsize=9)

            # runtime
            ax = axes[1]
            for scheme in ["A_fixed", "B_joint"]:
                sub = agg_all[agg_all["scheme"] == scheme].set_index("noise")
                xs = list(sub.index)
                ys = sub["runtime_sec_mean"].values
                ax.plot(xs, ys, marker="o", label=scheme)
            ax.set_ylabel("Mean runtime (sec)")
            ax.set_title("Exp2: runtime")
            ax.grid(True, axis="y", alpha=0.25)

            fig.suptitle("Noise ablation Exp2: fixed vs joint optimisation", y=1.02)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "exp2_joint_vs_fixed.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)
    except Exception:
        pass

    with open(os.path.join(out_dir, "exp2_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "seed": seed,
            "M": M,
            "p": p,
            "q": q,
            "r": r,
            "N": N,
            "n_starts": n_starts,
            "slm_max_iter": slm_max_iter,
            "optimizer": optimizer,
            "gtol": gtol,
            "xtol": xtol,
            "barrier_tol": barrier_tol,
            "noise_levels": [noise.__dict__ for noise in noise_levels],
        }, f, indent=2)






def main() -> None:
    parser = argparse.ArgumentParser(description="Noise pre-estimation ablation experiments")
    parser.add_argument("--output_dir", type=str, default="results_noise_ablation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run", type=str, default="all", choices=["all", "exp1", "exp2"])


    # Experiment-scale knobs (keep defaults close to your spec, but allow overrides)
    parser.add_argument("--exp1_M", type=int, default=50)
    parser.add_argument("--exp2_M", type=int, default=50)


    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run a speed-focused version (primarily affects exp2: smaller p/q/r/N, fewer starts/iters, looser tolerances)",
    )

    # Exp2 size/compute knobs (exp2 is the expensive one)
    parser.add_argument("--exp2_p", type=int, default=50)
    parser.add_argument("--exp2_q", type=int, default=50)
    parser.add_argument("--exp2_r", type=int, default=5)
    parser.add_argument("--exp2_N", type=int, default=500)

    parser.add_argument("--exp2_n_starts", type=int, default=16)
    parser.add_argument("--slm_max_iter", type=int, default=100)


    # Exp2 optimiser knobs
    parser.add_argument("--exp2_optimizer", type=str, default="trust-constr")
    parser.add_argument("--exp2_gtol", type=float, default=1e-2)
    parser.add_argument("--exp2_xtol", type=float, default=1e-2)
    parser.add_argument("--exp2_barrier_tol", type=float, default=1e-2)

    parser.add_argument("--theorem_delta", type=float, default=0.05)


    args = parser.parse_args()

    out_dir = _ensure_dir(args.output_dir)
    _set_global_seed(args.seed)

    # If user asks for a fast run, shrink only the expensive parts (exp2), unless explicitly overridden.
    if bool(getattr(args, "fast", False)):
        if args.exp2_M == 50:
            args.exp2_M = 3
        if args.exp2_p == 50:
            args.exp2_p = 20
        if args.exp2_q == 50:
            args.exp2_q = 20
        if args.exp2_r == 5:
            args.exp2_r = 3
        if args.exp2_N == 500:
            args.exp2_N = 300
        if args.exp2_n_starts == 16:
            args.exp2_n_starts = 4
        if args.slm_max_iter == 100:
            args.slm_max_iter = 40

    _log(
        f"[OK] noise ablation runner | run={args.run} | output_dir={os.path.abspath(out_dir)} | "
        f"fast={bool(getattr(args, 'fast', False))}"
    )


    if args.run in ("all", "exp1"):
        experiment_1_preestimate_accuracy(
            output_dir=out_dir,
            seed=args.seed,
            delta=float(args.theorem_delta),
            M=int(args.exp1_M),
            p=100,
            q=100,
            r=5,
            Ns=[100, 500, 2000, 5000],
            snr_sigma_t_diags={
                "lowSNR": 0.1,
                "midSNR": 1.0,
                "highSNR": 10.0,
            },
            sigma_e2=0.1,
            sigma_f2=0.1,
            sigma_h2=0.05,
        )

    if args.run in ("all", "exp2"):
        experiment_2_ablation_joint_vs_fixed(
            output_dir=out_dir,
            seed=args.seed,
            M=int(args.exp2_M),
            p=int(args.exp2_p),
            q=int(args.exp2_q),
            r=int(args.exp2_r),
            N=int(args.exp2_N),
            n_starts=int(args.exp2_n_starts),
            slm_max_iter=int(args.slm_max_iter),
            optimizer=str(args.exp2_optimizer),
            gtol=float(args.exp2_gtol),
            xtol=float(args.exp2_xtol),
            barrier_tol=float(args.exp2_barrier_tol),
            noise_levels=[
                NoiseLevel("low", sigma_e2=0.1, sigma_f2=0.1, sigma_h2=0.05),
                NoiseLevel("high", sigma_e2=0.5, sigma_f2=0.5, sigma_h2=0.25),
            ],
        )




    print(f"[OK] Done. Outputs written to: {os.path.abspath(out_dir)}")



if __name__ == "__main__":
    main()

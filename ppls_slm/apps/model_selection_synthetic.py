"""Synthetic model selection experiment for latent dimension r.

This script demonstrates how to select r via:
- BIC computed from the observed-data log-likelihood (PPLS advantage)
- K-fold CV using prediction MSE

Outputs are written under an output directory (default: results_model_selection/synthetic),
so they can be synced into `paper/artifacts/` via `scripts/sync_artifacts.py`.
"""


from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from typing import Dict, Iterable, List, Tuple




import numpy as np
import pandas as pd

from ppls_slm.algorithms import InitialPointGenerator, ScalarLikelihoodMethod
from ppls_slm.data_generator import SineDataGenerator
from ppls_slm.model_selection import CVResult, compute_bic, select_r_by_cv_prediction_mse


def _make_starting_points(*, p: int, q: int, r: int, n_starts: int, seed: int) -> List[np.ndarray]:
    gen = InitialPointGenerator(int(p), int(q), int(r), n_starts=int(n_starts), random_seed=int(seed))
    return gen.generate_starting_points()


def _fit_slm_fixed_noise(
    X: np.ndarray,
    Y: np.ndarray,
    r: int,
    *,
    n_starts: int,
    max_iter: int,
    seed: int,
    verbose: bool,
    slm_progress_every: int,
    starts: List[np.ndarray] | None = None,
) -> Dict:
    p, q = X.shape[1], Y.shape[1]

    if starts is None:
        starts2 = _make_starting_points(p=int(p), q=int(q), r=int(r), n_starts=int(n_starts), seed=int(seed))
    else:
        # Be defensive: some optimisers may mutate arrays in-place.
        starts2 = [np.array(s, copy=True) for s in starts]

    slm = ScalarLikelihoodMethod(
        int(p),
        int(q),
        int(r),
        max_iter=int(max_iter),
        use_noise_preestimation=True,
        optimize_noise_variances=False,
        verbose=bool(verbose),
        progress_every=int(slm_progress_every),
    )
    return slm.fit(X, Y, starts2)





def _generate_params_for_r(
    *,
    p: int,
    q: int,
    r: int,
    n_samples: int,
    sigma_e2: float,
    sigma_f2: float,
    sigma_h2: float,
    rng_seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    # Match the paper's simulation style (sine templates + Gram-Schmidt),
    # but use the requested diagonal schedules to make identifiability explicit.
    np.random.seed(int(rng_seed))

    gen = SineDataGenerator(p=p, q=q, r=r, n_samples=n_samples, random_seed=int(rng_seed))
    params = gen.generate_true_parameters(sigma_e2=sigma_e2, sigma_f2=sigma_f2, sigma_h2=sigma_h2)

    # Override B and Sigma_t schedules per task spec.
    # Sigma_t diag: [r, r-1, ..., 1]
    theta = np.arange(r, 0, -1, dtype=float)
    params["Sigma_t"] = np.diag(theta)

    # B diag: [0.9, 0.8, ..., 0.9-0.1*(r-1)]
    b = 0.9 - 0.1 * np.arange(r, dtype=float)
    b = np.maximum(b, 0.05)
    params["B"] = np.diag(b)

    # Quick sanity: enforce strictly decreasing Sigma_t_ii * B_ii
    prod = np.diag(params["Sigma_t"]) * np.diag(params["B"])
    if not np.all(prod[:-1] > prod[1:]):
        # If numerical tie occurs (rare), jitter slightly.
        prod_fix = prod.copy()
        for i in range(1, len(prod_fix)):
            if prod_fix[i - 1] <= prod_fix[i]:
                prod_fix[i] = prod_fix[i - 1] - 1e-3
        # adjust Sigma_t to restore ordering
        params["Sigma_t"] = np.diag(prod_fix / np.diag(params["B"]))

    X, Y = gen.generate_samples(params)
    return X, Y, params


def _format_distribution(counts: Counter, r_candidates: List[int]) -> str:
    parts = []
    for r in r_candidates:
        c = int(counts.get(int(r), 0))
        if c:
            parts.append(f"r={int(r)}:{c}")
    # include zeros? keep compact
    return ", ".join(parts) if parts else "(all failed)"


def _run_one_trial(
    *,
    N: int,
    t: int,
    p: int,
    q: int,
    r_true: int,
    r_cands: List[int],
    sigma_e2: float,
    sigma_f2: float,
    sigma_h2: float,
    seed: int,
    slm_n_starts: int,
    slm_max_iter: int,
    cv_folds: int,
    slm_progress_every: int,
) -> Tuple[List[dict], List[dict], int, int]:
    """Run a single (N, trial) and return rows + selected r's.

    Returns:
      (rows_bic, rows_cv, best_r_bic, best_r_cv)
    """

    trial_seed = int(seed) + 10_000 * int(N) + 100 * int(t)

    X, Y, _true_params = _generate_params_for_r(
        p=int(p),
        q=int(q),
        r=int(r_true),
        n_samples=int(N),
        sigma_e2=float(sigma_e2),
        sigma_f2=float(sigma_f2),
        sigma_h2=float(sigma_h2),
        rng_seed=int(trial_seed),
    )

    rows_bic: List[dict] = []
    bic_by_r: Dict[int, float] = {}

    for r in r_cands:
        try:
            params_hat = _fit_slm_fixed_noise(
                X,
                Y,
                int(r),
                n_starts=int(slm_n_starts),
                max_iter=int(slm_max_iter),
                seed=int(trial_seed) + int(r),
                verbose=False,
                slm_progress_every=int(slm_progress_every),
            )
            bic, ll = compute_bic(X, Y, params_hat, r=int(r))
        except Exception:
            bic, ll = float("inf"), float("-inf")

        bic_by_r[int(r)] = float(bic)
        rows_bic.append(
            {
                "N": int(N),
                "trial": int(t),
                "r": int(r),
                "bic": float(bic),
                "log_likelihood": float(ll),
            }
        )

    best_r_bic = int(min(bic_by_r, key=bic_by_r.get))

    # CV does many fits per r (one per fold). Generating identical multi-start points
    # repeatedly is wasted work, so we cache them per r and clone per fit.
    cv_starts_by_r: Dict[int, List[np.ndarray]] = {
        int(r): _make_starting_points(
            p=int(p),
            q=int(q),
            r=int(r),
            n_starts=int(slm_n_starts),
            seed=int(trial_seed) + 1000 + int(r),
        )
        for r in r_cands
    }

    cv_res: CVResult = select_r_by_cv_prediction_mse(
        X,
        Y,
        r_candidates=[int(r) for r in r_cands],
        fit_fn=lambda Xt, Yt, rr: _fit_slm_fixed_noise(
            Xt,
            Yt,
            int(rr),
            n_starts=int(slm_n_starts),
            max_iter=int(slm_max_iter),
            seed=int(trial_seed) + 1000 + int(rr),
            verbose=False,
            slm_progress_every=int(slm_progress_every),
            starts=cv_starts_by_r[int(rr)],
        ),
        n_folds=int(cv_folds),
        random_state=int(trial_seed % (2**31 - 1)),
    )


    rows_cv: List[dict] = []
    for r in r_cands:
        rows_cv.append(
            {
                "N": int(N),
                "trial": int(t),
                "r": int(r),
                "cv_mse": float(cv_res.cv_mse[int(r)]),
                "cv_mse_std_over_folds": float(cv_res.cv_mse_std[int(r)]),
            }
        )

    return rows_bic, rows_cv, int(best_r_bic), int(cv_res.best_r)


def run_experiment(

    *,
    p: int,
    q: int,
    r_true: int,
    n_trials: int,
    n_list: Iterable[int],
    r_candidates: Iterable[int],
    sigma_e2: float,
    sigma_f2: float,
    sigma_h2: float,
    seed: int,
    slm_n_starts: int,
    slm_max_iter: int,
    cv_folds: int,
    verbose: bool,
    slm_progress_every: int,
    parallel_trials: bool = False,
    n_jobs: int = 0,
    checkpoint_dir: str | None = None,
    heartbeat_sec: int = 30,
    resume: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:


    r_cands = [int(x) for x in r_candidates]
    n_list2 = [int(n) for n in n_list]

    # Per-trial records for curve aggregation
    rows_bic: List[dict] = []
    rows_cv: List[dict] = []
    rows_sel: List[dict] = []

    if int(n_jobs) <= 0:
        cpu = os.cpu_count() or 1
        n_jobs = max(1, min(cpu - 1, 8))

    # Optional checkpointing (useful because full runs can take a long time).
    bic_ck_f = None
    cv_ck_f = None
    bic_writer = None
    cv_writer = None

    completed_trials: set[tuple[int, int]] = set()

    if checkpoint_dir is not None:
        os.makedirs(str(checkpoint_dir), exist_ok=True)
        bic_ck_path = os.path.join(str(checkpoint_dir), "bic_per_trial.partial.csv")
        cv_ck_path = os.path.join(str(checkpoint_dir), "cv_mse_per_trial.partial.csv")

        # Resume support: load existing partial rows and append only missing trials.
        if bool(resume) and os.path.exists(bic_ck_path) and os.path.exists(cv_ck_path):
            try:
                if os.path.getsize(bic_ck_path) > 0:
                    df_prev_bic = pd.read_csv(bic_ck_path)
                    df_prev_bic = df_prev_bic.dropna(subset=["N", "trial"]).copy()
                    if not df_prev_bic.empty:
                        df_prev_bic["N"] = df_prev_bic["N"].astype(int)
                        df_prev_bic["trial"] = df_prev_bic["trial"].astype(int)
                        df_prev_bic["r"] = df_prev_bic["r"].astype(int)
                        rows_bic.extend(df_prev_bic.to_dict("records"))
                else:
                    df_prev_bic = pd.DataFrame()

                if os.path.getsize(cv_ck_path) > 0:
                    df_prev_cv = pd.read_csv(cv_ck_path)
                    df_prev_cv = df_prev_cv.dropna(subset=["N", "trial"]).copy()
                    if not df_prev_cv.empty:
                        df_prev_cv["N"] = df_prev_cv["N"].astype(int)
                        df_prev_cv["trial"] = df_prev_cv["trial"].astype(int)
                        df_prev_cv["r"] = df_prev_cv["r"].astype(int)
                        rows_cv.extend(df_prev_cv.to_dict("records"))
                else:
                    df_prev_cv = pd.DataFrame()


                if ("N" in df_prev_bic.columns) and ("trial" in df_prev_bic.columns) and ("N" in df_prev_cv.columns) and ("trial" in df_prev_cv.columns):
                    done_bic = set((int(n), int(t)) for n, t in zip(df_prev_bic["N"], df_prev_bic["trial"]))
                    done_cv = set((int(n), int(t)) for n, t in zip(df_prev_cv["N"], df_prev_cv["trial"]))
                    completed_trials = done_bic & done_cv
            except Exception:
                # If partial files are corrupted, fall back to a fresh run.
                completed_trials = set()

        bic_mode = "a" if (bool(resume) and os.path.exists(bic_ck_path) and os.path.getsize(bic_ck_path) > 0) else "w"
        cv_mode = "a" if (bool(resume) and os.path.exists(cv_ck_path) and os.path.getsize(cv_ck_path) > 0) else "w"

        bic_ck_f = open(bic_ck_path, bic_mode, newline="", encoding="utf-8")
        cv_ck_f = open(cv_ck_path, cv_mode, newline="", encoding="utf-8")

        bic_writer = csv.DictWriter(
            bic_ck_f,
            fieldnames=["N", "trial", "r", "bic", "log_likelihood"],
        )
        cv_writer = csv.DictWriter(
            cv_ck_f,
            fieldnames=["N", "trial", "r", "cv_mse", "cv_mse_std_over_folds"],
        )

        if bic_mode == "w":
            bic_writer.writeheader()
        if cv_mode == "w":
            cv_writer.writeheader()

        bic_ck_f.flush()
        cv_ck_f.flush()

    try:
        for N in n_list2:
            # Decide which trial indices still need running for this N.
            all_trials = list(range(int(n_trials)))
            trials_todo = [t for t in all_trials if (int(N), int(t)) not in completed_trials]




            already_done = int(n_trials) - int(len(trials_todo))

            if verbose:
                mode = "parallel" if bool(parallel_trials) else "serial"
                print(
                    f"[ModelSelection/Synthetic] N={int(N)} (trials={int(n_trials)}, folds={int(cv_folds)}, r={r_cands}, {mode}, n_jobs={int(n_jobs)})",
                    flush=True,
                )
                if bool(resume) and (already_done > 0):
                    print(
                        f"  resume: {already_done}/{int(n_trials)} trials already done; running {len(trials_todo)} remaining",
                        flush=True,
                    )

            if bool(parallel_trials):
                import multiprocessing
                import time

                heartbeat = max(1, int(heartbeat_sec))
                mp_ctx = multiprocessing.get_context("spawn")

                if verbose:
                    print(f"  (heartbeat every {heartbeat}s)", flush=True)

                done_new = 0
                t0 = time.time()

                with ProcessPoolExecutor(max_workers=int(n_jobs), mp_context=mp_ctx) as ex:
                    fut_to_trial: Dict = {}
                    fut_start: Dict = {}

                    for t in trials_todo:
                        fut = ex.submit(
                            _run_one_trial,
                            N=int(N),
                            t=int(t),
                            p=int(p),
                            q=int(q),
                            r_true=int(r_true),
                            r_cands=r_cands,
                            sigma_e2=float(sigma_e2),
                            sigma_f2=float(sigma_f2),
                            sigma_h2=float(sigma_h2),
                            seed=int(seed),
                            slm_n_starts=int(slm_n_starts),
                            slm_max_iter=int(slm_max_iter),
                            cv_folds=int(cv_folds),
                            slm_progress_every=int(slm_progress_every),
                        )
                        fut_to_trial[fut] = int(t)
                        fut_start[fut] = time.time()

                    if verbose:
                        print(f"  Submitted {len(fut_to_trial)}/{len(trials_todo)} remaining trial jobs.", flush=True)

                    pending = set(fut_to_trial.keys())
                    last_beat = time.time()
                    trial_durations: List[float] = []

                    while pending:
                        done, pending = wait(pending, timeout=heartbeat, return_when=FIRST_COMPLETED)

                        for fut in done:
                            t_done = int(fut_to_trial.get(fut, -1))
                            try:
                                rb, rc, _best_bic, _best_cv = fut.result()
                            except Exception as e:
                                if verbose:
                                    print(f"  [FAIL] trial {t_done + 1}/{int(n_trials)}: {e}", flush=True)
                                continue

                            rows_bic.extend(rb)
                            rows_cv.extend(rc)
                            completed_trials.add((int(N), int(t_done)))

                            if bic_writer is not None and cv_writer is not None:
                                bic_writer.writerows(rb)
                                cv_writer.writerows(rc)
                                bic_ck_f.flush()
                                cv_ck_f.flush()

                            done_new += 1
                            trial_durations.append(time.time() - float(fut_start.get(fut, t0)))

                            if verbose:
                                print(
                                    f"  done {already_done + done_new}/{int(n_trials)} (trial {t_done + 1})",
                                    flush=True,
                                )

                        now = time.time()
                        if verbose and pending and (now - last_beat) >= heartbeat:
                            elapsed = now - t0
                            remaining = max(0, len(trials_todo) - done_new)

                            # Simple ETA based on completed trials.
                            if trial_durations:
                                avg = float(np.mean(trial_durations))
                                eta = avg * remaining
                                eta_s = f", ETA ~{eta/60:.1f} min"
                            else:
                                eta_s = ""

                            running = [now - float(fut_start.get(f, now)) for f in pending]
                            longest = max(running) if running else 0.0
                            print(
                                f"  heartbeat: {already_done + done_new}/{int(n_trials)} done; {len(pending)} running; elapsed {elapsed/60:.1f} min; longest {longest/60:.1f} min{eta_s}",
                                flush=True,
                            )
                            last_beat = now


            else:
                for t in trials_todo:
                    if verbose:
                        print(f"  trial {int(t) + 1}/{int(n_trials)}", flush=True)

                    rb, rc, _best_bic, _best_cv = _run_one_trial(
                        N=int(N),
                        t=int(t),
                        p=int(p),
                        q=int(q),
                        r_true=int(r_true),
                        r_cands=r_cands,
                        sigma_e2=float(sigma_e2),
                        sigma_f2=float(sigma_f2),
                        sigma_h2=float(sigma_h2),
                        seed=int(seed),
                        slm_n_starts=int(slm_n_starts),
                        slm_max_iter=int(slm_max_iter),
                        cv_folds=int(cv_folds),
                        slm_progress_every=int(slm_progress_every),
                    )
                    rows_bic.extend(rb)
                    rows_cv.extend(rc)
                    completed_trials.add((int(N), int(t)))

                    if bic_writer is not None and cv_writer is not None:
                        bic_writer.writerows(rb)
                        cv_writer.writerows(rc)
                        bic_ck_f.flush()
                        cv_ck_f.flush()

            # Per-N selection summary (recomputed from all rows for this N).
            df_bic_N = pd.DataFrame([r for r in rows_bic if int(r.get("N", -1)) == int(N)])
            df_cv_N = pd.DataFrame([r for r in rows_cv if int(r.get("N", -1)) == int(N)])

            def _choose_r(df: pd.DataFrame, crit_col: str) -> List[int]:
                if df.empty:
                    return []
                d = df.copy()
                d["trial"] = d["trial"].astype(int)
                d["r"] = d["r"].astype(int)
                d = d.sort_values(["trial", crit_col, "r"], ascending=[True, True, True])
                return d.groupby("trial", sort=True)["r"].first().astype(int).tolist()

            chosen_bic = _choose_r(df_bic_N, "bic")
            chosen_cv = _choose_r(df_cv_N, "cv_mse")

            c_bic = Counter(chosen_bic)
            c_cv = Counter(chosen_cv)


            rows_sel.append(
                {
                    "N": int(N),
                    "r_true": int(r_true),
                    "method": "BIC",
                    "selection_distribution": _format_distribution(c_bic, r_cands),
                    "accuracy": float(c_bic.get(int(r_true), 0)) / float(n_trials),
                    "n_trials": int(n_trials),
                    "cv_folds": int(cv_folds),
                }
            )
            rows_sel.append(
                {
                    "N": int(N),
                    "r_true": int(r_true),
                    "method": "CV",
                    "selection_distribution": _format_distribution(c_cv, r_cands),
                    "accuracy": float(c_cv.get(int(r_true), 0)) / float(n_trials),
                    "n_trials": int(n_trials),
                    "cv_folds": int(cv_folds),
                }
            )

        df_bic = pd.DataFrame(rows_bic)
        df_cv = pd.DataFrame(rows_cv)

        # Build/refresh selection summary from all available trials in the checkpointed rows.
        rows_sel2: List[dict] = []

        def _choose_r(df: pd.DataFrame, crit_col: str) -> List[int]:
            if df.empty:
                return []
            d = df.copy()
            d["trial"] = d["trial"].astype(int)
            d["r"] = d["r"].astype(int)
            d = d.sort_values(["trial", crit_col, "r"], ascending=[True, True, True])
            return d.groupby("trial", sort=True)["r"].first().astype(int).tolist()

        Ns_all = []
        if (not df_bic.empty) and ("N" in df_bic.columns):
            Ns_all = sorted(int(x) for x in df_bic["N"].dropna().unique().tolist())

        for N in Ns_all:
            sub_bic = df_bic[df_bic["N"].astype(int) == int(N)]
            sub_cv = df_cv[df_cv["N"].astype(int) == int(N)] if (not df_cv.empty and "N" in df_cv.columns) else pd.DataFrame()

            chosen_bic = _choose_r(sub_bic, "bic")
            chosen_cv = _choose_r(sub_cv, "cv_mse")

            c_bic = Counter(chosen_bic)
            c_cv = Counter(chosen_cv)

            rows_sel2.append(
                {
                    "N": int(N),
                    "r_true": int(r_true),
                    "method": "BIC",
                    "selection_distribution": _format_distribution(c_bic, r_cands),
                    "accuracy": float(c_bic.get(int(r_true), 0)) / float(n_trials),
                    "n_trials": int(n_trials),
                    "cv_folds": int(cv_folds),
                }
            )
            rows_sel2.append(
                {
                    "N": int(N),
                    "r_true": int(r_true),
                    "method": "CV",
                    "selection_distribution": _format_distribution(c_cv, r_cands),
                    "accuracy": float(c_cv.get(int(r_true), 0)) / float(n_trials),
                    "n_trials": int(n_trials),
                    "cv_folds": int(cv_folds),
                }
            )

        df_sel = pd.DataFrame(rows_sel2)
        return df_sel, df_bic, df_cv


    finally:
        if bic_ck_f is not None:
            try:
                bic_ck_f.close()
            except Exception:
                pass
        if cv_ck_f is not None:
            try:
                cv_ck_f.close()
            except Exception:
                pass




def _plot_curves(df: pd.DataFrame, *, value_col: str, out_path: str, r_true: int, title: str):
    import matplotlib.pyplot as plt

    # Expect columns: N, r, value_col
    Ns = sorted(df["N"].unique().tolist())
    fig, axes = plt.subplots(1, len(Ns), figsize=(5.2 * len(Ns), 4.2), sharey=False)
    if len(Ns) == 1:
        axes = [axes]

    for ax, N in zip(axes, Ns):
        sub = df[df["N"] == N]
        g = sub.groupby("r")[value_col]
        rs = np.array(sorted(g.groups.keys()), dtype=int)
        mean = np.array([g.get_group(int(r)).mean() for r in rs], dtype=float)
        std = np.array([g.get_group(int(r)).std() for r in rs], dtype=float)

        ax.plot(rs, mean, color="#1565C0", linewidth=2.2)
        ax.fill_between(rs, mean - std, mean + std, color="#1565C0", alpha=0.18)
        ax.axvline(int(r_true), color="black", linestyle="--", linewidth=1.5)
        ax.set_title(f"N={int(N)}")
        ax.set_xlabel("r")
        ax.set_ylabel(value_col)
        ax.grid(True, alpha=0.25)

    fig.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Model selection on synthetic PPLS data (BIC + CV)")
    p.add_argument("--output_dir", type=str, default="results_model_selection/synthetic")
    p.add_argument("--seed", type=int, default=42)

    # Defaults tuned for feasible runtime; optimiser budget unchanged.
    p.add_argument("--p", type=int, default=30)
    p.add_argument("--q", type=int, default=30)
    p.add_argument("--r_true", type=int, default=5)
    p.add_argument("--n_trials", type=int, default=5)
    p.add_argument("--n_list", type=str, default="200,500")
    p.add_argument("--r_candidates", type=str, default="1,2,3,4,5,6,7,8")

    p.add_argument("--sigma_e2", type=float, default=0.1)
    p.add_argument("--sigma_f2", type=float, default=0.1)
    p.add_argument("--sigma_h2", type=float, default=0.05)
    p.add_argument("--slm_n_starts", type=int, default=4)
    p.add_argument("--slm_max_iter", type=int, default=100)
    p.add_argument("--slm_progress_every", type=int, default=1)
    p.add_argument("--cv_folds", type=int, default=2)


    p.add_argument("--parallel_trials", action="store_true", help="Parallelise independent trials using processes (recommended).")
    p.add_argument("--n_jobs", type=int, default=0, help="Worker processes for --parallel_trials (0=auto)")
    p.add_argument(
        "--checkpoint",
        action="store_true",
        help="Write partial CSVs (useful to monitor progress of long runs).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from partial CSVs under --output_dir (requires --checkpoint).",
    )
    p.add_argument(
        "--heartbeat_sec",
        type=int,
        default=30,
        help="Heartbeat period (seconds) for progress messages in --parallel_trials mode.",
    )
    p.add_argument("--verbose", action="store_true", help="Print progress (recommended for long runs)")


    p.add_argument(
        "--smoke",
        action="store_true",
        help="Quick smoke run (tiny grid) to validate plumbing; writes to results_model_selection_smoke/synthetic by default.",
    )



    return p.parse_args()







def main():
    args = parse_args()

    default_out_dir = "results_model_selection/synthetic"
    out_dir = str(args.output_dir)

    # Guardrail: smoke runs should not overwrite the full paper outputs by accident.
    if bool(getattr(args, "smoke", False)):
        if out_dir == default_out_dir:
            out_dir = "results_model_selection_smoke/synthetic"

        # Small, quick configuration.
        p_dim, q_dim = 50, 50
        n_trials = 2
        cv_folds = 2
        n_list = [200]
        r_true = 3
        r_candidates = [1, 2, 3]
        slm_n_starts = 4
        slm_max_iter = 100

        parallel_trials = False
        n_jobs = 1



    else:
        p_dim, q_dim = int(args.p), int(args.q)
        n_trials = int(args.n_trials)
        cv_folds = int(args.cv_folds)
        n_list = [int(x.strip()) for x in str(args.n_list).split(",") if x.strip()]
        r_candidates = [int(x.strip()) for x in str(args.r_candidates).split(",") if x.strip()]
        r_true = int(args.r_true)
        slm_n_starts = int(args.slm_n_starts)
        slm_max_iter = int(args.slm_max_iter)
        parallel_trials = bool(args.parallel_trials)
        n_jobs = int(args.n_jobs)





    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures")

    do_checkpoint = bool(getattr(args, "checkpoint", False)) or bool(getattr(args, "resume", False))



    df_sel, df_bic, df_cv = run_experiment(
        p=int(p_dim),
        q=int(q_dim),
        r_true=int(r_true),

        n_trials=int(n_trials),
        n_list=n_list,
        r_candidates=r_candidates,
        sigma_e2=float(args.sigma_e2),
        sigma_f2=float(args.sigma_f2),
        sigma_h2=float(args.sigma_h2),
        seed=int(args.seed),
        slm_n_starts=int(slm_n_starts),
        slm_max_iter=int(slm_max_iter),
        cv_folds=int(cv_folds),
        verbose=bool(args.verbose),
        slm_progress_every=int(args.slm_progress_every),
        parallel_trials=bool(parallel_trials),
        n_jobs=int(n_jobs),
        checkpoint_dir=str(out_dir) if bool(do_checkpoint) else None,
        heartbeat_sec=int(args.heartbeat_sec),
        resume=bool(getattr(args, "resume", False)),


    )







    df_sel.to_csv(os.path.join(out_dir, "selection_accuracy_table.csv"), index=False)
    df_bic.to_csv(os.path.join(out_dir, "bic_per_trial.csv"), index=False)
    df_cv.to_csv(os.path.join(out_dir, "cv_mse_per_trial.csv"), index=False)

    _plot_curves(
        df_bic,
        value_col="bic",
        out_path=os.path.join(fig_dir, "figure_bic_curves.png"),
        r_true=int(r_true),
        title="Synthetic model selection: BIC(r) mean ± std (M trials)",
    )
    _plot_curves(
        df_cv,
        value_col="cv_mse",
        out_path=os.path.join(fig_dir, "figure_cv_mse_curves.png"),
        r_true=int(r_true),
        title=f"Synthetic model selection: {int(cv_folds)}-fold CV-MSE(r) mean ± std (M trials)",
    )



    print("Saved:")
    print(f"  {out_dir}/selection_accuracy_table.csv")
    print(f"  {out_dir}/bic_per_trial.csv")
    print(f"  {out_dir}/cv_mse_per_trial.csv")
    print(f"  {fig_dir}/figure_bic_curves.png")
    print(f"  {fig_dir}/figure_cv_mse_curves.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

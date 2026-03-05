"""Sync reproducible experiment outputs into `paper/artifacts/`.

This script copies the small, paper-relevant outputs (JSON/CSV/PNG/XLSX)
from generated result folders into a stable location tracked with the paper.

Run from repo root (recommended):
    python scripts/sync_artifacts.py

It is safe to re-run; files are overwritten.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path



def copy_file(src: Path, dst: Path) -> bool:
    if not src.exists():
        print(f"[MISS] {src}")
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[COPY] {src} -> {dst}")
    return True


def copy_glob(src_dir: Path, pattern: str, dst_dir: Path) -> int:
    if not src_dir.exists():
        print(f"[MISS] {src_dir} (dir)")
        return 0

    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in sorted(src_dir.glob(pattern)):
        if src.is_file():
            shutil.copy2(src, dst_dir / src.name)
            n += 1
    if n == 0:
        print(f"[MISS] {src_dir}/{pattern}")
    else:
        print(f"[COPY] {n} files: {src_dir}/{pattern} -> {dst_dir}")
    return n


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    artifacts = repo_root / "paper" / "artifacts"

    # --- Simulation / Monte Carlo (ppls_slm.cli.montecarlo) ---
    out = repo_root / "output"
    sim_dir = artifacts / "simulation"

    copied = 0
    copied += int(copy_file(out / "robustness_summary.json", sim_dir / "robustness_summary.json"))
    copied += int(copy_file(out / "results" / "mse_table.json", sim_dir / "mse_table_low.json"))
    copied += int(copy_file(out / "results_high" / "mse_table.json", sim_dir / "mse_table_high.json"))
    copied += int(copy_file(out / "results" / "results_summary.json", sim_dir / "results_summary_low.json"))
    copied += int(copy_file(out / "results_high" / "results_summary.json", sim_dir / "results_summary_high.json"))
    copied += int(copy_file(out / "results" / "experiment_summary.json", sim_dir / "experiment_summary_low.json"))
    copied += int(copy_file(out / "results_high" / "experiment_summary.json", sim_dir / "experiment_summary_high.json"))

    # Figures/tables exported by visualization stage (low-noise only)
    copied += copy_glob(out / "figures", "*.png", sim_dir / "figures")
    copied += copy_glob(out / "figures", "*.xlsx", sim_dir / "figures")

    # --- Speed experiment ---
    # Some workflows keep the figure under paper/ for LaTeX compilation.
    speed_src_candidates = [
        repo_root / "output" / "figures" / "speed_comparison.png",
        repo_root / "speed_comparison.png",
        repo_root / "paper" / "speed_comparison.png",
    ]


    speed_src = next((p for p in speed_src_candidates if p.exists()), None)
    if speed_src is None:
        print(f"[MISS] speed comparison figure (tried: {', '.join(str(p) for p in speed_src_candidates)})")
    else:
        copied += int(copy_file(speed_src, artifacts / "speed" / "speed_comparison.png"))


    # --- Association application ---
    assoc_root = repo_root / "results_association"
    assoc_dir = artifacts / "association"
    copied += int(copy_file(assoc_root / "detection_table.csv", assoc_dir / "detection_table.csv"))
    copied += int(copy_file(assoc_root / "top10_pairs_slm.csv", assoc_dir / "top10_pairs_slm.csv"))
    copied += int(copy_file(assoc_root / "detection_comparison.png", assoc_dir / "detection_comparison.png"))

    # --- Prediction application ---
    pred_root = repo_root / "results_prediction"
    pred_dir = artifacts / "prediction"
    copied += int(copy_file(pred_root / "coverage_results.csv", pred_dir / "coverage_results.csv"))
    copied += int(copy_file(pred_root / "coverage_table.csv", pred_dir / "coverage_table.csv"))
    copied += int(copy_file(pred_root / "calibration_plot.png", pred_dir / "calibration_plot.png"))
    copied += int(copy_file(pred_root / "prediction_example.png", pred_dir / "prediction_example.png"))

    # --- Noise pre-estimation ablation ---
    noise_src_candidates = [
        repo_root / "output" / "noise_ablation",
        repo_root / "output" / "noise_preestimation_ablation",
    ]
    noise_src = next((p for p in noise_src_candidates if p.exists()), None)
    noise_dir = artifacts / "noise_ablation"
    if noise_src is None:
        print(f"[MISS] noise ablation outputs (tried: {', '.join(str(p) for p in noise_src_candidates)})")
    else:
        copied += copy_glob(noise_src / "exp1_preestimate_accuracy", "*", noise_dir / "exp1_preestimate_accuracy")
        copied += copy_glob(noise_src / "exp2_joint_vs_fixed", "*", noise_dir / "exp2_joint_vs_fixed")
        copied += copy_glob(noise_src / "exp3_error_propagation", "*", noise_dir / "exp3_error_propagation")

    # Also generate LaTeX tables from the synced artifacts so the paper stays consistent.
    gen_tables = repo_root / "scripts" / "generate_paper_tables.py"
    if gen_tables.exists():
        try:
            subprocess.check_call([sys.executable, str(gen_tables)], cwd=str(repo_root))
        except Exception as e:
            print(f"[WARN] table generation failed: {e}")
    else:
        print(f"[WARN] missing generator script: {gen_tables}")

    print("\n" + "=" * 72)
    print(f"Done. Copied {copied} file(s) into {artifacts}")
    print("=" * 72)

    return 0



if __name__ == "__main__":
    raise SystemExit(main())

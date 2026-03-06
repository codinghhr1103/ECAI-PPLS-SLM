"""Generate LaTeX tables from the latest `paper/artifacts/*` outputs.

This avoids manual copy/paste of numbers into the paper.

Usage:
  python scripts/generate_paper_tables.py

It will write generated .tex snippets under `paper/generated/tables/`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import pandas as pd


def _latex_escape(s: str) -> str:
    """Escape LaTeX special characters in text fields."""

    # Keep it minimal; most IDs are plain alphanumerics.
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }

    out = []
    for ch in s:
        out.append(repl.get(ch, ch))
    return "".join(out)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pm_makecell(table_str_x1e2: str) -> str:
    """Convert '7.43±1.23' into '\\makecell{7.43\\\\$\\pm$1.23}'."""

    # Some artifacts may use unicode ±.
    if "±" not in table_str_x1e2:
        return _latex_escape(table_str_x1e2)
    mean_s, std_s = table_str_x1e2.split("±", 1)
    mean_s = mean_s.strip()
    std_s = std_s.strip()
    return f"\\makecell{{{mean_s}\\\\$\\pm${std_s}}}"


def _format_float_1dp(x: Any) -> str:
    try:
        xf = float(x)
    except Exception:
        return str(x)
    return f"{xf:.1f}"


def _format_int(x: Any) -> str:
    try:
        return str(int(round(float(x))))
    except Exception:
        return str(x)


def generate_convergence_table(*, artifacts_dir: Path, out_path: Path) -> None:
    xlsx = artifacts_dir / "simulation" / "figures" / "Table_3_Convergence_Comparison.xlsx"
    if not xlsx.exists():
        raise FileNotFoundError(f"missing: {xlsx}")

    df = pd.read_excel(xlsx, sheet_name="Convergence_Statistics")

    required_cols = {
        "Algorithm",
        "Mean_Iterations",
        "Std_Iterations",
        "Min_Iterations",
        "Max_Iterations",
        "Median_Iterations",
        "Success_Rate",
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected convergence sheet columns; missing={sorted(missing)}")

    # Normalise algorithm names to match current Monte Carlo outputs.
    # (We now report SLM-Fixed / SLM-Joint / SLM-Oracle explicitly.)
    order = ["SLM-Fixed", "SLM-Joint", "SLM-Oracle", "EM", "ECM"]


    df = df.copy()
    df["Algorithm"] = df["Algorithm"].astype(str)

    rows = []
    for alg in order:
        sub = df[df["Algorithm"].str.upper() == alg.upper()]
        if sub.empty:
            raise ValueError(f"Algorithm '{alg}' not found in {xlsx}")
        r = sub.iloc[0]
        # Success_Rate is stored as a fraction in [0,1].
        try:
            succ = float(r["Success_Rate"])
            succ_s = f"{100.0 * succ:.0f}\\%"
        except Exception:
            succ_s = _latex_escape(str(r["Success_Rate"]))

        rows.append(
            (
                alg,
                _format_float_1dp(r["Mean_Iterations"]),
                _format_float_1dp(r["Std_Iterations"]),
                _format_int(r["Min_Iterations"]),
                _format_int(r["Max_Iterations"]),
                _format_int(r["Median_Iterations"]),
                succ_s,
            )
        )


    tex = []
    tex.append(r"\setlength{\tabcolsep}{3pt}")
    tex.append(r"\begin{table}[h]\footnotesize")
    tex.append(r"\centering")
    # Keep the caption consistent with the actual Monte Carlo trial count (M).
    m_trials = 100
    summary = artifacts_dir / "simulation" / "experiment_summary_low.json"
    if summary.exists():
        try:
            m_trials = int(_read_json(summary).get("experiment_info", {}).get("n_trials_completed", 100))
        except Exception:
            m_trials = 100

    tex.append(rf"\caption{{Convergence statistics across $M={m_trials}$ Monte Carlo trials}}")

    tex.append(r"\label{tab:algorithm_convergence}")
    tex.append(r"\begin{tabular}{lcccccc}")

    tex.append(r"\toprule")
    tex.append(r"\textbf{Method} & $\mathbb{E}[I]$ & $\sqrt{\text{Var}[I]}$ & $\min I$ & $\max I$ & $\text{Median}[I]$ & Success \\")

    tex.append(r"\midrule")
    for alg, mean, std, mn, mx, med, succ in rows:

        tex.append(f"{alg}  & {mean}  & {std}  & {mn}  & {mx}  & {med}  & {succ}  \\\\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    tex.append("")

    out_path.write_text("\n".join(tex), encoding="utf-8")


def generate_parameter_mse_table(*, artifacts_dir: Path, out_path: Path) -> None:
    low = _read_json(artifacts_dir / "simulation" / "mse_table_low.json")
    high = _read_json(artifacts_dir / "simulation" / "mse_table_high.json")

    methods = [
        ("slm", "SLM-Fixed"),
        ("slm_joint", "SLM-Joint"),
        ("slm_oracle", "SLM-Oracle"),
        ("em", "EM"),
        ("ecm", "ECM"),
    ]


    keys = [
        ("W", r"$\text{MSE}_W$"),
        ("C", r"$\text{MSE}_C$"),
        ("B", r"$\text{MSE}_B$"),
        ("Sigma_t", r"$\text{MSE}_{\Sigma_t}$"),
        ("sigma_h2", r"$\text{MSE}_{\sigma_h^2}$"),
    ]

    def row_for(data: Dict[str, Any], mkey: str) -> Tuple[str, ...]:
        out: list[str] = []
        for k, _hdr in keys:
            out.append(_pm_makecell(str(data[mkey][k]["table_str_x1e2"])))
        return tuple(out)

    tex = []
    tex.append(r"\setlength{\tabcolsep}{2.5pt}")
    tex.append(r"\begin{table}[h]\footnotesize")
    tex.append(r"\centering")
    tex.append(r"\caption{Parameter estimation accuracy under different noise levels: $\text{MSE} \times 10^2$ (mean $\pm$ standard deviation)}")
    tex.append(r"\label{tab:parameter_mse}")
    tex.append(r"\begin{tabular}{llccccc}")
    tex.append(r"\toprule")
    tex.append(r"\textbf{Noise} & \textbf{Method} & $\text{MSE}_W$ & $\text{MSE}_C$ & $\text{MSE}_B$ & $\text{MSE}_{\Sigma_t}$ & $\text{MSE}_{\sigma_h^2}$ \\")
    tex.append(r"\midrule")

    # Low noise block
    for i, (mkey, mname) in enumerate(methods):
        cells = row_for(low, mkey)
        noise = "Low" if i == 0 else ""
        lead = f"{noise}  & {mname}" if noise else f"     & {mname}"
        tex.append(lead + " & " + " & ".join(cells) + r" \\")
    tex.append(r"\midrule")

    # High noise block
    for i, (mkey, mname) in enumerate(methods):
        cells = row_for(high, mkey)
        noise = "High" if i == 0 else ""
        lead = f"{noise} & {mname}" if noise else f"     & {mname}"
        tex.append(lead + " & " + " & ".join(cells) + r" \\")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    tex.append("")

    out_path.write_text("\n".join(tex), encoding="utf-8")


def generate_top10_pairs_table(*, artifacts_dir: Path, out_path: Path) -> None:
    path = artifacts_dir / "association" / "top10_pairs_slm.csv"
    df = pd.read_csv(path)

    # Ensure fixed ordering and take top 10.
    df = df.head(10).copy()

    def f6(x: Any) -> str:
        try:
            return f"{float(x):.6f}"
        except Exception:
            return _latex_escape(str(x))

    tex = []
    tex.append(r"\setlength{\tabcolsep}{7pt}")
    tex.append(r"\begin{table}[t]\small")
    tex.append(r"\centering")
    tex.append(r"\caption{Top 10 gene-protein pairs identified by SLM among all latent variables sorted by the sum of absolute correlation coefficients}")
    tex.append(r"\label{tab:top10}")
    tex.append(r"\begin{tabular}{c| c c c c c}")
    tex.append(r"\toprule")
    tex.append(r"LV & Gene & $\rho$(G,\ LV) & Protein & $\rho$(P,\ LV) & $\sum \lvert \rho \rvert$ \\ ")
    tex.append(r"\midrule")

    for _, r in df.iterrows():
        lv = _latex_escape(str(r["LV"]))
        gene = _latex_escape(str(r["Gene"]))
        rg = f6(r["rho(G,LV)"])
        prot = _latex_escape(str(r["Protein"]))
        rp = f6(r["rho(P,LV)"])
        ssum = f6(r["sum|rho|"])
        tex.append(f"{lv} & {gene} & {rg} & {prot} & {rp} & {ssum} \\\\ ")
        tex.append(r"\midrule")

    # Replace last \midrule with \bottomrule
    if tex and tex[-1] == r"\midrule":
        tex[-1] = r"\bottomrule"
    else:
        tex.append(r"\bottomrule")

    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    tex.append("")

    out_path.write_text("\n".join(tex), encoding="utf-8")


def generate_detection_table(*, artifacts_dir: Path, out_path: Path) -> None:
    path = artifacts_dir / "association" / "detection_table.csv"
    df = pd.read_csv(path)

    # Expect thresholds in order.
    want = ["p < 1e-7", "p < 1e-6", "p < 1e-5", "p < 1e-4"]
    df["p-value threshold"] = df["p-value threshold"].astype(str)

    rows = []
    for th in want:
        sub = df[df["p-value threshold"].str.strip() == th]
        if sub.empty:
            raise ValueError(f"Missing threshold row: {th}")
        r = sub.iloc[0]
        rows.append((th, int(r["SLM"]), int(r["EM"]), int(r["Overlap"])))

    tex = []
    tex.append(r"\begin{table}[t]\small")
    tex.append(r"\centering")
    tex.append(r"\caption{Number of detected gene-protein pairs by SLM and EM under different p-values}")
    tex.append(r"\label{tab:Npairs}")
    tex.append(r"\begin{tabular}{l| c c c c}")
    tex.append(r"\toprule")
    tex.append(r"\textbf{Method} & p $< 1e^{-7}$ & p $< 1e^{-6}$ & p $< 1e^{-5}$ & p $< 1e^{-4}$ \\")
    tex.append(r"\midrule")

    # Pivot into the paper's layout.
    slm = [r[1] for r in rows]
    em = [r[2] for r in rows]
    ov = [r[3] for r in rows]

    tex.append("SLM & " + " & ".join(str(x) for x in slm) + r" \\")
    tex.append("EM & " + " & ".join(str(x) for x in em) + r" \\")
    tex.append(r"\midrule")
    tex.append("Overlap & " + " & ".join(str(x) for x in ov) + r" \\")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    tex.append("")

    out_path.write_text("\n".join(tex), encoding="utf-8")


def generate_prediction_coverage_table(*, artifacts_dir: Path, out_path: Path) -> None:
    # Backward-compatible: old location was `prediction/coverage_table.csv`.
    # New prediction pipeline stores artifacts under `prediction/synthetic/`.
    path_new = artifacts_dir / "prediction" / "synthetic" / "coverage_table.csv"
    path_old = artifacts_dir / "prediction" / "coverage_table.csv"
    path = path_new if path_new.exists() else path_old
    df = pd.read_csv(path, index_col=0)


    # Columns like 'Alpha=0.05'
    def parse_alpha(col: str) -> float:
        m = re.search(r"Alpha\s*=\s*([0-9.]+)", str(col))
        if not m:
            raise ValueError(f"Unrecognised alpha column: {col}")
        return float(m.group(1))

    alphas = [(parse_alpha(c), c) for c in df.columns]
    alphas.sort(key=lambda t: t[0])

    folds = [f"Fold {i}" for i in range(1, 6)]
    for f in folds:
        if f not in df.index:
            raise ValueError(f"Missing fold row in coverage_table: {f}")

    tex = []
    tex.append(r"\setlength{\tabcolsep}{2pt} ")
    tex.append(r"\renewcommand{\arraystretch}{1.2}")
    tex.append(r"\begin{table}[t]")
    tex.append(r"\centering")
    tex.append(r"\caption{Percentage of elements within prediction interval for different alpha values and folds. All the data are presented as \%.}")
    tex.append(r"\label{tab:prediction-accuracy}")
    tex.append(r"\begin{tabularx}{\linewidth}{p{0.2\linewidth}*{5}{X}}")
    tex.append(r"\hline")
    tex.append(r"Alpha & Fold 1 & Fold 2 & Fold 3 & Fold 4 & Fold 5 \\")
    tex.append(r"\hline")

    for a, col in alphas:
        # Keep 2dp; LaTeX table in paper uses 2dp.
        vals = [float(df.loc[f, col]) for f in folds]
        row = [f"{v:.2f}" for v in vals]
        tex.append(f"{a:.2f} & " + " & ".join(row) + r" \\")

    tex.append(r"\hline")
    tex.append(r"\end{tabularx}")
    tex.append(r"\end{table}")
    tex.append("")

    out_path.write_text("\n".join(tex), encoding="utf-8")


def _pm_makecell_float(mean: float, std: float, *, fmt: str = ".3f") -> str:
    m = format(float(mean), fmt)
    s = format(float(std), fmt)
    return f"\\makecell{{{m}\\\\$\\pm${s}}}"


def generate_prediction_synthetic_metrics_table(*, artifacts_dir: Path, out_path: Path) -> None:
    """Synthetic prediction accuracy table (MSE/MAE/R2) across 5 folds."""
    path = artifacts_dir / "prediction" / "synthetic" / "prediction_metrics_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing: {path}")

    df = pd.read_csv(path)

    order = ["PPLS-SLM", "PPLS-EM", "PLSR", "Ridge"]
    df["method"] = df["method"].astype(str)

    rows = []
    for m in order:
        sub = df[df["method"] == m]
        if sub.empty:
            continue
        r = sub.iloc[0]
        rows.append(
            (
                m,
                _pm_makecell_float(r["mse_mean"], r["mse_std"], fmt=".4g"),
                _pm_makecell_float(r["mae_mean"], r["mae_std"], fmt=".4g"),
                _pm_makecell_float(r["r2_mean"], r["r2_std"], fmt=".4g"),
            )
        )

    tex: list[str] = []
    tex.append(r"\setlength{\tabcolsep}{4pt}")
    tex.append(r"\begin{table}[t]\small")
    tex.append(r"\centering")
    tex.append(r"\caption{Synthetic prediction accuracy (5-fold CV): mean $\pm$ std.}")
    tex.append(r"\label{tab:pred_synth_metrics}")
    tex.append(r"\begin{tabular}{lccc}")
    tex.append(r"\toprule")
    tex.append(r"\textbf{Method} & \textbf{MSE} & \textbf{MAE} & \textbf{$R^2$} \\")
    tex.append(r"\midrule")
    for method, mse, mae, r2 in rows:
        tex.append(f"{method} & {mse} & {mae} & {r2} \\\\ ")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    tex.append("")

    out_path.write_text("\n".join(tex), encoding="utf-8")


def generate_prediction_synthetic_calibration_table(*, artifacts_dir: Path, out_path: Path) -> None:
    """Synthetic calibration comparison table (PPLS-SLM vs PPLS-EM)."""
    path = artifacts_dir / "prediction" / "synthetic" / "calibration_comparison.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing: {path}")

    df = pd.read_csv(path)

    def fmt_pct(x: float) -> str:
        return f"{float(x):.2f}\\%"

    tex: list[str] = []
    tex.append(r"\setlength{\tabcolsep}{4pt}")
    tex.append(r"\begin{table}[t]\small")
    tex.append(r"\centering")
    tex.append(r"\caption{Synthetic calibration of predictive credible intervals (5-fold CV).}")
    tex.append(r"\label{tab:pred_synth_calib}")
    tex.append(r"\begin{tabular}{c|c|cc}")
    tex.append(r"\toprule")
    tex.append(r"$\alpha$ & Expected & PPLS-SLM & PPLS-EM \\")
    tex.append(r"\midrule")

    for _, r in df.sort_values("alpha").iterrows():
        a = float(r["alpha"])
        expc = fmt_pct(r["expected_coverage"])
        slm = f"{fmt_pct(r['PPLS-SLM_mean'])} $\\pm$ {fmt_pct(r['PPLS-SLM_std'])}"
        em = f"{fmt_pct(r['PPLS-EM_mean'])} $\\pm$ {fmt_pct(r['PPLS-EM_std'])}"
        tex.append(f"{a:.2f} & {expc} & {slm} & {em} \\\\ ")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    tex.append("")

    out_path.write_text("\n".join(tex), encoding="utf-8")


def generate_prediction_brca_metrics_table(*, artifacts_dir: Path, out_path: Path) -> None:
    """BRCA prediction accuracy table at r* chosen by CV-MSE per method."""
    path = artifacts_dir / "prediction" / "brca" / "brca_prediction_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing: {path}")

    df = pd.read_csv(path)
    df["method"] = df["method"].astype(str)
    df["r"] = df["r"].astype(str)

    order = ["PPLS-SLM", "PPLS-EM", "PLSR", "Ridge"]

    tex: list[str] = []
    tex.append(r"\setlength{\tabcolsep}{4pt}")
    tex.append(r"\begin{table}[t]\small")
    tex.append(r"\centering")
    tex.append(r"\caption{BRCA prediction accuracy (5-fold CV). For PPLS/PLSR, $r^*$ minimises CV-MSE.}")
    tex.append(r"\label{tab:pred_brca_metrics}")
    tex.append(r"\begin{tabular}{lcccc}")
    tex.append(r"\toprule")
    tex.append(r"\textbf{Method} & $r$ & \textbf{MSE} & \textbf{MAE} & \textbf{$R^2$} \\")
    tex.append(r"\midrule")

    for m in order:
        sub = df[df["method"] == m]
        if sub.empty:
            continue
        r0 = sub.iloc[0]
        mse = _pm_makecell_float(r0["mse_mean"], r0["mse_std"], fmt=".4g")
        mae = _pm_makecell_float(r0["mae_mean"], r0["mae_std"], fmt=".4g")
        r2 = _pm_makecell_float(r0["r2_mean"], r0["r2_std"], fmt=".4g")
        tex.append(f"{m} & {str(r0['r'])} & {mse} & {mae} & {r2} \\")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    tex.append("")

    out_path.write_text("\n".join(tex), encoding="utf-8")


def generate_prediction_brca_calibration_table(*, artifacts_dir: Path, out_path: Path) -> None:
    """BRCA calibration table for PPLS-SLM at the selected r*."""
    path = artifacts_dir / "prediction" / "brca" / "brca_calibration_table.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing: {path}")

    df = pd.read_csv(path)

    # Expect columns: Alpha, Expected Coverage, Fold 1..Fold 5, Mean
    folds = [f"Fold {i}" for i in range(1, 6)]

    tex: list[str] = []
    tex.append(r"\setlength{\tabcolsep}{3pt}")
    tex.append(r"\begin{table}[t]\small")
    tex.append(r"\centering")
    tex.append(r"\caption{BRCA calibration of PPLS-SLM predictive credible intervals (element-wise coverage, \%).}")
    tex.append(r"\label{tab:pred_brca_calib}")
    tex.append(r"\begin{tabular}{c|c|ccccc|c}")
    tex.append(r"\toprule")
    tex.append(r"$\alpha$ & Expected & Fold 1 & Fold 2 & Fold 3 & Fold 4 & Fold 5 & Mean \\")
    tex.append(r"\midrule")

    for _, r in df.iterrows():
        a = float(r["Alpha"])
        expc = _latex_escape(str(r["Expected Coverage"]))
        vals = [_latex_escape(str(r[c])) for c in folds]
        mean = _latex_escape(str(r["Mean"]))
        tex.append(f"{a:.2f} & {expc} & " + " & ".join(vals) + f" & {mean} \\")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    tex.append("")

    out_path.write_text("\n".join(tex), encoding="utf-8")


def generate_noise_ablation_exp2_table(*, artifacts_dir: Path, out_path: Path) -> None:
    """Noise ablation Exp2 summary table.


    Source: `paper/artifacts/noise_ablation/exp2_joint_vs_fixed/exp2_summary_table.csv`
    """

    path = artifacts_dir / "noise_ablation" / "exp2_joint_vs_fixed" / "exp2_summary_table.csv"
    df = pd.read_csv(path)

    need = {"noise", "scheme", "success_rate", "runtime_sec_mean", "n_iterations_mean"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected exp2_summary_table columns; missing={sorted(missing)}")

    scheme_order = ["A_fixed", "B_joint"]

    # Normalise
    df["noise"] = df["noise"].astype(str)
    df["scheme"] = df["scheme"].astype(str)

    def noise_sort_key(n: str) -> tuple:
        # Prefer numeric ordering when noise labels encode sigma values, e.g. "e2_0p1".
        if n.startswith("e2_"):
            try:
                return (0, float(n[len("e2_"):].replace("p", ".")))
            except Exception:
                pass
        return (1, n)

    noise_values = df["noise"].unique().tolist()
    if set(noise_values) == {"low", "high"}:
        noise_order = ["low", "high"]
    else:
        noise_order = sorted(noise_values, key=noise_sort_key)

    def noise_name(n: str) -> str:
        if n.startswith("e2_"):
            try:
                v = float(n[len("e2_"):].replace("p", "."))
                return f"{v:g}"
            except Exception:
                pass
        return _latex_escape(n.capitalize())

    def scheme_name(s: str) -> str:
        # Keep compact labels for the paper.
        if s == "A_fixed":
            return "Fixed"
        if s == "B_joint":
            return "Joint"
        return _latex_escape(s)

    def fmt_pct(x: Any) -> str:
        try:
            xf = float(x)
        except Exception:
            return "--"
        if pd.isna(xf):
            return "--"
        return f"{100.0 * xf:.0f}\\%"

    def fmt_1dp(x: Any) -> str:
        try:
            xf = float(x)
        except Exception:
            return "--"
        if pd.isna(xf):
            return "--"
        return f"{xf:.1f}"

    rows = []
    for n in noise_order:
        for s in scheme_order:
            sub = df[(df["noise"] == n) & (df["scheme"] == s)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            rows.append(
                (
                    noise_name(n),
                    scheme_name(s),
                    fmt_pct(r["success_rate"]),
                    fmt_1dp(r["runtime_sec_mean"]),
                    fmt_1dp(r["n_iterations_mean"]),
                )
            )

    tex: list[str] = []
    tex.append(r"\begin{table}[t]\small")
    tex.append(r"\centering")
    tex.append(r"\caption{Noise ablation (Exp2): success rate and mean runtime for fixed vs.\ joint optimisation of $(\sigma_e^2,\sigma_f^2)$.}")
    tex.append(r"\label{tab:noise_ablation_exp2}")
    tex.append(r"\begin{tabular}{llccc}")
    tex.append(r"\toprule")
    tex.append(r"Noise & Scheme & Success & Runtime (sec) & Iterations \\")
    tex.append(r"\midrule")

    for noise, scheme, succ, rt, it in rows:
        tex.append(f"{noise} & {scheme} & {succ} & {rt} & {it} \\\\")


    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    tex.append("")

    out_path.write_text("\n".join(tex), encoding="utf-8")


def generate_noise_ablation_exp1_table(*, artifacts_dir: Path, out_path: Path) -> None:
    """Noise ablation Exp1: pre-estimation accuracy summary.

    Source: `paper/artifacts/noise_ablation/exp1_preestimate_accuracy/exp1_preestimate_accuracy.csv`
    """

    path = artifacts_dir / "noise_ablation" / "exp1_preestimate_accuracy" / "exp1_preestimate_accuracy.csv"
    df = pd.read_csv(path)

    need = {"snr", "N", "rel_error_mean", "rel_bound_mean"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected exp1_preestimate_accuracy columns; missing={sorted(missing)}")

    snr_order = ["lowSNR", "midSNR", "highSNR"]
    N_order = [200, 500, 2000, 5000]


    df = df.copy()
    df["snr"] = df["snr"].astype(str)
    df["N"] = df["N"].astype(int)

    def fmt_3dp(x: Any) -> str:
        try:
            xf = float(x)
        except Exception:
            return "--"
        if pd.isna(xf):
            return "--"
        return f"{xf:.3f}"

    def snr_name(s: str) -> str:
        if s == "lowSNR":
            return "Low"
        if s == "midSNR":
            return "Mid"
        if s == "highSNR":
            return "High"
        return _latex_escape(s)

    rows = []
    for snr in snr_order:
        for N in N_order:
            sub = df[(df["snr"] == snr) & (df["N"] == N)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            rows.append((snr_name(snr), str(N), fmt_3dp(r["rel_error_mean"]), fmt_3dp(r["rel_bound_mean"])))

    tex: list[str] = []
    tex.append(r"\begin{table}[t]\small")
    tex.append(r"\centering")
    tex.append(r"\caption{Noise pre-estimation accuracy (Exp1): empirical relative error vs. the theoretical relative bound across SNR regimes and sample sizes $N$.}")
    tex.append(r"\label{tab:noise_ablation_exp1}")
    tex.append(r"\begin{tabular}{llcc}")
    tex.append(r"\toprule")
    tex.append(r"SNR & $N$ & Empirical rel. error & Theorem rel. bound \\")
    tex.append(r"\midrule")

    for snr, N, err, bnd in rows:
        tex.append(f"{snr} & {N} & {err} & {bnd} \\\\")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    tex.append("")

    out_path.write_text("\n".join(tex), encoding="utf-8")




def generate_paper_metrics(*, artifacts_dir: Path, out_path: Path) -> None:
    """Generate LaTeX macros for key scalar results used in the prose.

    This keeps the narrative consistent with the latest `paper/artifacts/*`.

    Output: `paper/generated/metrics.tex`
    """

    # --- Convergence stats ---
    xlsx = artifacts_dir / "simulation" / "figures" / "Table_3_Convergence_Comparison.xlsx"
    df_conv = pd.read_excel(xlsx, sheet_name="Convergence_Statistics")

    def _get_row(alg: str) -> pd.Series:
        sub = df_conv[df_conv["Algorithm"].astype(str).str.upper() == alg.upper()]
        if sub.empty:
            raise ValueError(f"Algorithm '{alg}' not found in {xlsx}")
        return sub.iloc[0]

    # Keep macro names stable in LaTeX, but source values from the fixed-noise SLM row.
    r_slm = _get_row("SLM-Fixed")

    r_em = _get_row("EM")
    r_ecm = _get_row("ECM")

    mean_slm = float(r_slm["Mean_Iterations"])
    mean_em = float(r_em["Mean_Iterations"])
    mean_ecm = float(r_ecm["Mean_Iterations"])
    std_slm = float(r_slm["Std_Iterations"])
    std_em = float(r_em["Std_Iterations"])
    std_ecm = float(r_ecm["Std_Iterations"])

    def f1(x: float) -> str:
        return f"{x:.1f}"

    def f2(x: float) -> str:
        return f"{x:.2f}"

    def safe_div(a: float, b: float) -> float:
        return float(a / b) if b else float("nan")

    # --- Parameter MSE (x1e2 table values) ---
    low = _read_json(artifacts_dir / "simulation" / "mse_table_low.json")
    high = _read_json(artifacts_dir / "simulation" / "mse_table_high.json")

    def mean_x1e2(table_str: str) -> float:
        # e.g. "7.43±1.23" -> 7.43
        s = str(table_str)
        if "±" in s:
            s = s.split("±", 1)[0]
        return float(s.strip())

    def mse_mean(data: Dict[str, Any], method: str, key: str) -> float:
        return mean_x1e2(data[method][key]["table_str_x1e2"])

    methods = ["slm", "em", "ecm"]

    # Low-noise range used in prose for W/C together.
    low_wc = [mse_mean(low, m, k) for m in methods for k in ("W", "C")]
    low_wc_min, low_wc_max = min(low_wc), max(low_wc)

    # Overall ranges (across low/high + all methods) for W and C.
    all_w = [mse_mean(d, m, "W") for d in (low, high) for m in methods]
    all_c = [mse_mean(d, m, "C") for d in (low, high) for m in methods]

    # Specific low-noise scalar parameters.
    low_sigmat_emecm = [mse_mean(low, m, "Sigma_t") for m in ("em", "ecm")]
    low_sigmah_emecm = [mse_mean(low, m, "sigma_h2") for m in ("em", "ecm")]
    low_b_emecm = [mse_mean(low, m, "B") for m in ("em", "ecm")]

    # High-noise specific mentions.
    high_w_slm = mse_mean(high, "slm", "W")
    high_c_slm = mse_mean(high, "slm", "C")
    high_sigmah_emecm = [mse_mean(high, m, "sigma_h2") for m in ("em", "ecm")]
    high_sigmah_slm = mse_mean(high, "slm", "sigma_h2")

    # --- Association overlap counts ---
    det = pd.read_csv(artifacts_dir / "association" / "detection_table.csv")
    det["p-value threshold"] = det["p-value threshold"].astype(str).str.strip()

    def overlap_at(th: str) -> int:
        sub = det[det["p-value threshold"] == th]
        if sub.empty:
            raise ValueError(f"Missing threshold row: {th}")
        return int(sub.iloc[0]["Overlap"])

    # Write macros
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("% Auto-generated. Do not edit by hand.")
    lines.append("% Generated by scripts/generate_paper_tables.py")

    # Convergence
    lines.append(r"\providecommand{\ConvMeanSLM}{" + f1(mean_slm) + "}")
    lines.append(r"\providecommand{\ConvMeanEM}{" + f1(mean_em) + "}")
    lines.append(r"\providecommand{\ConvMeanECM}{" + f1(mean_ecm) + "}")
    lines.append(r"\providecommand{\ConvRatioEM}{" + f1(safe_div(mean_em, mean_slm)) + "}")
    lines.append(r"\providecommand{\ConvRatioECM}{" + f1(safe_div(mean_ecm, mean_slm)) + "}")
    lines.append(r"\providecommand{\ConvCVSLM}{" + f2(safe_div(std_slm, mean_slm)) + "}")
    lines.append(r"\providecommand{\ConvCVEM}{" + f2(safe_div(std_em, mean_em)) + "}")
    lines.append(r"\providecommand{\ConvCVECM}{" + f2(safe_div(std_ecm, mean_ecm)) + "}")

    # MSE (x1e2)
    lines.append(r"\providecommand{\MSELowWCMin}{" + f2(low_wc_min) + "}")
    lines.append(r"\providecommand{\MSELowWCMax}{" + f2(low_wc_max) + "}")

    lines.append(r"\providecommand{\MSEAllWMin}{" + f2(min(all_w)) + "}")
    lines.append(r"\providecommand{\MSEAllWMax}{" + f2(max(all_w)) + "}")
    lines.append(r"\providecommand{\MSEAllCMin}{" + f2(min(all_c)) + "}")
    lines.append(r"\providecommand{\MSEAllCMax}{" + f2(max(all_c)) + "}")

    lines.append(r"\providecommand{\MSESigmaTLowEMECMMin}{" + f1(min(low_sigmat_emecm)) + "}")
    lines.append(r"\providecommand{\MSESigmaTLowEMECMMax}{" + f1(max(low_sigmat_emecm)) + "}")
    lines.append(r"\providecommand{\MSESigmaHLowEMECM}{" + f2(min(low_sigmah_emecm)) + "}")
    lines.append(r"\providecommand{\MSESigmaTLowSLM}{" + f1(mse_mean(low, "slm", "Sigma_t")) + "}")
    lines.append(r"\providecommand{\MSESigmaHLowSLM}{" + f2(mse_mean(low, "slm", "sigma_h2")) + "}")
    lines.append(r"\providecommand{\MSEBLowSLM}{" + f2(mse_mean(low, "slm", "B")) + "}")
    lines.append(r"\providecommand{\MSEBLowEMECMMin}{" + f2(min(low_b_emecm)) + "}")
    lines.append(r"\providecommand{\MSEBLowEMECMMax}{" + f2(max(low_b_emecm)) + "}")

    lines.append(r"\providecommand{\MSEWHighSLM}{" + f2(high_w_slm) + "}")
    lines.append(r"\providecommand{\MSECHighSLM}{" + f2(high_c_slm) + "}")
    lines.append(r"\providecommand{\MSESigmaHHighEMECM}{" + f2(min(high_sigmah_emecm)) + "}")
    lines.append(r"\providecommand{\MSESigmaHHighSLM}{" + f2(high_sigmah_slm) + "}")

    # Association overlap
    lines.append(r"\providecommand{\OverlapPOneEminusSeven}{" + str(overlap_at("p < 1e-7")) + "}")
    lines.append(r"\providecommand{\OverlapPOneEminusSix}{" + str(overlap_at("p < 1e-6")) + "}")
    lines.append(r"\providecommand{\OverlapPOneEminusFour}{" + str(overlap_at("p < 1e-4")) + "}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    paper_dir = repo_root / "paper"
    artifacts_dir = paper_dir / "artifacts"
    out_dir = paper_dir / "generated" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    generate_convergence_table(artifacts_dir=artifacts_dir, out_path=out_dir / "tab_algorithm_convergence.tex")
    generate_parameter_mse_table(artifacts_dir=artifacts_dir, out_path=out_dir / "tab_parameter_mse.tex")
    generate_top10_pairs_table(artifacts_dir=artifacts_dir, out_path=out_dir / "tab_top10_pairs.tex")
    generate_detection_table(artifacts_dir=artifacts_dir, out_path=out_dir / "tab_detected_pairs.tex")

    # Prediction tables (synthetic + BRCA). These depend on synced artifacts.
    # Synthetic
    generate_prediction_synthetic_metrics_table(artifacts_dir=artifacts_dir, out_path=out_dir / "tab_prediction_synth_metrics.tex")
    generate_prediction_synthetic_calibration_table(artifacts_dir=artifacts_dir, out_path=out_dir / "tab_prediction_synth_calibration.tex")

    # BRCA (optional in config; skip cleanly when artifacts are absent)
    brca_dir = artifacts_dir / "prediction" / "brca"
    if (brca_dir / "brca_prediction_summary.csv").exists():
        generate_prediction_brca_metrics_table(artifacts_dir=artifacts_dir, out_path=out_dir / "tab_prediction_brca_metrics.tex")
    else:
        print(f"[SKIP] BRCA prediction table (missing: {brca_dir / 'brca_prediction_summary.csv'})")

    if (brca_dir / "brca_calibration_table.csv").exists():
        generate_prediction_brca_calibration_table(artifacts_dir=artifacts_dir, out_path=out_dir / "tab_prediction_brca_calibration.tex")
    else:
        print(f"[SKIP] BRCA calibration table (missing: {brca_dir / 'brca_calibration_table.csv'})")

    # Legacy coverage table (kept for backward compatibility).
    # (Some older paper drafts reference it.)
    try:
        generate_prediction_coverage_table(artifacts_dir=artifacts_dir, out_path=out_dir / "tab_prediction_coverage.tex")
    except FileNotFoundError as e:
        print(f"[SKIP] Legacy coverage table ({e})")

    generate_paper_metrics(artifacts_dir=artifacts_dir, out_path=paper_dir / "generated" / "metrics.tex")



    print(f"[OK] Wrote tables into: {out_dir}")




if __name__ == "__main__":
    main()


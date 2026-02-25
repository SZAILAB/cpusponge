#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS = [
    "Time",
    "Temperature",
    "Potential",
    "LJ",
    "PME",
    "Nb14_LJ",
    "Nb14_EE",
    "Bond",
    "Angle",
    "Dihedral",
]


def ensure_numeric(df: pd.DataFrame, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def plot_case_bar(summary_csv: Path, out_png: Path):
    df = pd.read_csv(summary_csv)
    df = ensure_numeric(df, ["mean_rel_err_pct", "max_rel_err_pct"])
    df = df.sort_values("mean_rel_err_pct")
    x = range(len(df))

    plt.figure(figsize=(12, 6))
    plt.bar(x, df["mean_rel_err_pct"], label="Mean RelErr (%)", alpha=0.85)
    plt.plot(x, df["max_rel_err_pct"], color="tab:red", marker="o", linewidth=2, label="Max RelErr (%)")
    plt.xticks(x, df["case"], rotation=20, ha="right")
    plt.ylabel("Relative Error (%)")
    plt.title("Step=1000 Baseline: local_cpu vs GPU by Case")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_case_metrics(relerr_csv: Path, out_png: Path):
    df = pd.read_csv(relerr_csv)
    df = ensure_numeric(df, ["Step"] + METRICS)

    plt.figure(figsize=(14, 7))
    for m in METRICS:
        if m in df.columns:
            plt.plot(df["Step"], df[m], linewidth=1.3, label=m)
    plt.title("Step=1000 Baseline: mdin_npt_andersenbaro metric-wise relative error")
    plt.xlabel("Step")
    plt.ylabel("Relative Error (%)")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_case_mean_line(summary_csv: Path, out_png: Path):
    df = pd.read_csv(summary_csv)
    rows = []
    for _, row in df.iterrows():
        case = row["case"]
        relerr_file = summary_csv.parent / "errors" / f"{case}.local_cpu.relerr.csv"
        one = pd.read_csv(relerr_file)
        one = ensure_numeric(one, ["Step"] + METRICS)
        one["case"] = case
        one["mean_rel_err"] = one[METRICS].mean(axis=1, skipna=True)
        rows.append(one[["Step", "case", "mean_rel_err"]])
    merged = pd.concat(rows, ignore_index=True)

    plt.figure(figsize=(14, 7))
    for case, grp in merged.groupby("case"):
        plt.plot(grp["Step"], grp["mean_rel_err"], linewidth=1.5, label=case)
    plt.title("Step=1000 Baseline: mean relative error vs step (local_cpu vs GPU)")
    plt.xlabel("Step")
    plt.ylabel("Mean Relative Error (%)")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", default="/home/wuping/cpusponge/reports/step1000_baseline")
    parser.add_argument("--fig-dir", default="/home/wuping/cpusponge/docs/figures")
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = baseline_dir / "local_cpu_relerr_summary.csv"
    andersen_csv = baseline_dir / "errors" / "mdin_npt_andersenbaro.local_cpu.relerr.csv"

    plot_case_bar(summary_csv, fig_dir / "step1000_case_summary.png")
    plot_case_metrics(andersen_csv, fig_dir / "step1000_andersenbaro_metrics.png")
    plot_case_mean_line(summary_csv, fig_dir / "step1000_case_mean_lines.png")

    print(f"figures generated in: {fig_dir}")


if __name__ == "__main__":
    main()

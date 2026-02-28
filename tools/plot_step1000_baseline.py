#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


def plot_case_bar(summary_csv: Path, out_png: Path, step_tag: str):
    df = pd.read_csv(summary_csv)
    df = ensure_numeric(df, ["mean_rel_err_pct", "max_rel_err_pct"])
    df = df.sort_values("mean_rel_err_pct")
    x = range(len(df))

    plt.figure(figsize=(12, 6))
    plt.bar(x, df["mean_rel_err_pct"], label="Mean RelErr (%)", alpha=0.85)
    plt.plot(x, df["max_rel_err_pct"], color="tab:red", marker="o", linewidth=2, label="Max RelErr (%)")
    plt.xticks(x, df["case"], rotation=20, ha="right")
    plt.ylabel("Relative Error (%)")
    plt.title(f"{step_tag} Baseline: local_cpu vs GPU by Case")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_case_metrics(relerr_csv: Path, out_png: Path, step_tag: str, focus_case: str):
    df = pd.read_csv(relerr_csv)
    df = ensure_numeric(df, ["Step"] + METRICS)

    plt.figure(figsize=(14, 7))
    for m in METRICS:
        if m in df.columns:
            plt.plot(df["Step"], df[m], linewidth=1.3, label=m)
    plt.title(f"{step_tag} Baseline: {focus_case} metric-wise relative error")
    plt.xlabel("Step")
    plt.ylabel("Relative Error (%)")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_case_mean_line(summary_csv: Path, out_png: Path, step_tag: str):
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
    plt.title(f"{step_tag} Baseline: mean relative error vs step (local_cpu vs GPU)")
    plt.xlabel("Step")
    plt.ylabel("Mean Relative Error (%)")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_case_heatmap(relerr_csv: Path, out_png: Path, step_tag: str, focus_case: str):
    df = pd.read_csv(relerr_csv)
    df = ensure_numeric(df, ["Step"] + METRICS)
    data = df[METRICS].to_numpy(dtype=float)
    if np.isnan(data).all():
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Relative Error (%)")

    ax.set_title(f"{step_tag} Baseline: {focus_case} relative error heatmap")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Step")
    ax.set_xticks(np.arange(len(METRICS)))
    ax.set_xticklabels(METRICS, rotation=30, ha="right")

    steps = df["Step"].fillna(0).astype(int).tolist()
    y_ticks = np.arange(0, len(steps), max(1, len(steps) // 12))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(steps[i]) for i in y_ticks])

    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", default="/home/wuping/cpusponge/reports/step1000_seedlock_hybrid_gpu")
    parser.add_argument("--fig-dir", default="/home/wuping/cpusponge/docs/figures")
    parser.add_argument("--step-tag", default="step1000")
    parser.add_argument("--focus-case", default="mdin_npt_andersenbaro")
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = baseline_dir / "local_cpu_relerr_summary.csv"
    focus_csv = baseline_dir / "errors" / f"{args.focus_case}.local_cpu.relerr.csv"

    plot_case_bar(summary_csv, fig_dir / f"{args.step_tag}_case_summary.png", args.step_tag)
    plot_case_metrics(focus_csv, fig_dir / f"{args.step_tag}_andersenbaro_metrics.png", args.step_tag, args.focus_case)
    plot_case_mean_line(summary_csv, fig_dir / f"{args.step_tag}_case_mean_lines.png", args.step_tag)
    plot_case_heatmap(focus_csv, fig_dir / f"{args.step_tag}_andersenbaro_heatmap_local_cpu.png", args.step_tag, args.focus_case)

    print(f"figures generated in: {fig_dir}")


if __name__ == "__main__":
    main()

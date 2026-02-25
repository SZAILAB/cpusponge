#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_COLS = [
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


def to_numeric(df: pd.DataFrame, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def read_relerr_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = to_numeric(df, ["Step"] + METRIC_COLS)
    return df


def plot_case_mean_lines(out_dir: Path, plot_dir: Path, case_name: str, backends):
    plt.figure(figsize=(12, 6))
    drew = False
    for backend in backends:
        rel_path = out_dir / "errors" / f"{case_name}.{backend}.relerr.csv"
        if not rel_path.exists():
            continue
        df = read_relerr_csv(rel_path)
        if df.empty:
            continue
        y = df[METRIC_COLS].mean(axis=1, skipna=True)
        if y.notna().sum() == 0:
            continue
        plt.plot(df["Step"], y, label=backend, linewidth=1.6)
        drew = True
    if not drew:
        plt.close()
        return
    plt.title(f"Mean Relative Error vs Step ({case_name})")
    plt.xlabel("Step")
    plt.ylabel("Mean Relative Error (%)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / f"relerr_mean_{case_name}.png", dpi=180)
    plt.close()


def plot_case_backend_metrics(out_dir: Path, plot_dir: Path, case_name: str, backend: str):
    rel_path = out_dir / "errors" / f"{case_name}.{backend}.relerr.csv"
    if not rel_path.exists():
        return
    df = read_relerr_csv(rel_path)
    if df.empty:
        return

    plt.figure(figsize=(14, 7))
    drew = False
    for col in METRIC_COLS:
        if col not in df.columns:
            continue
        y = df[col]
        if y.notna().sum() == 0:
            continue
        plt.plot(df["Step"], y, label=col, linewidth=1.25)
        drew = True
    if not drew:
        plt.close()
        return
    plt.title(f"Relative Error by Metric ({case_name}, {backend})")
    plt.xlabel("Step")
    plt.ylabel("Relative Error (%)")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(plot_dir / f"relerr_metrics_{case_name}.{backend}.png", dpi=180)
    plt.close()


def plot_heatmap(out_dir: Path, plot_dir: Path, case_name: str, backend: str):
    rel_path = out_dir / "errors" / f"{case_name}.{backend}.relerr.csv"
    if not rel_path.exists():
        return
    df = read_relerr_csv(rel_path)
    if df.empty:
        return

    data = df[METRIC_COLS].to_numpy(dtype=float)
    if np.isnan(data).all():
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Relative Error (%)")

    ax.set_title(f"Relative Error Heatmap ({case_name}, {backend})")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Step")
    ax.set_xticks(np.arange(len(METRIC_COLS)))
    ax.set_xticklabels(METRIC_COLS, rotation=30, ha="right")

    steps = df["Step"].fillna(0).astype(int).tolist()
    y_ticks = np.arange(0, len(steps), max(1, len(steps) // 12))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(steps[i]) for i in y_ticks])

    plt.tight_layout()
    plt.savefig(plot_dir / f"relerr_heatmap_{case_name}.{backend}.png", dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/home/wuping/cpusponge/reports/step5000")
    parser.add_argument("--focus-case", default="mdin_npt_andersenbaro")
    parser.add_argument("--focus-backend", default="local_cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    relerr_summary = out_dir / "relerr_summary.csv"
    if not relerr_summary.exists():
        raise FileNotFoundError(f"missing {relerr_summary}")

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(relerr_summary)
    cases = sorted(summary_df["case"].dropna().unique().tolist())
    backends = sorted(summary_df["backend"].dropna().unique().tolist())

    for case_name in cases:
        plot_case_mean_lines(out_dir, plot_dir, case_name, backends)

    focus_case = args.focus_case
    for backend in backends:
        plot_case_backend_metrics(out_dir, plot_dir, focus_case, backend)

    plot_heatmap(out_dir, plot_dir, focus_case, args.focus_backend)

    print(f"plots generated under: {plot_dir}")


if __name__ == "__main__":
    main()

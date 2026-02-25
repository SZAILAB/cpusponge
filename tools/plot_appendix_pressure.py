#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_mdout(mdout_path: Path) -> pd.DataFrame:
    header = None
    rows = []
    with mdout_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = re.split(r"\s+", line)
            if header is None:
                header = [p.lower() for p in parts]
                continue
            if not parts or not re.fullmatch(r"[0-9]+", parts[0]):
                continue
            if len(parts) < len(header):
                continue
            vals = parts[: len(header)]
            rec = {}
            for k, v in zip(header, vals):
                try:
                    rec[k] = float(v)
                except ValueError:
                    rec[k] = float("nan")
            rows.append(rec)
    if not rows:
        return pd.DataFrame(columns=["step", "pressure"])
    df = pd.DataFrame(rows)
    if "step" not in df.columns:
        raise ValueError(f"'step' column missing in {mdout_path}")
    if "pressure" not in df.columns:
        raise ValueError(f"'pressure' column missing in {mdout_path}")
    df = df.sort_values("step").reset_index(drop=True)
    return df


def save_gpu_csvs(gpu_dfs, out_dir: Path):
    rows = []
    for run_id, df in gpu_dfs.items():
        one = df[["step", "pressure"]].copy()
        one["run_id"] = run_id
        one["source"] = "gpu"
        one.to_csv(out_dir / f"{run_id}.pressure.csv", index=False)
        rows.append(one)
    combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["step", "pressure", "run_id", "source"])
    combined.to_csv(out_dir / "gpu_pressure_combined.csv", index=False)
    return combined


def plot_gpu_repeats(gpu_combined: pd.DataFrame, fig_path: Path):
    plt.figure(figsize=(14, 8))
    for run_id, grp in gpu_combined.groupby("run_id"):
        plt.plot(grp["step"], grp["pressure"], linewidth=1.5, label=run_id)
    plt.title("Pressure vs Step for each repeated NPT Andersen barostat run")
    plt.xlabel("Step")
    plt.ylabel("Pressure")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()


def plot_gpu_gray_vs_cpu(gpu_combined: pd.DataFrame, cpu_df: pd.DataFrame, fig_path: Path):
    plt.figure(figsize=(14, 8))
    for _, grp in gpu_combined.groupby("run_id"):
        plt.plot(grp["step"], grp["pressure"], color="gray", alpha=0.35, linewidth=1.5)
    plt.plot(cpu_df["step"], cpu_df["pressure"], color="#d62728", linewidth=3, label="CPU reference")
    plt.title("Pressure vs Step: GPU repeated runs (gray) vs CPU reference (highlighted)")
    plt.xlabel("Step")
    plt.ylabel("Pressure")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()


def plot_cpu_ma(cpu_df: pd.DataFrame, fig_path: Path, ma_window: int):
    one = cpu_df.copy()
    one["pressure_ma"] = one["pressure"].rolling(ma_window, min_periods=1).mean()
    plt.figure(figsize=(12, 6))
    plt.plot(one["step"], one["pressure"], linewidth=1.3, alpha=0.35, label="pressure")
    plt.plot(one["step"], one["pressure_ma"], linewidth=2.0, label=f"pressure MA({ma_window})")
    plt.title(f"SPONGE_CPU / pressure moving average ({ma_window})")
    plt.xlabel("Step")
    plt.ylabel("Pressure")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()


def plot_overlay_with_gpu_mean(gpu_combined: pd.DataFrame, cpu_df: pd.DataFrame, fig_path: Path):
    gpu_mean = gpu_combined.groupby("step", as_index=False)["pressure"].mean().rename(columns={"pressure": "gpu_mean_pressure"})

    plt.figure(figsize=(16, 9))
    for _, grp in gpu_combined.groupby("run_id"):
        plt.step(grp["step"], grp["pressure"], where="mid", linewidth=0.8, alpha=0.25, color="#4ea3c8")
    plt.step(gpu_mean["step"], gpu_mean["gpu_mean_pressure"], where="mid", color="black", linewidth=2, linestyle="--", label="GPU mean")
    plt.step(cpu_df["step"], cpu_df["pressure"], where="mid", color="#cf2f2f", linewidth=2.2, label="CPU reference")
    plt.title("Original covid-tip4p: 10 GPU + 1 CPU pressure trajectories (5000-step run)")
    plt.xlabel("Step")
    plt.ylabel("Pressure")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()


def build_summary(gpu_combined: pd.DataFrame, cpu_df: pd.DataFrame, out_csv: Path):
    rows = []
    for run_id, grp in gpu_combined.groupby("run_id"):
        rows.append(
            {
                "source": "gpu",
                "run_id": run_id,
                "points": len(grp),
                "pressure_mean": grp["pressure"].mean(),
                "pressure_std": grp["pressure"].std(ddof=1),
                "pressure_min": grp["pressure"].min(),
                "pressure_max": grp["pressure"].max(),
            }
        )
    rows.append(
        {
            "source": "cpu",
            "run_id": "cpu_reference",
            "points": len(cpu_df),
            "pressure_mean": cpu_df["pressure"].mean(),
            "pressure_std": cpu_df["pressure"].std(ddof=1),
            "pressure_min": cpu_df["pressure"].min(),
            "pressure_max": cpu_df["pressure"].max(),
        }
    )
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--appendix-dir", default="/home/wuping/cpusponge/reports/step5000_appendix")
    parser.add_argument("--matrix-dir", default="/home/wuping/cpusponge/reports/step5000")
    parser.add_argument("--case-name", default="mdin_npt_andersenbaro")
    parser.add_argument("--cpu-backend", default="local_cpu")
    parser.add_argument("--ma-window", type=int, default=10)
    args = parser.parse_args()

    appendix_dir = Path(args.appendix_dir)
    matrix_dir = Path(args.matrix_dir)

    figure_dir = appendix_dir / "figures"
    pressure_dir = appendix_dir / "pressure"
    figure_dir.mkdir(parents=True, exist_ok=True)
    pressure_dir.mkdir(parents=True, exist_ok=True)

    gpu_mdouts = sorted((appendix_dir / "mdout").glob(f"{args.case_name}.gpu_ref.run_*.mdout.txt"))
    if not gpu_mdouts:
        raise FileNotFoundError(f"no GPU mdout files under {(appendix_dir / 'mdout')}")

    gpu_dfs = {}
    for path in gpu_mdouts:
        m = re.search(r"\.gpu_ref\.(run_[0-9]{3})\.mdout\.txt$", path.name)
        if not m:
            continue
        run_id = m.group(1)
        gpu_dfs[run_id] = parse_mdout(path)
    if not gpu_dfs:
        raise RuntimeError("failed to parse any GPU run")

    gpu_combined = save_gpu_csvs(gpu_dfs, pressure_dir)

    cpu_mdout = matrix_dir / "mdout" / f"{args.case_name}.{args.cpu_backend}.run1.mdout.txt"
    if not cpu_mdout.exists():
        raise FileNotFoundError(f"missing CPU reference mdout: {cpu_mdout}")
    cpu_df = parse_mdout(cpu_mdout)
    cpu_df = cpu_df[["step", "pressure"]].copy()
    cpu_df["run_id"] = "cpu_reference"
    cpu_df["source"] = "cpu"
    cpu_df.to_csv(pressure_dir / "cpu_reference.pressure.csv", index=False)

    plot_gpu_repeats(gpu_combined, figure_dir / "pressure_vs_step_gpu_repeats.png")
    plot_gpu_gray_vs_cpu(gpu_combined, cpu_df, figure_dir / "pressure_vs_step_gpu_gray_vs_cpu.png")
    plot_cpu_ma(cpu_df, figure_dir / "pressure_cpu_ma10.png", args.ma_window)
    plot_overlay_with_gpu_mean(gpu_combined, cpu_df, figure_dir / "pressure_overlay_gpu10_cpu1.png")

    combined_all = pd.concat(
        [
            gpu_combined[["step", "pressure", "run_id", "source"]],
            cpu_df[["step", "pressure", "run_id", "source"]],
        ],
        ignore_index=True,
    )
    combined_all.to_csv(pressure_dir / "pressure_all_runs.csv", index=False)
    build_summary(gpu_combined, cpu_df, pressure_dir / "pressure_summary_stats.csv")

    print(f"appendix pressure records: {pressure_dir}")
    print(f"appendix figures: {figure_dir}")


if __name__ == "__main__":
    main()

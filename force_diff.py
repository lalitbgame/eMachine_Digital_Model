"""
Electromagnetic Force Difference Analysis
==========================================
Compares radial and tangential stator tooth forces between:
  - Non-uniform airgap (eccentric rotor)
  - Uniform airgap (wide/nominal airgap)

Operating condition: 150 Nm, 48-slot machine

Outputs
-------
- Console summary statistics per tooth
- CSV: absolute and relative differences for every tooth & angle step
- CSV: per-tooth aggregated stats (mean, peak, RMS of difference)
- Plots: spatial maps, per-tooth bar charts, time waveforms, FFT comparison
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

ECCENTRIC_FILE = "/mnt/user-data/uploads/emag_forces_eccentric_150Nm_48slot.csv"
UNIFORM_FILE   = "/mnt/user-data/uploads/emag_forces_wide_150Nm_48slot.csv"
OUTPUT_DIR     = Path("/mnt/user-data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_TEETH  = 48
TORQUE   = 150   # Nm (for labelling)
ANGLE_COL = "Electrical_Angle_deg"

# Teeth to highlight in waveform / FFT plots (1-based)
HIGHLIGHT_TEETH = [1, 12, 24, 36, 48]


# ── Helper functions ─────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    assert ANGLE_COL in df.columns, f"Missing column: {ANGLE_COL}"
    return df.set_index(ANGLE_COL)


def get_force_columns(df: pd.DataFrame):
    """Return sorted lists of radial and tangential column names."""
    radial = sorted(
        [c for c in df.columns if c.startswith("Radial_Force_")],
        key=lambda c: int(c.split("_")[-1])
    )
    tangential = sorted(
        [c for c in df.columns if c.startswith("Tangential_Force_")],
        key=lambda c: int(c.split("_")[-1])
    )
    return radial, tangential


def rms(series: pd.Series) -> float:
    return float(np.sqrt(np.mean(series**2)))


def compute_differences(ecc: pd.DataFrame, uni: pd.DataFrame,
                        radial_cols, tangential_cols):
    """
    Returns
    -------
    diff_abs  : absolute difference  (eccentric – uniform)
    diff_rel  : relative difference  (%) w.r.t. uniform
    """
    diff_abs = pd.DataFrame(index=ecc.index)
    diff_rel = pd.DataFrame(index=ecc.index)

    for col in radial_cols + tangential_cols:
        diff_abs[col] = ecc[col] - uni[col]
        # avoid division by zero
        diff_rel[col] = np.where(
            uni[col].abs() > 1e-9,
            100.0 * diff_abs[col] / uni[col].abs(),
            np.nan
        )

    return diff_abs, diff_rel


def per_tooth_stats(diff_abs: pd.DataFrame,
                    radial_cols, tangential_cols) -> pd.DataFrame:
    """Aggregate statistics per tooth (across all electrical angle steps)."""
    records = []
    for tooth_id in range(1, N_TEETH + 1):
        rc = f"Radial_Force_{tooth_id}"
        tc = f"Tangential_Force_{tooth_id}"
        row = {"Tooth": tooth_id}

        for label, col in [("Radial", rc), ("Tangential", tc)]:
            s = diff_abs[col]
            row[f"{label}_Mean_N"]   = float(s.mean())
            row[f"{label}_Std_N"]    = float(s.std())
            row[f"{label}_Peak_N"]   = float(s.abs().max())
            row[f"{label}_RMS_N"]    = rms(s)
            row[f"{label}_PeakAngle"]= int(s.abs().idxmax())

        records.append(row)

    return pd.DataFrame(records).set_index("Tooth")


# ── Plotting functions ────────────────────────────────────────────────────────

def plot_spatial_map(stats: pd.DataFrame, output_dir: Path):
    """Polar bar chart showing peak radial & tangential diff per tooth."""
    teeth  = np.arange(1, N_TEETH + 1)
    angles = 2 * np.pi * (teeth - 1) / N_TEETH

    fig, axes = plt.subplots(1, 2,
                             subplot_kw={"projection": "polar"},
                             figsize=(14, 6))
    fig.suptitle(f"Peak |ΔForce| per Stator Tooth — {TORQUE} Nm\n"
                 "Eccentric vs Uniform Airgap", fontsize=13)

    for ax, col, label, color in [
        (axes[0], "Radial_Peak_N",     "Radial ΔForce (N)",     "steelblue"),
        (axes[1], "Tangential_Peak_N", "Tangential ΔForce (N)", "tomato"),
    ]:
        values = stats[col].values
        width  = 2 * np.pi / N_TEETH * 0.85
        bars   = ax.bar(angles, values, width=width, color=color,
                        alpha=0.8, edgecolor="white", linewidth=0.4)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[::6])
        ax.set_xticklabels([f"T{t}" for t in teeth[::6]], fontsize=8)
        ax.set_title(label, pad=14, fontsize=11)
        # annotate max
        idx_max = np.argmax(values)
        ax.annotate(f"T{teeth[idx_max]}\n{values[idx_max]:.1f} N",
                    xy=(angles[idx_max], values[idx_max]),
                    xytext=(angles[idx_max], values[idx_max] * 1.15),
                    fontsize=7, ha="center", color="black",
                    arrowprops=dict(arrowstyle="->", lw=0.8))

    plt.tight_layout()
    out = output_dir / "01_spatial_peak_diff_map.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_mean_bar(stats: pd.DataFrame, output_dir: Path):
    """Bar chart of mean absolute difference across all teeth."""
    teeth = np.arange(1, N_TEETH + 1)
    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    fig.suptitle(f"Mean |ΔForce| per Tooth — {TORQUE} Nm\n"
                 "Eccentric – Uniform Airgap", fontsize=13)

    for ax, col, label, color in [
        (axes[0], "Radial_Mean_N",     "Mean ΔRadial Force (N)",     "steelblue"),
        (axes[1], "Tangential_Mean_N", "Mean ΔTangential Force (N)", "tomato"),
    ]:
        vals = stats[col].values
        bars = ax.bar(teeth, vals, color=color, alpha=0.75, width=0.7)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(vals.mean(), color="darkgray", linewidth=1,
                   linestyle="--", label=f"Overall mean={vals.mean():.1f} N")
        ax.set_ylabel(label, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        # highlight highlighted teeth
        for ht in HIGHLIGHT_TEETH:
            ax.get_children()[ht - 1].set_edgecolor("black")
            ax.get_children()[ht - 1].set_linewidth(1.5)

    axes[1].set_xlabel("Stator Tooth Number", fontsize=10)
    axes[1].set_xticks(teeth[::3])
    plt.tight_layout()
    out = output_dir / "02_mean_diff_bar.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_waveforms(diff_abs: pd.DataFrame, output_dir: Path):
    """Time-domain waveforms of ΔForce for selected teeth."""
    angle = diff_abs.index.values
    fig, axes = plt.subplots(len(HIGHLIGHT_TEETH), 2,
                             figsize=(16, 3.5 * len(HIGHLIGHT_TEETH)),
                             sharex=True)
    fig.suptitle(f"ΔForce Waveforms — Selected Teeth — {TORQUE} Nm",
                 fontsize=13)

    for row, tooth in enumerate(HIGHLIGHT_TEETH):
        rc = f"Radial_Force_{tooth}"
        tc = f"Tangential_Force_{tooth}"

        for col_ax, col, label, color in [
            (axes[row, 0], rc, f"T{tooth}: ΔRadial (N)",     "steelblue"),
            (axes[row, 1], tc, f"T{tooth}: ΔTangential (N)", "tomato"),
        ]:
            col_ax.plot(angle, diff_abs[col], color=color, linewidth=1.2)
            col_ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
            col_ax.fill_between(angle, diff_abs[col], alpha=0.15, color=color)
            col_ax.set_ylabel(label, fontsize=9)
            col_ax.grid(linestyle=":", alpha=0.5)
            peak = diff_abs[col].abs().max()
            col_ax.set_title(f"Peak = {peak:.1f} N", fontsize=9)

    for ax in axes[-1]:
        ax.set_xlabel("Electrical Angle (°)", fontsize=10)
    plt.tight_layout()
    out = output_dir / "03_waveforms_selected_teeth.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_fft_comparison(ecc: pd.DataFrame, uni: pd.DataFrame,
                        output_dir: Path):
    """
    FFT of radial force for selected teeth — overlay eccentric vs uniform
    to show harmonic content shift.
    """
    fig, axes = plt.subplots(len(HIGHLIGHT_TEETH), 1,
                             figsize=(14, 3.5 * len(HIGHLIGHT_TEETH)))
    fig.suptitle(f"Radial Force FFT — Eccentric vs Uniform — {TORQUE} Nm",
                 fontsize=13)

    for ax, tooth in zip(axes, HIGHLIGHT_TEETH):
        rc = f"Radial_Force_{tooth}"
        n  = len(ecc)
        freqs = np.fft.rfftfreq(n, d=1.0)  # cycles per electrical degree

        for df_case, label, color, ls in [
            (ecc, "Eccentric (non-uniform)", "tomato",    "-"),
            (uni, "Uniform",                 "steelblue", "--"),
        ]:
            sig = df_case[rc].values - df_case[rc].values.mean()
            mag = np.abs(np.fft.rfft(sig)) * 2 / n
            ax.plot(freqs, mag, color=color, linewidth=1.2,
                    linestyle=ls, label=label)

        ax.set_title(f"Tooth {tooth}", fontsize=10)
        ax.set_ylabel("Amplitude (N)", fontsize=9)
        ax.set_xlim(0, freqs[-1] / 2)   # show first half of spectrum
        ax.legend(fontsize=8)
        ax.grid(linestyle=":", alpha=0.5)

    axes[-1].set_xlabel("Frequency (cycles / electrical °)", fontsize=10)
    plt.tight_layout()
    out = output_dir / "04_fft_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_rms_map(stats: pd.DataFrame, output_dir: Path):
    """Heatmap-style comparison: RMS diff radial vs tangential, sorted."""
    fig, ax = plt.subplots(figsize=(14, 5))
    teeth = np.arange(1, N_TEETH + 1)
    x     = np.arange(N_TEETH)
    width = 0.4

    ax.bar(x - width / 2, stats["Radial_RMS_N"],
           width=width, color="steelblue", alpha=0.8, label="Radial RMS ΔF (N)")
    ax.bar(x + width / 2, stats["Tangential_RMS_N"],
           width=width, color="tomato",    alpha=0.8, label="Tangential RMS ΔF (N)")

    ax.set_xticks(x[::3])
    ax.set_xticklabels([f"T{t}" for t in teeth[::3]], fontsize=8)
    ax.set_xlabel("Stator Tooth Number", fontsize=10)
    ax.set_ylabel("RMS |ΔForce| (N)", fontsize=10)
    ax.set_title(f"RMS Force Difference per Tooth — {TORQUE} Nm\n"
                 "Eccentric vs Uniform Airgap", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    out = output_dir / "05_rms_diff_per_tooth.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" Electromagnetic Force Difference Analysis")
    print(f" Operating condition: {TORQUE} Nm | 48-slot machine")
    print("=" * 60)

    # 1. Load
    print("\n[1] Loading data...")
    ecc = load_data(ECCENTRIC_FILE)
    uni = load_data(UNIFORM_FILE)
    radial_cols, tangential_cols = get_force_columns(ecc)
    print(f"    Electrical angle steps : {len(ecc)}")
    print(f"    Stator teeth           : {N_TEETH}")

    # 2. Compute differences
    print("\n[2] Computing differences (Eccentric – Uniform)...")
    diff_abs, diff_rel = compute_differences(ecc, uni, radial_cols, tangential_cols)

    # 3. Per-tooth stats
    print("\n[3] Aggregating per-tooth statistics...")
    stats = per_tooth_stats(diff_abs, radial_cols, tangential_cols)

    # ── Console summary ──────────────────────────────────────────────────
    print("\n── Top 5 Teeth by Peak Radial ΔForce ──")
    top_rad = stats["Radial_Peak_N"].nlargest(5)
    for tooth, val in top_rad.items():
        ang = stats.loc[tooth, "Radial_PeakAngle"]
        print(f"   Tooth {tooth:>2d}: {val:>8.2f} N  @ {ang}°")

    print("\n── Top 5 Teeth by Peak Tangential ΔForce ──")
    top_tan = stats["Tangential_Peak_N"].nlargest(5)
    for tooth, val in top_tan.items():
        ang = stats.loc[tooth, "Tangential_PeakAngle"]
        print(f"   Tooth {tooth:>2d}: {val:>8.2f} N  @ {ang}°")

    print(f"\n── Overall (all teeth, all angles) ──")
    rad_series  = diff_abs[radial_cols].values.ravel()
    tan_series  = diff_abs[tangential_cols].values.ravel()
    print(f"   Radial    — Mean: {rad_series.mean():+.2f} N | "
          f"RMS: {np.sqrt(np.mean(rad_series**2)):.2f} N | "
          f"Peak: {np.abs(rad_series).max():.2f} N")
    print(f"   Tangential— Mean: {tan_series.mean():+.2f} N | "
          f"RMS: {np.sqrt(np.mean(tan_series**2)):.2f} N | "
          f"Peak: {np.abs(tan_series).max():.2f} N")

    # 4. Save CSVs
    print("\n[4] Saving CSV outputs...")

    # Full difference table (absolute)
    diff_abs_out = OUTPUT_DIR / "force_diff_absolute_N.csv"
    diff_abs.to_csv(diff_abs_out)
    print(f"    Saved: {diff_abs_out.name}")

    # Full difference table (relative %)
    diff_rel_out = OUTPUT_DIR / "force_diff_relative_pct.csv"
    diff_rel.to_csv(diff_rel_out)
    print(f"    Saved: {diff_rel_out.name}")

    # Per-tooth aggregated stats
    stats_out = OUTPUT_DIR / "per_tooth_stats.csv"
    stats.to_csv(stats_out)
    print(f"    Saved: {stats_out.name}")

    # 5. Plots
    print("\n[5] Generating plots...")
    plot_spatial_map(stats, OUTPUT_DIR)
    plot_mean_bar(stats, OUTPUT_DIR)
    plot_waveforms(diff_abs, OUTPUT_DIR)
    plot_fft_comparison(ecc, uni, OUTPUT_DIR)
    plot_rms_map(stats, OUTPUT_DIR)

    print("\n✓ Analysis complete. All outputs written to:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()

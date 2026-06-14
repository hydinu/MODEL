# =============================================================================
# analyse.py — Crowd Analytics: trend graph · average count · peak crowd hour
# =============================================================================
"""
Usage
-----
    python analyse.py                            # reads logs/crowd_log.csv
    python analyse.py --csv path/to/custom.csv  # custom CSV path
    python analyse.py --no-show                 # save PNG only, don't open window

Output
------
    Console : summary statistics (average, peak hour, max count, …)
    PNG file: logs/crowd_analysis.png  (multi-panel dark-themed report)
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ─── Styling constants ────────────────────────────────────────────────────────

DARK_BG      = "#0d1117"
PANEL_BG     = "#161b22"
GRID_COLOR   = "#30363d"
ACCENT_TEAL  = "#39d0d8"
ACCENT_AMBER = "#f0a500"
ACCENT_CORAL = "#ff6b6b"
ACCENT_GREEN = "#3fb950"
TEXT_PRIMARY = "#e6edf3"
TEXT_MUTED   = "#8b949e"

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    GRID_COLOR,
    "axes.labelcolor":   TEXT_PRIMARY,
    "axes.titlecolor":   TEXT_PRIMARY,
    "xtick.color":       TEXT_MUTED,
    "ytick.color":       TEXT_MUTED,
    "grid.color":        GRID_COLOR,
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "text.color":        TEXT_PRIMARY,
    "font.family":       "sans-serif",
    "font.size":         10,
    "lines.linewidth":   2,
})


# ─── Data loading & validation ────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    """Load and validate the crowd-log CSV.  Returns a clean DataFrame."""
    p = Path(path)
    if not p.exists():
        sys.exit(
            f"[Error] CSV file not found: {p.resolve()}\n"
            "Run main.py first to generate log data."
        )

    df = pd.read_csv(p)

    required = {"timestamp", "crowd_count"}
    missing  = required - set(df.columns)
    if missing:
        sys.exit(f"[Error] CSV is missing column(s): {missing}")

    if df.empty:
        sys.exit("[Error] CSV file contains no data rows.")

    # Parse timestamps robustly
    df["timestamp"] = pd.to_datetime(df["timestamp"], infer_datetime_format=True)
    df["crowd_count"] = pd.to_numeric(df["crowd_count"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Derived time columns
    df["hour"]  = df["timestamp"].dt.hour
    df["date"]  = df["timestamp"].dt.date
    df["minute"] = df["timestamp"].dt.floor("min")

    return df


# ─── Statistics ───────────────────────────────────────────────────────────────

def compute_stats(df: pd.DataFrame) -> dict:
    """Return a dictionary of key statistics."""
    hourly       = df.groupby("hour")["crowd_count"].mean()
    peak_hour    = int(hourly.idxmax())
    peak_avg     = float(hourly.max())

    return {
        "total_records"  : len(df),
        "avg_crowd"      : round(df["crowd_count"].mean(), 2),
        "max_crowd"      : int(df["crowd_count"].max()),
        "min_crowd"      : int(df["crowd_count"].min()),
        "std_dev"        : round(df["crowd_count"].std(), 2),
        "peak_hour"      : peak_hour,
        "peak_hour_avg"  : round(peak_avg, 2),
        "peak_hour_label": f"{peak_hour:02d}:00 – {peak_hour:02d}:59",
        "start_time"     : df["timestamp"].min().strftime("%Y-%m-%d %H:%M:%S"),
        "end_time"       : df["timestamp"].max().strftime("%Y-%m-%d %H:%M:%S"),
        "duration_min"   : round(
            (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 60, 1
        ),
        "hourly_avg"     : hourly,
    }


def print_stats(stats: dict) -> None:
    """Pretty-print a statistics summary to the console."""
    sep = "─" * 52
    print(f"\n{sep}")
    print("  📊  CROWD ANALYTICS REPORT")
    print(sep)
    print(f"  Period        : {stats['start_time']}  →  {stats['end_time']}")
    print(f"  Duration      : {stats['duration_min']} minutes")
    print(f"  Log entries   : {stats['total_records']}")
    print(sep)
    print(f"  Average count : {stats['avg_crowd']} persons")
    print(f"  Maximum count : {stats['max_crowd']} persons")
    print(f"  Minimum count : {stats['min_crowd']} persons")
    print(f"  Std deviation : {stats['std_dev']}")
    print(sep)
    print(f"  Peak hour     : {stats['peak_hour_label']}")
    print(f"  Peak hour avg : {stats['peak_hour_avg']} persons")
    print(sep + "\n")


# ─── Plotting ─────────────────────────────────────────────────────────────────

def _stat_card(ax, label: str, value: str, color: str, unit: str = "") -> None:
    """Render a stat card inside a given Axes."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Card background
    ax.add_patch(plt.Rectangle(
        (0.05, 0.05), 0.90, 0.90,
        transform=ax.transAxes,
        facecolor=GRID_COLOR, alpha=0.5,
        linewidth=1.5, edgecolor=color,
        zorder=1,
    ))

    ax.text(0.5, 0.72, label,  ha="center", va="center",
            fontsize=10, color=TEXT_MUTED,  transform=ax.transAxes)
    ax.text(0.5, 0.42, value,  ha="center", va="center",
            fontsize=26, color=color, fontweight="bold", transform=ax.transAxes)
    if unit:
        ax.text(0.5, 0.18, unit,  ha="center", va="center",
                fontsize=9,  color=TEXT_MUTED, transform=ax.transAxes)


def _trend_panel(ax, df: pd.DataFrame, stats: dict) -> None:
    """Line chart: crowd count over time with rolling average and shaded area."""
    x = df["timestamp"]
    y = df["crowd_count"]

    # Raw values (translucent)
    ax.fill_between(x, y, alpha=0.15, color=ACCENT_TEAL)
    ax.plot(x, y, color=ACCENT_TEAL, alpha=0.45, linewidth=1, label="Raw count")

    # Rolling mean (5-sample window)
    if len(df) >= 5:
        roll = y.rolling(window=5, center=True, min_periods=1).mean()
        ax.plot(x, roll, color=ACCENT_TEAL, linewidth=2.5, label="Rolling avg (5)")

    # Average line
    avg = stats["avg_crowd"]
    ax.axhline(avg, color=ACCENT_AMBER, linewidth=1.5,
               linestyle="--", alpha=0.85, label=f"Overall avg ({avg})")

    # Max annotation
    max_idx = y.idxmax()
    ax.annotate(
        f"Peak: {stats['max_crowd']}",
        xy=(x[max_idx], y[max_idx]),
        xytext=(15, 12), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color=ACCENT_CORAL, lw=1.5),
        color=ACCENT_CORAL, fontsize=9, fontweight="bold",
    )

    # Axis formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    ax.set_title("Crowd Trend Over Time", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Time", labelpad=8)
    ax.set_ylabel("Person Count", labelpad=8)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True)
    ax.legend(fontsize=9, framealpha=0.3)


def _hourly_panel(ax, stats: dict) -> None:
    """Bar chart: average crowd count per hour of the day."""
    hourly = stats["hourly_avg"]
    hours  = hourly.index.tolist()
    values = hourly.values.tolist()

    # Colour bars — highlight the peak hour
    colors = [
        ACCENT_CORAL if h == stats["peak_hour"] else ACCENT_TEAL
        for h in hours
    ]

    bars = ax.bar(hours, values, color=colors, width=0.7, alpha=0.85, zorder=2)

    # Value labels on top of each bar
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            f"{val:.1f}",
            ha="center", va="bottom",
            fontsize=7.5, color=TEXT_MUTED,
        )

    ax.set_title("Average Crowd by Hour of Day", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Hour (24-h)", labelpad=8)
    ax.set_ylabel("Avg Person Count", labelpad=8)
    ax.set_xticks(range(0, 24))
    ax.set_xlim(-0.5, 23.5)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, axis="y")

    # Peak hour annotation
    ph = stats["peak_hour"]
    ax.annotate(
        f"Peak hour\n{ph:02d}:00",
        xy=(ph, stats["peak_hour_avg"]),
        xytext=(ph + 1.5 if ph < 20 else ph - 4, stats["peak_hour_avg"] + 0.5),
        arrowprops=dict(arrowstyle="->", color=ACCENT_CORAL, lw=1.5),
        color=ACCENT_CORAL, fontsize=9, fontweight="bold",
    )


def build_figure(df: pd.DataFrame, stats: dict) -> plt.Figure:
    """Compose the full multi-panel analytics figure."""

    # Layout: 2 rows
    #   Row 0: [trend chart spanning all 3 columns]
    #   Row 1: [hourly bar chart (2 cols)] [3 stat cards (1 col each)]
    fig = plt.figure(figsize=(16, 10), facecolor=DARK_BG)
    fig.suptitle(
        "📊  Crowd Detection — Analytics Report",
        fontsize=16, fontweight="bold", color=TEXT_PRIMARY, y=0.97,
    )

    gs = fig.add_gridspec(
        2, 4,
        height_ratios=[1.6, 1],
        hspace=0.42, wspace=0.35,
        left=0.06, right=0.97,
        top=0.92, bottom=0.08,
    )

    ax_trend  = fig.add_subplot(gs[0, :])          # full-width trend chart
    ax_hourly = fig.add_subplot(gs[1, :2])          # hourly bar (left half)
    ax_card1  = fig.add_subplot(gs[1, 2])           # stat card 1
    ax_card2  = fig.add_subplot(gs[1, 3])           # stat card 2
    # We'll squeeze a third card by splitting ax_card2 column later via text

    _trend_panel(ax_trend, df, stats)
    _hourly_panel(ax_hourly, stats)

    _stat_card(ax_card1,
               "Average Crowd",
               str(stats["avg_crowd"]),
               ACCENT_TEAL,
               "persons / snapshot")

    _stat_card(ax_card2,
               "Peak Hour",
               f"{stats['peak_hour']:02d}:00",
               ACCENT_CORAL,
               f"avg {stats['peak_hour_avg']} persons")

    # Subtitle with session metadata
    session_info = (
        f"Session: {stats['start_time']}  →  {stats['end_time']}  "
        f"| {stats['total_records']} log entries  "
        f"| Max: {stats['max_crowd']}  Min: {stats['min_crowd']}  "
        f"| σ: {stats['std_dev']}"
    )
    fig.text(0.5, 0.005, session_info, ha="center", va="bottom",
             fontsize=8, color=TEXT_MUTED)

    return fig


# ─── Entry point ─────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate crowd analytics from a crowd_log.csv file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv", type=str, default="logs/crowd_log.csv",
        help="Path to the CSV file produced by logger.py.",
    )
    parser.add_argument(
        "--out", type=str, default="logs/crowd_analysis.png",
        help="Output path for the saved PNG report.",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Save the PNG without opening an interactive window.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"[Analyse] Reading → {os.path.abspath(args.csv)}")
    df = load_csv(args.csv)

    # ── Compute & print stats ────────────────────────────────────────────────
    stats = compute_stats(df)
    print_stats(stats)

    # ── Build and save figure ────────────────────────────────────────────────
    fig = build_figure(df, stats)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"[Analyse] Report saved → {out_path.resolve()}")

    if not args.no_show:
        plt.show()

    plt.close(fig)
    print("[Analyse] Done.")


if __name__ == "__main__":
    main()

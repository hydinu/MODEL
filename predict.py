# =============================================================================
# predict.py — Crowd-Count Forecasting using scikit-learn (Random Forest)
# =============================================================================
"""
Trains a Random Forest Regressor on historical crowd-count data and predicts
the crowd count for the next 15 minutes (30 steps × 30 s each).

Features used
─────────────
  Cyclical time  : sin/cos encoding of hour and minute
  Calendar       : day-of-week, is_weekend flag
  Lag values     : crowd_count at t-1, t-2, t-3, t-5, t-10
  Rolling stats  : 5-step rolling mean and rolling std-dev

Forecasting strategy
────────────────────
  Recursive multi-step forecasting:
    ŷ(t+1) = f(features_t,   lag_t,   lag_{t-1}, …)
    ŷ(t+2) = f(features_t+1, ŷ(t+1), lag_t,     …)
    …
  Each predicted value is fed back as a lag for the next step.

Usage
─────
  python predict.py                       # reads logs/crowd_log.csv
  python predict.py --csv custom.csv      # custom CSV
  python predict.py --demo               # generate synthetic data & run
  python predict.py --no-show            # save PNG only, skip window
  python predict.py --steps 60           # predict 60 steps (30 min)
"""

import argparse
import os
import sys
import warnings
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 0. Styling
# ─────────────────────────────────────────────────────────────────────────────

DARK_BG      = "#0d1117"
PANEL_BG     = "#161b22"
GRID_COLOR   = "#30363d"
ACCENT_TEAL  = "#39d0d8"
ACCENT_AMBER = "#f0a500"
ACCENT_CORAL = "#ff6b6b"
ACCENT_VIOLET= "#a371f7"
TEXT_PRIMARY = "#e6edf3"
TEXT_MUTED   = "#8b949e"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,  "axes.facecolor":   PANEL_BG,
    "axes.edgecolor":   GRID_COLOR,"axes.labelcolor":  TEXT_PRIMARY,
    "axes.titlecolor":  TEXT_PRIMARY,"xtick.color":    TEXT_MUTED,
    "ytick.color":      TEXT_MUTED, "grid.color":      GRID_COLOR,
    "grid.linestyle":   "--",       "grid.alpha":      0.55,
    "text.color":       TEXT_PRIMARY,"font.family":    "sans-serif",
    "font.size":        10,          "lines.linewidth": 2,
})


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data loading
# ─────────────────────────────────────────────────────────────────────────────

# Lag window sizes (in number of 30-second steps)
LAG_STEPS = [1, 2, 3, 5, 10]
ROLL_WIN   = 5          # rolling-statistics window


def load_csv(path: str) -> pd.DataFrame:
    """Read crowd_log.csv and return a clean, sorted DataFrame."""
    p = Path(path)
    if not p.exists():
        sys.exit(
            f"[Error] CSV not found: {p.resolve()}\n"
            "Run  main.py  to generate data, or use  --demo  for synthetic data."
        )
    df = pd.read_csv(p)
    if not {"timestamp", "crowd_count"}.issubset(df.columns):
        sys.exit("[Error] CSV must contain 'timestamp' and 'crowd_count' columns.")
    df["timestamp"]   = pd.to_datetime(df["timestamp"], infer_datetime_format=True)
    df["crowd_count"] = pd.to_numeric(df["crowd_count"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    if len(df) < 20:
        sys.exit(f"[Error] Need ≥ 20 log rows to train; found {len(df)}. Use --demo.")
    return df


def make_demo_data(n: int = 300) -> pd.DataFrame:
    """
    Generate synthetic crowd data with realistic daily rhythm,
    weekend dip, random noise, and occasional spikes.
    Sampling interval: 30 seconds (matches logger.py default).
    """
    np.random.seed(42)
    start = datetime(2026, 6, 9, 8, 0, 0)          # Monday 08:00
    ts    = [start + timedelta(seconds=30 * i) for i in range(n)]

    counts = []
    for t in ts:
        h   = t.hour + t.minute / 60
        # Bimodal daily curve: morning peak ~10h, afternoon peak ~15h
        base  = (
            6 * np.exp(-((h - 10) ** 2) / 4)
            + 8 * np.exp(-((h - 15) ** 2) / 6)
            + 2
        )
        # Weekend dip
        if t.weekday() >= 5:
            base *= 0.55
        # Random walk component
        noise = np.random.normal(0, 1.0)
        spike = np.random.choice([0, 5], p=[0.97, 0.03])
        counts.append(max(0, int(round(base + noise + spike))))

    return pd.DataFrame({"timestamp": ts, "crowd_count": counts})


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def encode_time_cyclically(series_hour: pd.Series,
                           series_minute: pd.Series,
                           series_dow: pd.Series) -> pd.DataFrame:
    """
    Cyclical sine/cosine encoding of hour and minute.

    Why?  Hour 23 and hour 0 are only 1 hour apart, but numerically they
    are 23 apart.  Projecting onto a unit circle fixes this:

        hour_sin  = sin(2π · h / 24)
        hour_cos  = cos(2π · h / 24)
        min_sin   = sin(2π · m / 60)
        min_cos   = cos(2π · m / 60)
        dow_sin   = sin(2π · d / 7)
        dow_cos   = cos(2π · d / 7)
    """
    return pd.DataFrame({
        "hour_sin"  : np.sin(2 * np.pi * series_hour   / 24),
        "hour_cos"  : np.cos(2 * np.pi * series_hour   / 24),
        "minute_sin": np.sin(2 * np.pi * series_minute / 60),
        "minute_cos": np.cos(2 * np.pi * series_minute / 60),
        "dow_sin"   : np.sin(2 * np.pi * series_dow    / 7),
        "dow_cos"   : np.cos(2 * np.pi * series_dow    / 7),
        "is_weekend": (series_dow >= 5).astype(int),
    })


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the full feature matrix from the raw DataFrame.

    Feature groups
    ──────────────
    • Cyclical time  (6 cols) : hour_sin, hour_cos, minute_sin, minute_cos,
                                 dow_sin,  dow_cos
    • Calendar       (1 col)  : is_weekend
    • Lag values     (5 cols) : lag_1 … lag_10
    • Rolling stats  (2 cols) : rolling_mean_5, rolling_std_5

    Total: 14 features per sample.
    """
    out = df.copy()

    # ── Time features
    time_df = encode_time_cyclically(
        out["timestamp"].dt.hour,
        out["timestamp"].dt.minute,
        out["timestamp"].dt.dayofweek,
    )
    out = pd.concat([out, time_df], axis=1)

    # ── Lag features  (shift by k steps)
    for k in LAG_STEPS:
        out[f"lag_{k}"] = out["crowd_count"].shift(k)

    # ── Rolling statistics over the last ROLL_WIN steps
    out["rolling_mean"] = (
        out["crowd_count"].shift(1).rolling(window=ROLL_WIN, min_periods=1).mean()
    )
    out["rolling_std"] = (
        out["crowd_count"].shift(1).rolling(window=ROLL_WIN, min_periods=1).std().fillna(0)
    )

    # Drop rows that have NaN lags (first LAG_STEPS[-1] rows)
    out.dropna(inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


FEATURE_COLS = (
    ["hour_sin", "hour_cos", "minute_sin", "minute_cos",
     "dow_sin",  "dow_cos",  "is_weekend"]
    + [f"lag_{k}" for k in LAG_STEPS]
    + ["rolling_mean", "rolling_std"]
)
TARGET_COL = "crowd_count"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model Training
# ─────────────────────────────────────────────────────────────────────────────

def train_model(df_feat: pd.DataFrame):
    """
    Train a Random Forest Regressor using walk-forward (time-series) CV.

    Random Forest maths (brief)
    ───────────────────────────
    Given B trees, each trained on a bootstrap sample of the data:

        ŷ(x) = (1/B) Σ_{b=1}^{B}  T_b(x)          ← ensemble mean

    Each tree T_b is grown by:
      • At every split: choose the best feature from a random subset of size
        √(n_features)  (reduces correlation between trees).
      • Split criterion: minimise MSE = (1/N) Σ (y_i - ȳ)²

    Prediction variance (used for confidence bands):
        σ²(x) = (1/(B-1)) Σ_{b=1}^{B}  (T_b(x) - ŷ(x))²

    Returns
    ───────
    model  : fitted RandomForestRegressor
    scaler : fitted StandardScaler (for time features)
    metrics: dict of cross-validated MAE, RMSE, R²
    """
    X = df_feat[FEATURE_COLS].values
    y = df_feat[TARGET_COL].values

    # ── Scale features (helps regularisation; RF itself is scale-invariant
    #    but good practice for reproducibility)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    # ── Time-series cross validation (no data leakage)
    tscv     = TimeSeriesSplit(n_splits=5)
    mae_list, rmse_list, r2_list = [], [], []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_sc), 1):
        Xtr, Xval = X_sc[train_idx], X_sc[val_idx]
        ytr, yval = y[train_idx],    y[val_idx]

        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=3,
            max_features="sqrt",      # √p features per split
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(Xtr, ytr)
        ypred = rf.predict(Xval)

        mae_list.append(mean_absolute_error(yval, ypred))
        rmse_list.append(np.sqrt(mean_squared_error(yval, ypred)))
        r2_list.append(r2_score(yval, ypred))

        print(f"  Fold {fold}/5 → MAE={mae_list[-1]:.3f}  "
              f"RMSE={rmse_list[-1]:.3f}  R²={r2_list[-1]:.3f}")

    metrics = {
        "mae"  : round(float(np.mean(mae_list)),  3),
        "rmse" : round(float(np.mean(rmse_list)), 3),
        "r2"   : round(float(np.mean(r2_list)),   3),
    }

    # ── Final model trained on ALL data
    model = RandomForestRegressor(
        n_estimators=200, max_depth=10,
        min_samples_leaf=3, max_features="sqrt",
        random_state=42, n_jobs=-1,
    )
    model.fit(X_sc, y)

    return model, scaler, metrics


# ─────────────────────────────────────────────────────────────────────────────
# 4. Recursive Multi-Step Forecasting
# ─────────────────────────────────────────────────────────────────────────────

def _time_features(ts: datetime) -> list:
    """Return the 7 cyclical+calendar features for a given datetime."""
    h   = ts.hour
    m   = ts.minute
    dow = ts.weekday()
    return [
        np.sin(2 * np.pi * h   / 24), np.cos(2 * np.pi * h   / 24),
        np.sin(2 * np.pi * m   / 60), np.cos(2 * np.pi * m   / 60),
        np.sin(2 * np.pi * dow / 7),  np.cos(2 * np.pi * dow / 7),
        int(dow >= 5),
    ]


def _tree_predictions(model: RandomForestRegressor, x_scaled: np.ndarray) -> np.ndarray:
    """Return per-tree predictions for confidence interval estimation."""
    return np.array([tree.predict(x_scaled)[0] for tree in model.estimators_])


def forecast(
    model: RandomForestRegressor,
    scaler: StandardScaler,
    df_feat: pd.DataFrame,
    n_steps: int = 30,          # 30 steps × 30 s = 15 minutes
    step_seconds: int = 30,
) -> pd.DataFrame:
    """
    Recursive multi-step forecast.

    Algorithm
    ─────────
    Maintain a sliding window of the last max(LAG_STEPS) crowd values.
    For each future step t+k:

        1. Build time features for timestamp t+k.
        2. Compute lag features from the sliding window.
        3. Compute rolling mean/std from the sliding window.
        4. Assemble feature vector x_{t+k}.
        5. ŷ_{t+k} = model.predict(scaler.transform(x_{t+k}))
        6. Append ŷ_{t+k} to the sliding window (feed-forward).

    Confidence interval (±1 σ across trees):
        lower = ŷ - std(tree predictions)
        upper = ŷ + std(tree predictions)

    Parameters
    ──────────
    n_steps      : number of 30-second steps to predict
    step_seconds : seconds per step (must match logger interval)
    """
    max_lag    = max(LAG_STEPS)
    last_ts    = df_feat["timestamp"].iloc[-1]

    # Seed the lag window with the last max_lag observed values
    history = deque(
        df_feat["crowd_count"].iloc[-max_lag:].tolist(),
        maxlen=max_lag,
    )

    future_ts, preds, lower_ci, upper_ci = [], [], [], []

    for step in range(1, n_steps + 1):
        next_ts  = last_ts + timedelta(seconds=step_seconds * step)
        hist_arr = list(history)        # oldest → newest

        # ── Assemble feature vector
        time_feats = _time_features(next_ts)

        lag_feats = [
            hist_arr[-k] if k <= len(hist_arr) else 0
            for k in LAG_STEPS
        ]

        window = hist_arr[-ROLL_WIN:] if len(hist_arr) >= ROLL_WIN else hist_arr
        roll_mean = float(np.mean(window))
        roll_std  = float(np.std(window)) if len(window) > 1 else 0.0

        x_raw = np.array([time_feats + lag_feats + [roll_mean, roll_std]])
        x_sc  = scaler.transform(x_raw)

        # ── Ensemble mean prediction
        tree_preds = _tree_predictions(model, x_sc)
        y_hat      = float(np.mean(tree_preds))
        y_std      = float(np.std(tree_preds))

        y_hat_clipped = max(0.0, y_hat)

        future_ts.append(next_ts)
        preds.append(y_hat_clipped)
        lower_ci.append(max(0.0, y_hat_clipped - y_std))
        upper_ci.append(y_hat_clipped + y_std)

        # ── Feed prediction back as lag for next step
        history.append(round(y_hat_clipped))

    return pd.DataFrame({
        "timestamp": future_ts,
        "predicted": preds,
        "lower_ci" : lower_ci,
        "upper_ci" : upper_ci,
    })


# ─────────────────────────────────────────────────────────────────────────────
# 5. Feature Importance
# ─────────────────────────────────────────────────────────────────────────────

def feature_importance_df(model: RandomForestRegressor) -> pd.DataFrame:
    """
    Random Forest feature importances (Mean Decrease in Impurity).

    Each tree records, for every feature, the total weighted reduction in
    node impurity (MSE) achieved by splitting on that feature.
    The final importance is the average across all trees, normalised to
    sum to 1.
    """
    importances = model.feature_importances_
    fi = pd.DataFrame({
        "feature"   : FEATURE_COLS,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return fi


# ─────────────────────────────────────────────────────────────────────────────
# 6. Plotting
# ─────────────────────────────────────────────────────────────────────────────

def build_figure(
    df_feat  : pd.DataFrame,
    forecast_df: pd.DataFrame,
    fi_df    : pd.DataFrame,
    metrics  : dict,
) -> plt.Figure:
    """Compose a 3-panel dark-themed forecast report figure."""

    fig = plt.figure(figsize=(18, 11), facecolor=DARK_BG)
    fig.suptitle(
        "🤖  Crowd Count Forecast — Random Forest (scikit-learn)",
        fontsize=16, fontweight="bold", color=TEXT_PRIMARY, y=0.97,
    )

    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1.6, 1],
        hspace=0.42, wspace=0.32,
        left=0.06, right=0.97,
        top=0.92, bottom=0.08,
    )

    ax_fore  = fig.add_subplot(gs[0, :])   # Full-width forecast panel
    ax_fi    = fig.add_subplot(gs[1, 0])   # Feature importance bar
    ax_stats = fig.add_subplot(gs[1, 1])   # Model metrics card

    # ── Panel 1: Historical + Forecast ──────────────────────────────────────
    hist_x   = df_feat["timestamp"]
    hist_y   = df_feat["crowd_count"]
    fut_x    = forecast_df["timestamp"]
    fut_y    = forecast_df["predicted"]
    lo       = forecast_df["lower_ci"]
    hi       = forecast_df["upper_ci"]

    # Historical line
    ax_fore.fill_between(hist_x, hist_y, alpha=0.12, color=ACCENT_TEAL)
    ax_fore.plot(hist_x, hist_y, color=ACCENT_TEAL, linewidth=1.8,
                 alpha=0.7, label="Historical")

    # Forecast line + CI band
    ax_fore.fill_between(fut_x, lo, hi, alpha=0.20, color=ACCENT_AMBER,
                         label="±1σ Confidence")
    ax_fore.plot(fut_x, fut_y, color=ACCENT_AMBER, linewidth=2.5,
                 linestyle="--", label="Forecast (RF)", zorder=5)

    # Boundary marker
    ax_fore.axvline(hist_x.iloc[-1], color=ACCENT_CORAL, linewidth=1.5,
                    linestyle=":", alpha=0.9)
    ax_fore.text(
        hist_x.iloc[-1], ax_fore.get_ylim()[1] if ax_fore.get_ylim()[1] != 0 else 1,
        "  Now", color=ACCENT_CORAL, fontsize=9, va="top",
    )

    ax_fore.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_fore.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax_fore.get_xticklabels(), rotation=30, ha="right")

    ax_fore.set_title(
        f"Crowd Count: Historical + Next {len(forecast_df) // 2} min Forecast",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax_fore.set_xlabel("Time", labelpad=8)
    ax_fore.set_ylabel("Person Count", labelpad=8)
    ax_fore.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax_fore.grid(True)
    ax_fore.legend(fontsize=9, framealpha=0.3, loc="upper left")

    # ── Panel 2: Feature Importance ─────────────────────────────────────────
    fi_top = fi_df.head(10)
    colors = [ACCENT_CORAL if i == 0 else ACCENT_TEAL
              for i in range(len(fi_top))]
    bars = ax_fi.barh(
        fi_top["feature"][::-1],
        fi_top["importance"][::-1],
        color=colors[::-1], alpha=0.85, zorder=2,
    )
    ax_fi.set_title("Top 10 Feature Importances", fontsize=12,
                    fontweight="bold", pad=10)
    ax_fi.set_xlabel("Mean Decrease in Impurity (normalised)")
    ax_fi.grid(True, axis="x")

    for bar, val in zip(bars, fi_top["importance"][::-1]):
        ax_fi.text(
            bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left",
            fontsize=8, color=TEXT_MUTED,
        )

    # ── Panel 3: Model Metrics Card ──────────────────────────────────────────
    ax_stats.set_xlim(0, 1)
    ax_stats.set_ylim(0, 1)
    ax_stats.axis("off")

    # Card background
    ax_stats.add_patch(plt.Rectangle(
        (0.03, 0.03), 0.94, 0.94,
        transform=ax_stats.transAxes,
        facecolor=GRID_COLOR, alpha=0.4,
        linewidth=1.5, edgecolor=ACCENT_VIOLET,
    ))

    lines = [
        ("Model",          "Random Forest Regressor",    TEXT_PRIMARY,  11),
        ("Cross-val",      "5-Fold Walk-Forward",         TEXT_MUTED,    9),
        ("",               "",                            TEXT_MUTED,    6),
        ("MAE",            f"{metrics['mae']:.3f} persons",ACCENT_TEAL, 13),
        ("RMSE",           f"{metrics['rmse']:.3f} persons",ACCENT_AMBER,13),
        ("R²",             f"{metrics['r2']:.3f}",        ACCENT_CORAL, 13),
        ("",               "",                            TEXT_MUTED,    6),
        ("Forecast steps", f"{len(forecast_df)} × 30 s", TEXT_MUTED,    9),
        ("Horizon",        f"{len(forecast_df) // 2} min",ACCENT_VIOLET,11),
    ]
    y_pos = 0.93
    for label, value, color, fsize in lines:
        if label:
            ax_stats.text(0.10, y_pos, f"{label}:", fontsize=9,
                          color=TEXT_MUTED, transform=ax_stats.transAxes, va="top")
            ax_stats.text(0.55, y_pos, value, fontsize=fsize,
                          color=color, fontweight="bold",
                          transform=ax_stats.transAxes, va="top")
        y_pos -= 0.10

    ax_stats.set_title("Model Performance (CV)", fontsize=12,
                       fontweight="bold", pad=10)

    # Footer
    fig.text(
        0.5, 0.005,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
        f"Training samples: {len(df_feat)}  |  Features: {len(FEATURE_COLS)}  |  Trees: 200",
        ha="center", fontsize=8, color=TEXT_MUTED,
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. Console Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(metrics: dict, forecast_df: pd.DataFrame, fi_df: pd.DataFrame) -> None:
    sep = "─" * 56
    print(f"\n{sep}")
    print("  🤖  CROWD FORECAST REPORT")
    print(sep)
    print(f"  Model          : Random Forest Regressor (200 trees)")
    print(f"  Validation     : 5-Fold Walk-Forward CV (no data leakage)")
    print(f"  Features       : {len(FEATURE_COLS)} total")
    print(sep)
    print(f"  MAE            : {metrics['mae']:.3f} persons")
    print(f"  RMSE           : {metrics['rmse']:.3f} persons")
    print(f"  R²             : {metrics['r2']:.3f}")
    print(sep)
    print(f"\n  Forecast horizon: {len(forecast_df)} steps "
          f"({len(forecast_df) // 2} minutes)\n")
    print(f"  {'Timestamp':<22}  {'Predicted':>9}  {'Lower CI':>9}  {'Upper CI':>9}")
    print(f"  {'─'*22}  {'─'*9}  {'─'*9}  {'─'*9}")
    for _, row in forecast_df.iterrows():
        print(
            f"  {row['timestamp'].strftime('%Y-%m-%d %H:%M:%S'):<22}"
            f"  {row['predicted']:>8.1f}"
            f"  {row['lower_ci']:>8.1f}"
            f"  {row['upper_ci']:>8.1f}"
        )
    print(f"\n  Top-3 most important features:")
    for i, (_, r) in enumerate(fi_df.head(3).iterrows(), 1):
        print(f"    {i}. {r['feature']:<18}  {r['importance']:.4f}")
    print(f"\n{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Forecast crowd count using Random Forest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv",     default="logs/crowd_log.csv",
                   help="Path to crowd_log.csv.")
    p.add_argument("--out",     default="logs/crowd_forecast.png",
                   help="Output PNG path.")
    p.add_argument("--steps",   type=int, default=30,
                   help="Number of 30-second steps to forecast (30 = 15 min).")
    p.add_argument("--demo",    action="store_true",
                   help="Use synthetic demo data instead of a real CSV.")
    p.add_argument("--no-show", action="store_true",
                   help="Save PNG without opening interactive window.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Load / generate data ──────────────────────────────────────────────────
    if args.demo:
        print("[Predict] Generating synthetic demo data …")
        raw_df = make_demo_data(n=300)
    else:
        print(f"[Predict] Loading → {os.path.abspath(args.csv)}")
        raw_df = load_csv(args.csv)

    # ── Feature engineering ───────────────────────────────────────────────────
    print("[Predict] Engineering features …")
    df_feat = build_features(raw_df)
    print(f"          {len(df_feat)} samples  ×  {len(FEATURE_COLS)} features")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("[Predict] Training Random Forest (5-fold walk-forward CV) …")
    model, scaler, metrics = train_model(df_feat)

    # ── Forecast ──────────────────────────────────────────────────────────────
    print(f"[Predict] Forecasting {args.steps} steps ({args.steps // 2} min) …")
    forecast_df = forecast(model, scaler, df_feat, n_steps=args.steps)

    # ── Feature importances ───────────────────────────────────────────────────
    fi_df = feature_importance_df(model)

    # ── Print summary ─────────────────────────────────────────────────────────
    print_summary(metrics, forecast_df, fi_df)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = build_figure(df_feat, forecast_df, fi_df, metrics)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"[Predict] Forecast chart saved → {out_path.resolve()}")

    if not args.no_show:
        plt.show()

    plt.close(fig)
    print("[Predict] Done.")


if __name__ == "__main__":
    main()

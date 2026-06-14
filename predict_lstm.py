# =============================================================================
# predict_lstm.py — Multi-Horizon Crowd Forecasting with TensorFlow LSTM
# =============================================================================
"""
Trains a two-layer stacked LSTM that simultaneously predicts crowd count at
three future horizons from a single forward pass:

    Horizon 1 :  15 minutes  (30 steps × 30 s)
    Horizon 2 :  30 minutes  (60 steps × 30 s)
    Horizon 3 :   1 hour    (120 steps × 30 s)

Architecture (direct multi-output)
────────────────────────────────────
    Input      (batch, SEQ_LEN=20, FEATURES=3)
      │         features = [crowd_norm, hour_sin, hour_cos]
      ▼
    LSTM-1     64 units · tanh · return_sequences=True
    Dropout    20 %
      ▼
    LSTM-2     32 units · tanh · return_sequences=False
    Dropout    20 %
      ▼
    Dense      32 units · ReLU
      ▼
    Dense      3 units  · linear   (one output per horizon)

Loss  : Mean Squared Error (sum over all three heads)
Opt   : Adam (lr=1e-3, with ReduceLROnPlateau)
Callbacks: EarlyStopping · ModelCheckpoint · ReduceLROnPlateau

Usage
─────
    python predict_lstm.py              # reads logs/crowd_log.csv
    python predict_lstm.py --demo       # synthetic data (no webcam needed)
    python predict_lstm.py --epochs 60  # override max epochs
    python predict_lstm.py --no-show    # save PNG, skip interactive window
    python predict_lstm.py --save-model # save trained model to logs/lstm_model.keras
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"       # suppress TF C++ info logs
warnings.filterwarnings("ignore")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks as keras_callbacks
except ImportError:
    sys.exit(
        "[Error] TensorFlow is not installed.\n"
        "Install it with:  pip install tensorflow>=2.16.0\n"
        "GPU build      :  pip install tensorflow[and-cuda]"
    )

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

tf.random.set_seed(42)
np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# 0. Theming
# ─────────────────────────────────────────────────────────────────────────────

DARK_BG       = "#0d1117"
PANEL_BG      = "#161b22"
GRID_COLOR    = "#30363d"
ACCENT_TEAL   = "#39d0d8"
ACCENT_AMBER  = "#f0a500"
ACCENT_CORAL  = "#ff6b6b"
ACCENT_VIOLET = "#a371f7"
ACCENT_GREEN  = "#3fb950"
TEXT_PRIMARY  = "#e6edf3"
TEXT_MUTED    = "#8b949e"

H15_COLOR = ACCENT_TEAL
H30_COLOR = ACCENT_AMBER
H60_COLOR = ACCENT_CORAL

plt.rcParams.update({
    "figure.facecolor": DARK_BG,   "axes.facecolor":  PANEL_BG,
    "axes.edgecolor":   GRID_COLOR, "axes.labelcolor": TEXT_PRIMARY,
    "axes.titlecolor":  TEXT_PRIMARY,"xtick.color":    TEXT_MUTED,
    "ytick.color":      TEXT_MUTED, "grid.color":      GRID_COLOR,
    "grid.linestyle":   "--",       "grid.alpha":      0.5,
    "text.color":       TEXT_PRIMARY,"font.family":    "sans-serif",
    "font.size":        10,          "lines.linewidth": 2,
})


# ─────────────────────────────────────────────────────────────────────────────
# 1. Constants
# ─────────────────────────────────────────────────────────────────────────────

SEQ_LEN      = 20          # look-back window  (20 × 30 s = 10 min of history)
STEP_SECONDS = 30          # seconds per CSV row  (matches logger.py default)

# Horizon definitions: (label, steps_ahead, colour)
HORIZONS = [
    ("15 min", 30,  H15_COLOR),
    ("30 min", 60,  H30_COLOR),
    ("1 hour", 120, H60_COLOR),
]
MAX_HORIZON = max(h[1] for h in HORIZONS)   # 120 steps


# ─────────────────────────────────────────────────────────────────────────────
# 2. Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    """Load and validate crowd_log.csv."""
    p = Path(path)
    if not p.exists():
        sys.exit(
            f"[Error] CSV not found: {p.resolve()}\n"
            "Run  main.py  to generate data, or pass  --demo  for synthetic data."
        )
    df = pd.read_csv(p)
    if not {"timestamp", "crowd_count"}.issubset(df.columns):
        sys.exit("[Error] CSV must contain 'timestamp' and 'crowd_count'.")
    df["timestamp"]   = pd.to_datetime(df["timestamp"], infer_datetime_format=True)
    df["crowd_count"] = pd.to_numeric(df["crowd_count"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    min_rows = SEQ_LEN + MAX_HORIZON + 10
    if len(df) < min_rows:
        sys.exit(f"[Error] Need ≥ {min_rows} rows; found {len(df)}. Use --demo.")
    return df


def make_demo_data(n: int = 400) -> pd.DataFrame:
    """
    Synthetic crowd data with realistic bimodal daily rhythm,
    weekend dip, autocorrelated noise, and occasional spikes.
    """
    np.random.seed(42)
    start = datetime(2026, 6, 9, 7, 0, 0)
    ts    = [start + timedelta(seconds=STEP_SECONDS * i) for i in range(n)]
    counts, prev = [], 3.0
    for t in ts:
        h    = t.hour + t.minute / 60.0
        base = (
            7  * np.exp(-((h - 10.0) ** 2) / 3.0)
            + 9 * np.exp(-((h - 15.5) ** 2) / 5.0)
            + 1.5
        )
        if t.weekday() >= 5:
            base *= 0.50
        noise = 0.6 * (np.random.randn()) + 0.3 * (prev - base)
        spike = float(np.random.choice([0, 6], p=[0.97, 0.03]))
        val   = max(0.0, base + noise + spike)
        counts.append(int(round(val)))
        prev = val
    return pd.DataFrame({"timestamp": ts, "crowd_count": counts})


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature engineering & sequence building
# ─────────────────────────────────────────────────────────────────────────────

def build_sequences(
    df: pd.DataFrame,
    scaler: MinMaxScaler | None = None,
    fit_scaler: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Construct (X, Y, timestamps, scaler) suitable for LSTM training.

    Input features per timestep
    ───────────────────────────
        crowd_norm  : crowd_count scaled to [0, 1]  via MinMaxScaler
        hour_sin    : sin(2π · h / 24)
        hour_cos    : cos(2π · h / 24)

    For each sample i the input window is rows [i … i+SEQ_LEN-1]
    and the three targets are the crowd_norm values at:
        i + SEQ_LEN + HORIZON_15 - 1
        i + SEQ_LEN + HORIZON_30 - 1
        i + SEQ_LEN + HORIZON_60 - 1

    Returns
    ───────
    X          : (N, SEQ_LEN, 3)   float32
    Y          : (N, 3)            float32  — normalised targets
    ts_index   : (N,)              timestamps of the last input step
    scaler     : fitted MinMaxScaler (inverse_transform for de-normalising)
    """
    counts = df["crowd_count"].values.reshape(-1, 1).astype(np.float32)

    if fit_scaler or scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(counts)

    counts_norm = scaler.transform(counts).flatten()

    hours   = df["timestamp"].dt.hour.values
    hour_sin = np.sin(2 * np.pi * hours / 24.0).astype(np.float32)
    hour_cos = np.cos(2 * np.pi * hours / 24.0).astype(np.float32)

    # Stack features: shape (T, 3)
    features = np.stack([counts_norm, hour_sin, hour_cos], axis=1)

    X_list, Y_list, ts_list = [], [], []
    max_h = max(h[1] for h in HORIZONS)

    for i in range(len(df) - SEQ_LEN - max_h):
        window   = features[i : i + SEQ_LEN]           # (SEQ_LEN, 3)
        targets  = np.array([
            counts_norm[i + SEQ_LEN + h - 1]
            for _, h, _ in HORIZONS
        ], dtype=np.float32)                            # (3,)

        X_list.append(window)
        Y_list.append(targets)
        ts_list.append(df["timestamp"].iloc[i + SEQ_LEN - 1])

    X  = np.array(X_list, dtype=np.float32)
    Y  = np.array(Y_list, dtype=np.float32)
    ts = np.array(ts_list)
    return X, Y, ts, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 4. LSTM model definition
# ─────────────────────────────────────────────────────────────────────────────

def build_model(seq_len: int = SEQ_LEN, n_features: int = 3) -> keras.Model:
    """
    Stacked LSTM with direct multi-output head.

    Layer-by-layer shape flow (batch dimension omitted)
    ────────────────────────────────────────────────────
    Input          : (SEQ_LEN, n_features)  = (20, 3)
    LSTM-1 out     : (SEQ_LEN, 64)          return_sequences=True keeps all steps
    Dropout-1      : (SEQ_LEN, 64)          20 % randomly zeroed
    LSTM-2 out     : (64,)                  return_sequences=False — last step only
    Dropout-2      : (64,)
    Dense-32-ReLU  : (32,)                  non-linear projection
    Dense-3-linear : (3,)                   one logit per forecast horizon
    """
    inp  = keras.Input(shape=(seq_len, n_features), name="input_seq")

    x    = layers.LSTM(64, return_sequences=True,
                       activation="tanh",
                       recurrent_activation="sigmoid",
                       name="lstm_1")(inp)
    x    = layers.Dropout(0.20, name="drop_1")(x)

    x    = layers.LSTM(32, return_sequences=False,
                       activation="tanh",
                       recurrent_activation="sigmoid",
                       name="lstm_2")(x)
    x    = layers.Dropout(0.20, name="drop_2")(x)

    x    = layers.Dense(32, activation="relu", name="dense_proj")(x)

    out  = layers.Dense(3, activation="linear", name="horizons")(x)

    model = keras.Model(inputs=inp, outputs=out, name="CrowdLSTM")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 5. Training
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    X: np.ndarray,
    Y: np.ndarray,
    max_epochs: int = 150,
    batch_size: int = 32,
    val_split: float = 0.20,
    save_path: str | None = None,
) -> tuple[keras.Model, dict]:
    """
    Train on a chronological train/val split (no shuffling — preserves
    temporal order, preventing data leakage from the future).

    Callbacks
    ─────────
    EarlyStopping      : stop if val_loss does not improve for 20 epochs;
                         restore best weights
    ReduceLROnPlateau  : halve LR when val_loss plateaus for 8 epochs
    ModelCheckpoint    : save best model weights (optional, --save-model)

    Returns
    ───────
    model   : trained keras.Model
    history : dict with keys  loss / val_loss / mae / val_mae / lr
    """
    # Chronological split
    split     = int(len(X) * (1 - val_split))
    X_tr, X_val = X[:split],  X[split:]
    Y_tr, Y_val = Y[:split],  Y[split:]

    print(f"  Train samples : {len(X_tr)}")
    print(f"  Val   samples : {len(X_val)}")
    print(f"  Input shape   : {X_tr.shape}  →  Output shape: {Y_tr.shape}\n")

    cb_list = [
        keras_callbacks.EarlyStopping(
            monitor="val_loss", patience=20,
            restore_best_weights=True, verbose=1,
        ),
        keras_callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=8, min_lr=1e-6, verbose=1,
        ),
    ]

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cb_list.append(
            keras_callbacks.ModelCheckpoint(
                save_path, monitor="val_loss",
                save_best_only=True, verbose=0,
            )
        )

    model = build_model()
    model.summary(print_fn=lambda s: print("  " + s))

    hist = model.fit(
        X_tr, Y_tr,
        validation_data=(X_val, Y_val),
        epochs=max_epochs,
        batch_size=batch_size,
        shuffle=False,           # preserve temporal order
        verbose=1,
        callbacks=cb_list,
    )

    return model, hist.history


# ─────────────────────────────────────────────────────────────────────────────
# 6. Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model: keras.Model,
    X: np.ndarray,
    Y: np.ndarray,
    scaler: MinMaxScaler,
    val_split: float = 0.20,
) -> dict:
    """
    Compute per-horizon MAE, RMSE, R² on the held-out validation set.
    Predictions are inverse-transformed back to person counts.
    """
    split = int(len(X) * (1 - val_split))
    X_val = X[split:]
    Y_val = Y[split:]

    Y_pred_norm = model.predict(X_val, verbose=0)          # (N, 3) normalised

    # Inverse-transform each column independently
    def inv(col: np.ndarray) -> np.ndarray:
        return scaler.inverse_transform(col.reshape(-1, 1)).flatten()

    metrics = {}
    for k, (label, _, _) in enumerate(HORIZONS):
        true = inv(Y_val[:, k])
        pred = inv(Y_pred_norm[:, k])
        metrics[label] = {
            "mae"  : round(float(mean_absolute_error(true, pred)), 3),
            "rmse" : round(float(np.sqrt(mean_squared_error(true, pred))), 3),
            "r2"   : round(float(r2_score(true, pred)), 3),
        }
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 7. Prediction (last window → future timestamps)
# ─────────────────────────────────────────────────────────────────────────────

def predict_future(
    model: keras.Model,
    df: pd.DataFrame,
    scaler: MinMaxScaler,
) -> dict[str, pd.DataFrame]:
    """
    For each of the three horizons, produce one point estimate using the
    most recent SEQ_LEN rows as the input window.

    Unlike recursive forecasting (which compounds errors), the direct
    multi-output head gives all three predictions in a single forward pass
    — no accumulated error.

    Returns
    ───────
    dict mapping horizon_label → DataFrame with columns:
        timestamp   : future datetime
        predicted   : persons (inverse-transformed, clipped ≥ 0)
    """
    # Build input from the last SEQ_LEN rows
    tail = df.tail(SEQ_LEN)
    counts_norm = scaler.transform(
        tail["crowd_count"].values.reshape(-1, 1)
    ).flatten()
    hours    = tail["timestamp"].dt.hour.values
    hour_sin = np.sin(2 * np.pi * hours / 24.0)
    hour_cos = np.cos(2 * np.pi * hours / 24.0)

    x_input = np.stack([counts_norm, hour_sin, hour_cos], axis=1)  # (20, 3)
    x_input = x_input[np.newaxis, :, :].astype(np.float32)          # (1, 20, 3)

    preds_norm = model.predict(x_input, verbose=0)[0]               # (3,)

    last_ts = df["timestamp"].iloc[-1]
    result  = {}

    for k, (label, steps, _) in enumerate(HORIZONS):
        future_ts  = last_ts + timedelta(seconds=STEP_SECONDS * steps)
        pred_count = float(
            scaler.inverse_transform([[preds_norm[k]]])[0][0]
        )
        pred_count = max(0.0, pred_count)
        result[label] = pd.DataFrame({
            "timestamp" : [future_ts],
            "predicted" : [round(pred_count, 1)],
        })

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 8. Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _add_horizon_marker(
    ax,
    ts: "datetime",
    value: float,
    label: str,
    color: str,
    y_offset: float = 0.5,
) -> None:
    """Draw a vertical dashed line + annotation for a single forecast point."""
    ax.axvline(ts, color=color, linewidth=1.4, linestyle=":", alpha=0.85)
    ax.scatter([ts], [value], color=color, s=90, zorder=6, edgecolors="white", linewidths=0.8)
    ax.annotate(
        f"{label}\n{value:.1f} persons",
        xy=(ts, value),
        xytext=(12, 10),
        textcoords="offset points",
        color=color,
        fontsize=8.5,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL_BG,
                  edgecolor=color, alpha=0.85),
    )


def build_figure(
    df         : pd.DataFrame,
    future     : dict[str, pd.DataFrame],
    history    : dict,
    metrics    : dict,
    scaler     : MinMaxScaler,
    X          : np.ndarray,
    Y          : np.ndarray,
    model      : keras.Model,
) -> plt.Figure:
    """4-panel dark-themed figure."""

    fig = plt.figure(figsize=(20, 13), facecolor=DARK_BG)
    fig.suptitle(
        "🧠  Crowd Forecast — Stacked LSTM  (TensorFlow / Keras)",
        fontsize=17, fontweight="bold", color=TEXT_PRIMARY, y=0.97,
    )

    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[1.6, 1],
        hspace=0.40, wspace=0.35,
        left=0.05, right=0.97,
        top=0.93, bottom=0.07,
    )

    ax_fore  = fig.add_subplot(gs[0, :])      # full-width — historical + forecast
    ax_loss  = fig.add_subplot(gs[1, 0])      # training curves
    ax_metric= fig.add_subplot(gs[1, 1])      # per-horizon bar chart
    ax_card  = fig.add_subplot(gs[1, 2])      # model summary card

    # ── Panel 1: Historical + 3 forecast markers ────────────────────────────
    hist_x = df["timestamp"]
    hist_y = df["crowd_count"]

    ax_fore.fill_between(hist_x, hist_y, alpha=0.10, color=ACCENT_TEAL)
    ax_fore.plot(hist_x, hist_y, color=ACCENT_TEAL, linewidth=1.5,
                 alpha=0.65, label="Historical")

    # Mark the "Now" boundary
    ax_fore.axvline(hist_x.iloc[-1], color="#555555", linewidth=1.2,
                    linestyle="--", alpha=0.7)
    ax_fore.text(hist_x.iloc[-1], ax_fore.get_ylim()[1] if ax_fore.get_ylim()[1] else 1,
                 "  Now", color="#aaaaaa", fontsize=8.5, va="top")

    # Plot each forecast point
    for label, h_steps, color in HORIZONS:
        row = future[label]
        _add_horizon_marker(
            ax_fore,
            row["timestamp"].iloc[0],
            row["predicted"].iloc[0],
            label, color,
        )

    ax_fore.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_fore.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax_fore.get_xticklabels(), rotation=30, ha="right")
    ax_fore.set_title("Historical Crowd + Direct Multi-Horizon LSTM Forecast",
                      fontsize=13, fontweight="bold", pad=12)
    ax_fore.set_xlabel("Time", labelpad=8)
    ax_fore.set_ylabel("Person Count", labelpad=8)
    ax_fore.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax_fore.grid(True)
    ax_fore.legend(fontsize=9, framealpha=0.3, loc="upper left")

    # ── Panel 2: Training / validation loss curves ───────────────────────────
    epochs = range(1, len(history["loss"]) + 1)
    ax_loss.plot(epochs, history["loss"],     color=ACCENT_TEAL,
                 linewidth=1.8, label="Train loss (MSE)")
    ax_loss.plot(epochs, history["val_loss"], color=ACCENT_AMBER,
                 linewidth=1.8, linestyle="--", label="Val loss (MSE)")
    best_ep = int(np.argmin(history["val_loss"])) + 1
    ax_loss.axvline(best_ep, color=ACCENT_CORAL, linewidth=1.3,
                    linestyle=":", alpha=0.8)
    ax_loss.text(best_ep + 0.5, max(history["loss"]) * 0.9,
                 f"Best\nepoch {best_ep}", color=ACCENT_CORAL, fontsize=8)
    ax_loss.set_title("Training Curves", fontsize=12, fontweight="bold", pad=10)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("MSE Loss (normalised)")
    ax_loss.legend(fontsize=8.5, framealpha=0.3)
    ax_loss.grid(True)

    # ── Panel 3: Per-horizon MAE bar chart ───────────────────────────────────
    horizon_labels = [h[0] for h in HORIZONS]
    colors         = [h[2] for h in HORIZONS]
    mae_vals  = [metrics[l]["mae"]  for l in horizon_labels]
    rmse_vals = [metrics[l]["rmse"] for l in horizon_labels]
    r2_vals   = [metrics[l]["r2"]   for l in horizon_labels]

    x_pos = np.arange(len(HORIZONS))
    width = 0.38
    b1 = ax_metric.bar(x_pos - width/2, mae_vals,  width, color=colors,
                       alpha=0.80, label="MAE",  zorder=2)
    b2 = ax_metric.bar(x_pos + width/2, rmse_vals, width, color=colors,
                       alpha=0.45, label="RMSE", zorder=2, hatch="//",
                       edgecolor="white", linewidth=0.5)

    for bar, val in zip(list(b1) + list(b2), mae_vals + rmse_vals):
        ax_metric.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.2f}", ha="center", va="bottom", fontsize=8, color=TEXT_MUTED,
        )

    ax_metric.set_xticks(x_pos)
    ax_metric.set_xticklabels(horizon_labels)
    ax_metric.set_title("Validation Metrics per Horizon",
                        fontsize=12, fontweight="bold", pad=10)
    ax_metric.set_ylabel("Persons")
    ax_metric.legend(fontsize=8.5, framealpha=0.3)
    ax_metric.grid(True, axis="y")

    # ── Panel 4: Model summary card ──────────────────────────────────────────
    ax_card.axis("off")
    ax_card.set_xlim(0, 1)
    ax_card.set_ylim(0, 1)
    ax_card.add_patch(plt.Rectangle(
        (0.02, 0.02), 0.96, 0.96,
        facecolor=GRID_COLOR, alpha=0.35,
        linewidth=1.5, edgecolor=ACCENT_VIOLET,
        transform=ax_card.transAxes,
    ))
    ax_card.set_title("Architecture Summary", fontsize=12, fontweight="bold", pad=10)

    card_lines = [
        ("Input",     f"({SEQ_LEN}, 3)",       TEXT_PRIMARY, 11),
        ("LSTM-1",    "64 units  tanh",         ACCENT_TEAL,  11),
        ("Dropout",   "20 %",                   TEXT_MUTED,   9),
        ("LSTM-2",    "32 units  tanh",         ACCENT_TEAL,  11),
        ("Dropout",   "20 %",                   TEXT_MUTED,   9),
        ("Dense",     "32 units  ReLU",         ACCENT_AMBER, 11),
        ("Output",    "3 units  linear",        ACCENT_CORAL, 11),
        ("",          "",                        TEXT_MUTED,   4),
        ("Loss",      "MSE",                    TEXT_PRIMARY, 10),
        ("Optimizer", "Adam  lr=1e-3",          ACCENT_VIOLET,10),
        ("Params",    f"{model.count_params():,}", TEXT_PRIMARY, 10),
        ("",          "",                        TEXT_MUTED,   4),
    ]
    for k, (lbl, val, color, fsize) in enumerate(HORIZONS):
        m = metrics[lbl]
        card_lines.append((
            lbl, f"MAE {m['mae']:.2f}  R² {m['r2']:.3f}", color, 9
        ))

    y_pos = 0.94
    for label, value, color, fsize in card_lines:
        if not label:
            y_pos -= 0.04
            continue
        ax_card.text(0.08, y_pos, f"{label}:", fontsize=9,
                     color=TEXT_MUTED, transform=ax_card.transAxes, va="top")
        ax_card.text(0.42, y_pos, value, fontsize=fsize, color=color,
                     fontweight="bold", transform=ax_card.transAxes, va="top")
        y_pos -= 0.075

    fig.text(
        0.5, 0.004,
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
        f"Training samples: {int(len(X)*0.8)}  |  "
        f"Seq len: {SEQ_LEN} steps ({SEQ_LEN//2} min)  |  "
        f"Horizons: {', '.join(h[0] for h in HORIZONS)}",
        ha="center", fontsize=8, color=TEXT_MUTED,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 9. Console output
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(
    future : dict[str, pd.DataFrame],
    metrics: dict,
    model  : keras.Model,
    hist   : dict,
) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print("  🧠  LSTM CROWD FORECAST  —  TensorFlow")
    print(sep)
    print(f"  Architecture : Stacked LSTM  (64→32 units)")
    print(f"  Parameters   : {model.count_params():,}")
    print(f"  Trained for  : {len(hist['loss'])} epochs")
    print(f"  Best val MSE : {min(hist['val_loss']):.6f}")
    print(sep)
    print(f"\n  {'Horizon':<10}  {'MAE':>7}  {'RMSE':>7}  {'R²':>7}")
    print(f"  {'─'*10}  {'─'*7}  {'─'*7}  {'─'*7}")
    for label, _, _ in HORIZONS:
        m = metrics[label]
        print(f"  {label:<10}  {m['mae']:>7.3f}  {m['rmse']:>7.3f}  {m['r2']:>7.3f}")
    print(sep)
    print(f"\n  Predictions from last observed window:\n")
    print(f"  {'Horizon':<10}  {'At (timestamp)':^22}  {'Persons':>8}")
    print(f"  {'─'*10}  {'─'*22}  {'─'*8}")
    for label, h_df in future.items():
        ts  = h_df["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M:%S")
        cnt = h_df["predicted"].iloc[0]
        print(f"  {label:<10}  {ts:^22}  {cnt:>8.1f}")
    print(f"\n{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-horizon crowd forecasting with TensorFlow LSTM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv",        default="logs/crowd_log.csv")
    p.add_argument("--out",        default="logs/crowd_lstm_forecast.png")
    p.add_argument("--epochs",     type=int, default=150)
    p.add_argument("--batch",      type=int, default=32)
    p.add_argument("--demo",       action="store_true",
                   help="Use synthetic data (no real CSV needed).")
    p.add_argument("--no-show",    action="store_true",
                   help="Save PNG without opening window.")
    p.add_argument("--save-model", action="store_true",
                   help="Save trained model to logs/lstm_model.keras.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # GPU memory growth (prevents OOM on shared GPUs)
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    print(f"  TensorFlow  : {tf.__version__}")
    print(f"  GPUs found  : {len(tf.config.list_physical_devices('GPU'))}\n")

    # ── Load data ────────────────────────────────────────────────────────────
    if args.demo:
        print("[LSTM] Generating synthetic demo data …")
        df = make_demo_data(n=400)
    else:
        print(f"[LSTM] Loading  →  {os.path.abspath(args.csv)}")
        df = load_csv(args.csv)

    print(f"[LSTM] Rows loaded : {len(df)}")

    # ── Sequences ────────────────────────────────────────────────────────────
    print("[LSTM] Building sequences …")
    X, Y, ts_idx, scaler = build_sequences(df, fit_scaler=True)
    print(f"       X: {X.shape}   Y: {Y.shape}")

    # ── Train ─────────────────────────────────────────────────────────────────
    save_path = "logs/lstm_model.keras" if args.save_model else None
    print("\n[LSTM] Training model …\n")
    model, history = train_model(
        X, Y,
        max_epochs=args.epochs,
        batch_size=args.batch,
        save_path=save_path,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n[LSTM] Evaluating on validation set …")
    metrics = evaluate(model, X, Y, scaler)

    # ── Forecast ──────────────────────────────────────────────────────────────
    print("[LSTM] Predicting future horizons …")
    future = predict_future(model, df, scaler)

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(future, metrics, model, history)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = build_figure(df, future, history, metrics, scaler, X, Y, model)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"[LSTM] Chart saved  →  {out_path.resolve()}")

    if not args.no_show:
        plt.show()
    plt.close(fig)
    print("[LSTM] Done.")


if __name__ == "__main__":
    main()

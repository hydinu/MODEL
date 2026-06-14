# =============================================================================
# forecaster.py — LSTM and Random Forest forecasting wrappers
# =============================================================================
"""
Both forecasters expose a single method:

    predict(series: list[dict]) -> dict

where series is a list of {timestamp, crowd_count} dicts (oldest first)
and the return value is:

    {
        'horizons': [
            {'label': '15 min', 'predicted_at': ISO, 'crowd_count': float},
            {'label': '30 min', 'predicted_at': ISO, 'crowd_count': float},
            {'label': '1 hour', 'predicted_at': ISO, 'crowd_count': float},
        ],
        'metrics': {}   # populated after training
    }
"""
from __future__ import annotations

import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

SEQ_LEN      = int(os.environ.get('SEQ_LEN', 20))
STEP_SECONDS = 30
HORIZONS = [
    ('15 min', 30),
    ('30 min', 60),
    ('1 hour', 120),
]


# ─── Shared preprocessing ─────────────────────────────────────────────────────

from sklearn.preprocessing import MinMaxScaler


def _parse_series(series: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(series)
    df['timestamp']   = pd.to_datetime(df['timestamp'], infer_datetime_format=True)
    df['crowd_count'] = pd.to_numeric(df['crowd_count'], errors='coerce').fillna(0).astype(int)
    return df.sort_values('timestamp').reset_index(drop=True)


def _build_input_window(df: pd.DataFrame, scaler: MinMaxScaler) -> np.ndarray:
    """Return the last SEQ_LEN rows as a normalised feature array (SEQ_LEN, 3)."""
    tail         = df.tail(SEQ_LEN)
    counts_norm  = scaler.transform(tail[['crowd_count']].values.astype(np.float32)).flatten()
    hours        = tail['timestamp'].dt.hour.values
    hour_sin     = np.sin(2 * np.pi * hours / 24.0)
    hour_cos     = np.cos(2 * np.pi * hours / 24.0)
    return np.stack([counts_norm, hour_sin, hour_cos], axis=1).astype(np.float32)


# ─── LSTM Forecaster ──────────────────────────────────────────────────────────

class LSTMForecaster:
    """
    Loads a pre-trained Keras model from MODEL_SAVE_PATH if available;
    otherwise trains a new model on the supplied series on every call.
    (In production, pre-train offline and mount the saved model.)
    """

    _SAVE_PATH = os.environ.get('MODEL_SAVE_PATH', '/app/models/saved/lstm_model.keras')

    def __init__(self):
        self._model  = None
        self._scaler = None
        self._try_load()

    def _try_load(self):
        if Path(self._SAVE_PATH).exists():
            try:
                import tensorflow as tf
                self._model = tf.keras.models.load_model(self._SAVE_PATH)
                print(f'[LSTM] Loaded model from {self._SAVE_PATH}')
            except Exception as exc:
                print(f'[LSTM] Could not load model: {exc}')

    def _build_model(self, seq_len: int, n_features: int = 3):
        import tensorflow as tf
        from tensorflow.keras import layers
        inp = tf.keras.Input(shape=(seq_len, n_features))
        x   = layers.LSTM(64, return_sequences=True, activation='tanh')(inp)
        x   = layers.Dropout(0.2)(x)
        x   = layers.LSTM(32, return_sequences=False, activation='tanh')(x)
        x   = layers.Dropout(0.2)(x)
        x   = layers.Dense(32, activation='relu')(x)
        out = layers.Dense(len(HORIZONS), activation='linear')(x)
        m   = tf.keras.Model(inputs=inp, outputs=out)
        m.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return m

    def _train(self, df: pd.DataFrame, scaler: MinMaxScaler):
        """Quick in-request training (use only for demo; pre-train in production)."""
        counts_norm = scaler.transform(df[['crowd_count']].values.astype(np.float32)).flatten()
        hours       = df['timestamp'].dt.hour.values
        hour_sin    = np.sin(2 * np.pi * hours / 24.0)
        hour_cos    = np.cos(2 * np.pi * hours / 24.0)
        features    = np.stack([counts_norm, hour_sin, hour_cos], axis=1)

        max_h = max(h for _, h in HORIZONS)
        X_list, Y_list = [], []
        for i in range(len(df) - SEQ_LEN - max_h):
            X_list.append(features[i:i + SEQ_LEN])
            Y_list.append([counts_norm[i + SEQ_LEN + h - 1] for _, h in HORIZONS])

        if not X_list:
            raise ValueError('Not enough data to build training sequences.')

        X = np.array(X_list, dtype=np.float32)
        Y = np.array(Y_list, dtype=np.float32)

        model = self._build_model(SEQ_LEN)
        model.fit(X, Y, epochs=30, batch_size=16, verbose=0, shuffle=False,
                  validation_split=0.1)
        self._model = model

    def predict(self, series: list[dict]) -> dict:
        import tensorflow as tf
        df = _parse_series(series)
        scaler = MinMaxScaler()
        scaler.fit(df[['crowd_count']].values.astype(np.float32))
        self._scaler = scaler

        if self._model is None:
            print('[LSTM] No saved model — training on request data …')
            self._train(df, scaler)

        window = _build_input_window(df, scaler)          # (SEQ_LEN, 3)
        x_in   = window[np.newaxis, :, :]
        preds_norm = self._model.predict(x_in, verbose=0)[0]   # (3,)
        last_ts = df['timestamp'].iloc[-1]

        horizons = []
        for k, (label, steps) in enumerate(HORIZONS):
            val = float(scaler.inverse_transform([[preds_norm[k]]])[0][0])
            horizons.append({
                'label'       : label,
                'predicted_at': (last_ts + timedelta(seconds=STEP_SECONDS * steps)).isoformat(),
                'crowd_count' : round(max(0.0, val), 1),
            })
        return {'horizons': horizons, 'metrics': {}}


# ─── Random Forest Forecaster ─────────────────────────────────────────────────

class RFForecaster:
    """Trains a Random Forest on the supplied series and predicts 15 min ahead."""

    def predict(self, series: list[dict]) -> dict:
        from sklearn.ensemble import RandomForestRegressor

        df = _parse_series(series)
        scaler = MinMaxScaler()
        counts_norm = scaler.fit_transform(
            df[['crowd_count']].values.astype(np.float32)
        ).flatten()

        LAG_STEPS = [1, 2, 3, 5, 10]
        max_lag   = max(LAG_STEPS)

        X_list, y_list = [], []
        for i in range(max_lag, len(df) - 30):
            lags    = [counts_norm[i - k] for k in LAG_STEPS]
            roll    = counts_norm[max(0, i-5):i]
            hours   = df['timestamp'].iloc[i].hour
            feats   = lags + [float(np.mean(roll)), float(np.std(roll)),
                              np.sin(2*np.pi*hours/24), np.cos(2*np.pi*hours/24)]
            X_list.append(feats)
            y_list.append(counts_norm[i + 30 - 1])

        if not X_list:
            raise ValueError('Not enough data for Random Forest sequences.')

        X = np.array(X_list)
        y = np.array(y_list)
        rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        rf.fit(X, y)

        # Predict from last window
        last_lags   = [counts_norm[-(k)] for k in LAG_STEPS]
        last_roll   = counts_norm[-5:]
        last_hour   = df['timestamp'].iloc[-1].hour
        last_feats  = last_lags + [float(np.mean(last_roll)), float(np.std(last_roll)),
                                   np.sin(2*np.pi*last_hour/24), np.cos(2*np.pi*last_hour/24)]
        pred_norm   = rf.predict([last_feats])[0]
        pred_val    = float(scaler.inverse_transform([[pred_norm]])[0][0])
        last_ts     = df['timestamp'].iloc[-1]

        return {
            'horizons': [{
                'label'       : '15 min',
                'predicted_at': (last_ts + timedelta(seconds=30*30)).isoformat(),
                'crowd_count' : round(max(0.0, pred_val), 1),
            }],
            'metrics': {}
        }

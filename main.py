# main.py — live predictor (aligned with backtest)
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import pickle
import pytz
import datetime
import time
import os
import requests
import sys
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV

# Single source of truth for essentials file path (matches backtest exporter)
ESSENTIALS_PATH = "suggested_essential_features.txt"


class NASDAQTradingModel:
    def __init__(
        self,
        model_path="NASDAQ_model.pkl",
        symbol="US100Cash",
        timeframe=mt5.TIMEFRAME_D1,
        retrain_period_days=30,
        confidence_threshold=0.60,
        webhook_url=None,
        frequency=2500,
        duration=1000,
        local_timezone="America/Toronto",
        target_threshold_factor=None,   # None = standard close→future close direction
        model_type="xgboost",
        feature_mode="all",
        use_calibration=False,
        use_adaptive_targets=False
    ):
        self.model_path = model_path
        self.symbol = symbol
        self.timeframe = timeframe
        self.retrain_period_days = int(retrain_period_days)
        self.confidence_threshold = float(confidence_threshold)
        self.webhook_url = webhook_url
        self.frequency = int(frequency)
        self.duration = int(duration)
        self.local_timezone = pytz.timezone(local_timezone)
        self.target_threshold_factor = (
            float(target_threshold_factor) if target_threshold_factor is not None else None
        )
        self.model_type = (model_type or "xgboost").lower()
        self.feature_mode = feature_mode
        self.use_calibration = bool(use_calibration)
        self.use_adaptive_targets = bool(use_adaptive_targets)
        self.model = None
        self.last_retrain_time = None
        self.feature_order = None

    # ------------------ Utilities: time/bar alignment -------------------
    def bar_seconds(self) -> int:
        tf = self.timeframe
        if tf == mt5.TIMEFRAME_M1:   return 60
        if tf == mt5.TIMEFRAME_M5:   return 5 * 60
        if tf == mt5.TIMEFRAME_M15:  return 15 * 60
        if tf == mt5.TIMEFRAME_M30:  return 30 * 60
        if tf == mt5.TIMEFRAME_H1:   return 60 * 60
        if tf == mt5.TIMEFRAME_H4:   return 4 * 60 * 60
        if tf == mt5.TIMEFRAME_D1:   return 24 * 60 * 60
        return 60  # fallback

    def _now_utc(self) -> pd.Timestamp:
        return pd.Timestamp.now(tz=pytz.UTC)

    def wait_until_next_bar(self):
        """Sleep so we wake up just after the next bar closes."""
        sec = self.bar_seconds()
        now = self._now_utc()
        epoch = int(now.timestamp())
        next_boundary = ((epoch // sec) + 1) * sec
        sleep_s = max(1, next_boundary - epoch + 1)  # +1s buffer
        time.sleep(sleep_s)

    # ------------------ Feature sets -------------------
    def get_feature_list(self, mode=None):
        mode = mode or self.feature_mode
        if mode == "all":
            return [
                # Candle shape
                "upper_wick_percent", "lower_wick_percent", "candle_range",
                # Volatility (true ATR)
                "atr_14", "volatility_ratio", "range_change",
                # Trend
                "ma_fast", "ma_slow", "trend_direction",
                # Momentum
                "rsi_14", "rsi_slope", "macd", "macd_signal", "macd_hist",
                # Breakout / S-R
                "gap", "prev_high", "prev_low", "breaks_prev_high", "breaks_prev_low",
                # Sessions & time
                "hour", "minute", "day_of_week",
                # Positioning
                "above_ma_fast", "above_ema_200", "ema_200",
                # Patterns
                "engulfing_bull", "engulfing_bear",
                "confirmed_bullish_engulf", "confirmed_bearish_engulf",
            ]
        elif mode == "essential":
            # Read once if exists; ignore commented lines starting with '#'
            if os.path.exists(ESSENTIALS_PATH):
                with open(ESSENTIALS_PATH, "r", encoding="utf-8") as f:
                    return [line.strip()
                            for line in f
                            if line.strip() and not line.lstrip().startswith("#")]
            # sensible fallback essentials (D1-friendly)
            return [
                "above_ema_200", "trend_direction", "macd_hist",
                "rsi_14", "gap", "breaks_prev_high", "breaks_prev_low",
                "confirmed_bullish_engulf", "confirmed_bearish_engulf",
            ]
        else:
            raise ValueError(f"Unknown feature mode: {mode}")

    # ------------------ Models -------------------
    def _get_model(self, model_type):
        mt_ = (model_type or "").lower()
        if mt_ == "xgboost":
            return XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss",
                n_jobs=-1,
            )
        elif mt_ == "randomforest":
            return RandomForestClassifier(
                n_estimators=400, random_state=42,
                class_weight="balanced_subsample", n_jobs=-1
            )
        elif mt_ == "lgbm":
            return LGBMClassifier(n_estimators=400, random_state=42)
        else:
            raise ValueError("Unsupported model type")

    # ------------------ Data -------------------
    def initialize_mt5(self):
        if not mt5.initialize():
            raise RuntimeError("Failed to initialize MT5")
        # Ensure symbol is visible/selected
        info = mt5.symbol_info(self.symbol)
        if info is None or not info.visible:
            if not mt5.symbol_select(self.symbol, True):
                raise RuntimeError(f"Could not select symbol '{self.symbol}'.")
        return True

    def fetch_data(self, num_bars=1000) -> pd.DataFrame:
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, int(num_bars))
        if rates is None:
            return pd.DataFrame()
        df = pd.DataFrame(rates)
        if df.empty:
            return df
        # MT5 'time' is epoch seconds at bar start (OPEN); normalize to UTC DatetimeIndex
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        df.sort_index(inplace=True)
        return df

    # ------------------ Time/index hardening -------------------
    def _coerce_time_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure df is non-empty and has a proper UTC DatetimeIndex named 'time'.
        """
        if df is None or df.empty:
            raise ValueError("Empty data: MT5 returned no rows.")

        if isinstance(df.index, pd.DatetimeIndex):
            idx = df.index
            if idx.tz is None:
                idx = idx.tz_localize("UTC")
            df = df.copy()
            df.index = idx
            df.index.name = "time"
            return df.sort_index()

        if "time" in df.columns:
            s = df["time"]
            if np.issubdtype(s.dtype, np.number):
                s = pd.to_datetime(s, unit="s", utc=True, errors="coerce")
            else:
                s = pd.to_datetime(s, utc=True, errors="coerce")
            df = df.assign(time=s).dropna(subset=["time"]).sort_values("time")
            return df.set_index("time")

        for cand in ("datetime", "date", "Date", "Time", "DATE"):
            if cand in df.columns:
                s = pd.to_datetime(df[cand], utc=True, errors="coerce")
                df = df.assign(time=s).dropna(subset=["time"]).sort_values("time")
                return df.set_index("time")

        raise ValueError("No 'time' column or DatetimeIndex found in input DataFrame")

    # ------------------ Feature engineering -------------------
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Normalize index safely on every call; do not mutate caller's df
        df = self._coerce_time_index(df.copy())
        epsilon = 1e-5  # Avoid division by zero

        # --- Candle Shape ---
        df["candle_range"] = (df["high"] - df["low"]).abs() + epsilon
        df["upper_wick_percent"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["candle_range"]
        df["lower_wick_percent"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["candle_range"]

        # --- Volatility (true ATR) ---
        tr1 = (df["high"] - df["low"]).abs()
        tr2 = (df["high"] - df["close"].shift(1)).abs()
        tr3 = (df["low"]  - df["close"].shift(1)).abs()
        tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()
        df["volatility_ratio"] = (df["high"] - df["low"]) / (df["atr_14"] + epsilon)
        df["range_change"] = (df["high"] - df["low"]) / (df["close"].shift(1) + epsilon)

        # --- Trend ---
        df["ma_fast"] = df["close"].rolling(5).mean()
        df["ma_slow"] = df["close"].rolling(20).mean()
        df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()
        df["above_ma_fast"] = (df["close"] > df["ma_fast"]).astype(int)
        df["above_ema_200"] = (df["close"] > df["ema_200"]).astype(int)
        df["trend_direction"] = np.sign(df["ma_fast"] - df["ma_slow"])

        # --- RSI ---
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / (avg_loss + epsilon)
        df["rsi_14"] = 100 - (100 / (1 + rs))
        df["rsi_slope"] = df["rsi_14"].diff()

        # --- MACD ---
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # --- Breakout / Support-Resistance ---
        df["gap"] = df["open"] - df["close"].shift(1)
        df["prev_high"] = df["high"].shift(1)
        df["prev_low"] = df["low"].shift(1)
        df["breaks_prev_high"] = (df["high"] > df["prev_high"]).astype(int)
        df["breaks_prev_low"] = (df["low"] < df["prev_low"]).astype(int)

        # --- Session/time features ---
        idx_local = df.index.tz_convert(self.local_timezone)
        df["hour"] = idx_local.hour
        df["minute"] = idx_local.minute
        df["day_of_week"] = idx_local.dayofweek

        # --- Candlestick Patterns ---
        df["engulfing_bull"] = (
            (df["close"] > df["open"]) &
            (df["close"].shift(1) < df["open"].shift(1)) &
            (df["close"] > df["open"].shift(1)) &
            (df["open"]  < df["close"].shift(1))
        ).astype(int)

        df["engulfing_bear"] = (
            (df["close"] < df["open"]) &
            (df["close"].shift(1) > df["open"].shift(1)) &
            (df["open"]  > df["close"].shift(1)) &
            (df["close"] < df["open"].shift(1))
        ).astype(int)

        df["confirmed_bullish_engulf"] = (
            (df["engulfing_bull"] == 1) &
            (df["trend_direction"] > 0) &
            (df["above_ema_200"] == 1)
        ).astype(int)

        df["confirmed_bearish_engulf"] = (
            (df["engulfing_bear"] == 1) &
            (df["trend_direction"] < 0) &
            (df["above_ema_200"] == 0)
        ).astype(int)

        # No target here (avoid confusion). Targets are applied in prepare_data().
        df = df.dropna()
        return df

    def prepare_data(self, df: pd.DataFrame):
        """
        Build labels aligned with backtest:
          - close → future close (1 bar lookahead)
          - optional fixed/adaptive thresholds
        """
        features = self.get_feature_list()
        features = [c for c in features if c in df.columns]
        df = df.copy()

        next_close = df["close"].shift(-1)
        ret1 = next_close / df["close"] - 1.0  # ALIGNMENT with backtest

        if self.use_adaptive_targets:
            df["vol_20"] = df["close"].pct_change().rolling(20).std()
            base_thr = self.target_threshold_factor or 0.005
            df["adaptive_threshold"] = base_thr * (1 + df["vol_20"] * 20)
            df["target3"] = np.where(
                ret1 >= df["adaptive_threshold"], 1,
                np.where(ret1 <= -df["adaptive_threshold"], 0, -1)
            )
            df = df[df["target3"] != -1].copy()
            df["target"] = df["target3"].astype(int)
            df.drop(columns=["target3"], inplace=True)
            df = df.iloc[:-1]
        elif self.target_threshold_factor is None:
            df["target"] = (ret1 > 0).astype(int)
            df = df.iloc[:-1]
        else:
            thr = self.target_threshold_factor
            df["target3"] = np.where(
                ret1 >= thr, 1,
                np.where(ret1 <= -thr, 0, -1)
            )
            df = df[df["target3"] != -1].copy()
            df["target"] = df["target3"].astype(int)
            df.drop(columns=["target3"], inplace=True)
            df = df.iloc[:-1]

        df.dropna(inplace=True)

        X = df[features]
        y = df["target"]
        return X, y

    # ------------------ Persistence -------------------
    def _save_model(self):
        with open(self.model_path, "wb") as f:
            meta = {"feature_order": self.feature_order,
                    "use_calibration": self.use_calibration,
                    "use_adaptive_targets": self.use_adaptive_targets,
                    "feature_mode": self.feature_mode,
                    "model_type": self.model_type}
            pickle.dump((self.model, meta), f)
        print(f"Saved model to {self.model_path}")

    def _load_model(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[1], list):
                # legacy (no meta)
                self.model, self.feature_order = obj
                print(f"Loaded legacy model (no meta) from {self.model_path}")
            else:
                self.model, meta = obj
                self.feature_order = meta.get("feature_order")
                saved_cal = meta.get("use_calibration")
                saved_mode = meta.get("feature_mode")
                saved_type = meta.get("model_type")
                if saved_cal is not None and saved_cal != self.use_calibration:
                    print(f"Warning: saved use_calibration={saved_cal} "
                          f"but current={self.use_calibration}.")
                if saved_mode and saved_mode != self.feature_mode:
                    print(f"Warning: saved feature_mode='{saved_mode}' "
                          f"but current='{self.feature_mode}'.")
                if saved_type and saved_type != self.model_type:
                    print(f"Warning: saved model_type='{saved_type}' "
                          f"but current='{self.model_type}'.")
            print(f"Loaded model from {self.model_path}")
            return True
        return False

    # ------------------ Train / Predict -------------------
    def train_model(self, X, y):
        base = self._get_model(self.model_type)

        if isinstance(base, XGBClassifier):
            n_pos = int((y == 1).sum())
            n_neg = int((y == 0).sum())
            if n_pos > 0:
                spw = max(n_neg / max(1, n_pos), 1.0)
                base.set_params(scale_pos_weight=spw)
                print(f"scale_pos_weight={spw:.3f}")

        if self.use_calibration:
            tscv = TimeSeriesSplit(n_splits=5)
            self.model = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=tscv)
            print("Training with probability calibration (sigmoid).")
        else:
            self.model = base
            print("Training WITHOUT probability calibration.")

        self.model.fit(X, y)

        if hasattr(self.model, "feature_names_in_"):
            self.feature_order = list(self.model.feature_names_in_)
        else:
            self.feature_order = list(X.columns)

        self._save_model()

    def predict_next_action(self, df_last):
        features = self.get_feature_list()
        features = [c for c in features if c in df_last.columns]
        X_pred = df_last[features].tail(1)

        if self.feature_order:
            X_pred = X_pred.reindex(columns=self.feature_order, fill_value=0)

        pred = int(self.model.predict(X_pred)[0])
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_pred)[0]
            classes = getattr(self.model, "classes_", np.array([0, 1]))
            idx_by_class = {int(c): i for i, c in enumerate(classes)}
            conf = float(proba[idx_by_class.get(pred, 1 if pred == 1 else 0)])
        else:
            conf = 0.5

        if conf > self.confidence_threshold:
            return ("BUY" if pred == 1 else "SELL"), conf
        return "HOLD", conf

    # ------------------ Close-time helpers & printer -------------------
    def _bar_close_times(self, bar_ts_open_utc: pd.Timestamp):
        close_utc = bar_ts_open_utc.tz_convert(pytz.UTC) + pd.Timedelta(seconds=self.bar_seconds())
        close_local = close_utc.tz_convert(self.local_timezone)
        return close_utc, close_local

    def _print_signal(self, row: pd.DataFrame, action: str, conf: float):
        bar_open_utc = row.index[-1]  # tz-aware UTC
        close_utc, close_local = self._bar_close_times(bar_open_utc)
        price = float(row["close"].iloc[-1])
        print(
            f"[BAR CLOSED] Local={close_local:%Y-%m-%d %H:%M} | UTC={close_utc:%Y-%m-%d %H:%M} | "
            f"close={price:.2f} -> {action} (conf={conf:.3f}, thr={self.confidence_threshold:.2f}, mode={self.feature_mode})"
        )

    # ------------------ Internal: ensure model ready -------------------
    def _ensure_model_ready(self):
        now_utc = datetime.datetime.now(pytz.utc)
        self.last_retrain_time = now_utc

        if not self._load_model():
            # If you ONLY want to use the backtest-trained model, uncomment:
            # raise RuntimeError("Backtest-trained model not found; aborting by design.")
            hist = self.fetch_data(1000)
            feats = self.create_features(hist)
            if feats.empty:
                raise RuntimeError("No data for initial training.")
            X, y = self.prepare_data(feats)
            if len(X) < 50 or y.nunique() < 2:
                raise RuntimeError("Insufficient samples or only one class for initial training.")
            self.train_model(X, y)
            self.last_retrain_time = now_utc

    # ------------------ Single-shot prediction -------------------
    def run_once(self):
        if not self.initialize_mt5():
            return
        try:
            self._ensure_model_ready()

            raw = self.fetch_data(260)
            if raw is None or raw.empty:
                print("[WARN] Empty MT5 fetch. Aborting run_once.")
                return
            feats = self.create_features(raw)
            if feats.empty:
                print("[WARN] No features built; aborting.")
                return

            # Use the last CLOSED bar only
            sec = self.bar_seconds()
            last_ts = feats.index[-1]           # OPEN time of latest bar in features
            next_open = last_ts + pd.Timedelta(seconds=sec)
            now = self._now_utc()
            closed = feats.iloc[:-1] if now < next_open else feats
            if closed.empty:
                print("[WARN] No closed bar available yet.")
                return

            row = closed.tail(1)
            action, conf = self.predict_next_action(row)
            self._print_signal(row, action, conf)

            if self.webhook_url and action != "HOLD":
                bar_open_utc = row.index[-1]
                close_local = (bar_open_utc.tz_convert(self.local_timezone)
                               + pd.Timedelta(seconds=self.bar_seconds()))
                try:
                    requests.post(self.webhook_url, json={
                        "symbol": self.symbol,
                        "timeframe": int(self.timeframe),
                        "action": action,
                        "confidence": conf,
                        "timestamp": close_local.strftime("%Y-%m-%d %H:%M:%S"),
                        "price": float(row["close"].iloc[-1]),
                        "feature_mode": self.feature_mode,
                    }, timeout=5)
                except Exception as e:
                    print("Webhook error:", e)

            try:
                if sys.platform.startswith("win") and action != "HOLD":
                    import winsound
                    winsound.Beep(self.frequency, self.duration)
            except Exception as e:
                print("Beep error:", e)

        finally:
            mt5.shutdown()

    # ------------------ Continuous live loop -------------------
    def run(self):
        if not self.initialize_mt5():
            return

        try:
            self._ensure_model_ready()

            last_bar_time = None
            while True:
                # Periodic retrain (optional, very infrequent by default)
                now_utc = datetime.datetime.now(pytz.utc)
                if (now_utc - self.last_retrain_time) >= datetime.timedelta(days=self.retrain_period_days):
                    hist = self.fetch_data(1000)
                    feats_hist = self.create_features(hist)
                    X, y = self.prepare_data(feats_hist)
                    if len(X) >= 50 and y.nunique() >= 2:
                        self.train_model(X, y)
                        self.last_retrain_time = now_utc

                # Align to next bar close
                self.wait_until_next_bar()

                # Fetch fresh data
                raw = self.fetch_data(260)
                if raw is None or raw.empty:
                    print("[WARN] Empty MT5 fetch. Retrying next bar.")
                    continue

                try:
                    feats = self.create_features(raw)
                except ValueError as e:
                    print(f"[WARN] Bad data: {e}. Retrying next bar.")
                    continue

                if feats.empty:
                    print("[WARN] Empty features. Retrying next bar.")
                    continue

                # Ensure we predict once per new closed bar
                current_bar_time = feats.index[-1]  # OPEN time of the newest bar
                if last_bar_time is not None and current_bar_time == last_bar_time:
                    time.sleep(2)
                    continue

                # Predict on the last *closed* bar
                sec = self.bar_seconds()
                next_open = current_bar_time + pd.Timedelta(seconds=sec)
                now = self._now_utc()
                closed = feats.iloc[:-1] if now < next_open else feats
                if closed.empty:
                    continue

                row = closed.tail(1)
                action, conf = self.predict_next_action(row)
                self._print_signal(row, action, conf)

                if self.webhook_url and action != "HOLD":
                    bar_open_utc = row.index[-1]
                    close_local = (bar_open_utc.tz_convert(self.local_timezone)
                                   + pd.Timedelta(seconds=self.bar_seconds()))
                    try:
                        requests.post(self.webhook_url, json={
                            "symbol": self.symbol,
                            "timeframe": int(self.timeframe),
                            "action": action,
                            "confidence": conf,
                            "timestamp": close_local.strftime("%Y-%m-%d %H:%M:%S"),
                            "price": float(row["close"].iloc[-1]),
                            "feature_mode": self.feature_mode,
                        }, timeout=5)
                    except Exception as e:
                        print("Webhook error:", e)

                try:
                    if sys.platform.startswith("win") and action != "HOLD":
                        import winsound
                        winsound.Beep(self.frequency, self.duration)
                except Exception as e:
                    print("Beep error:", e)

                # Mark we handled this bar (by OPEN time stamp)
                last_bar_time = current_bar_time

        except KeyboardInterrupt:
            print("Stopped by user")
        finally:
            mt5.shutdown()


# ------------------ PARAMETERS (edit here) ------------------
def main():
    """
    Pick symbol/timeframe/mode; model_path auto-derives from those to match backtest:
      f"{symbol}_{int(timeframe)}_{feature_mode}_{model_type}.pkl"
    """
    # === USER PARAMS ===
    SYMBOL = "US100Cash"               # e.g., "BTCUSD", "XAUUSD"
    TIMEFRAME = mt5.TIMEFRAME_D1         # switch to mt5.TIMEFRAME_M5 for 5-min live
    FEATURE_MODE = "essential"           # "all" | "essential"
    MODEL_TYPE = "randomforest"          # "xgboost" | "randomforest" | "lgbm"
    CONFIDENCE_THRESHOLD = 0.58          # set from backtest sweep
    USE_CALIBRATION = False
    USE_ADAPTIVE_TARGETS = False         # adaptive OFF (align to backtest)
    TARGET_THRESHOLD_FACTOR =None     # None => pure direction; or float like 0.005
    LOCAL_TZ = "America/Toronto"
    WEBHOOK_URL = None                   # or "https://your-webhook"
    RETRAIN_PERIOD_DAYS = 99999          # prevent retraining over the backtest fit
    ONE_SHOT = True                      # True -> run_once(); False -> run()

    # derive model_path like backtest saved it
    MODEL_PATH = f"{SYMBOL}_{int(TIMEFRAME)}_{FEATURE_MODE}_{MODEL_TYPE}.pkl"

    model = NASDAQTradingModel(
        model_path=MODEL_PATH,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        local_timezone=LOCAL_TZ,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        model_type=MODEL_TYPE,
        feature_mode=FEATURE_MODE,
        webhook_url=WEBHOOK_URL,
        target_threshold_factor=TARGET_THRESHOLD_FACTOR,
        use_calibration=USE_CALIBRATION,
        use_adaptive_targets=USE_ADAPTIVE_TARGETS,
        retrain_period_days=RETRAIN_PERIOD_DAYS
    )

    if ONE_SHOT:
        model.run_once()
    else:
        model.run()


if __name__ == "__main__":
    main()

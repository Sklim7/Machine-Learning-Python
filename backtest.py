# nasdaq_backtest.py
import os, random, warnings, pickle
from dataclasses import dataclass
from typing import Optional, List, Iterable, Dict

import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.facecolor"] = "white"
random.seed(42); np.random.seed(42)

# ------------------ Config ------------------
@dataclass(frozen=True)
class Cfg:
    symbol: str = "US100Cash"
    timeframe: int = mt5.TIMEFRAME_D1
    candles: int = 5000
    test_ratio: float = 0.2  # 20% for testing (80/20 split)

    feature_mode: str = "essential"     # "all" | "essential"
    model_type: str = "randomforest"    # "xgboost" | "randomforest"
    use_calibration: bool = False

    move_threshold: Optional[float] = None
    use_adaptive_targets: bool = False

    sweep_range: Iterable[float] = tuple(np.arange(0.50, 0.91, 0.02))
    auto_min_trades: int = 100
    fallback_threshold: float = 0.54
    auto_pick_metric: str = "winrate"   # "winrate" | "pnl"

    top_k_essential: int = 10
    export_essential_file: str = "suggested_essential_features.txt"


    # $ PnL config with profit factor
    initial_capital: float = 1_000.0
    fee_pct_per_side: float = 0.0002
    spread_pct: float = 0.0005
    allocation: float = 1.0
    risk_per_trade_pct: float = 0.01  # Risk 1% per trade
    profit_factor: float = 2.0  # 2:1 reward/risk ratio
    use_profit_factor: bool = True  # Toggle for new vs old PnL method
    clip_at_abs_ret: Optional[float] = None


# ------------------ Small utils ------------------
TF_NAME_MAP: Dict[int, str] = {
    mt5.TIMEFRAME_M1: "M1", mt5.TIMEFRAME_M5: "M5", mt5.TIMEFRAME_M15: "M15",
    mt5.TIMEFRAME_M30: "M30", mt5.TIMEFRAME_H1: "H1", mt5.TIMEFRAME_H4: "H4",
    mt5.TIMEFRAME_D1: "D1"
}
def tf_name(tf: int) -> str:
    return TF_NAME_MAP.get(tf, str(tf))


# ------------------ MT5 helpers ------------------
def ensure_symbol(symbol: str) -> None:
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"Symbol '{symbol}' not found in MT5.")
    if not info.visible:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Could not select symbol '{symbol}'.")


# ------------------ Models ------------------
def make_model(model_type: str):
    mt_ = (model_type or "").lower()
    if mt_ == "xgboost":
        return XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric="logloss", n_jobs=-1
        )
    if mt_ == "randomforest":
        return RandomForestClassifier(
            n_estimators=400, random_state=42,
            class_weight="balanced_subsample", n_jobs=-1
        )
    raise ValueError("MODEL_TYPE must be 'xgboost' or 'randomforest'")


# ------------------ Targets ------------------
def build_targets(feats: pd.DataFrame,
                  move_thr: Optional[float] = None,
                  adaptive: bool = False,
                  horizon: int = 1) -> pd.DataFrame:
    """
    Labels for direction of the next candle (default horizon=1):
      - Pure direction (default): target = 1 if close[t+h] > close[t], else 0
      - Fixed threshold (move_thr): keep only moves >= move_thr in magnitude
      - Adaptive threshold (adaptive=True): dynamic threshold from recent vol

    Notes:
    - Labels are based on close→future close (this backtest’s ground truth).
    - PnL simulation can use open→close; that is independent of labels.
    """
    f = feats.copy()
    f["next_close"] = f["close"].shift(-horizon)
    f = f.iloc[:-horizon].copy()

    retN = f["next_close"] / f["close"] - 1.0  # close→future close

    if adaptive:
        f["vol_20"] = f["close"].pct_change().rolling(20).std()
        base = move_thr if (move_thr is not None) else 0.005
        f["thr"] = base * (1 + f["vol_20"] * 20)
        lab = np.where(retN >= f["thr"], 1, np.where(retN <= -f["thr"], 0, -1))
        f = f[lab != -1].copy()
        f["target"] = lab[lab != -1].astype(int)
        return f.drop(columns=["next_close", "vol_20", "thr"])

    if move_thr is not None:
        lab = np.where(retN >= move_thr, 1, np.where(retN <= -move_thr, 0, -1))
        f = f[lab != -1].copy()
        f["target"] = lab[lab != -1].astype(int)
        return f.drop(columns=["next_close"])

    f["target"] = (retN > 0).astype(int)
    return f.drop(columns=["next_close"])


# ------------------ Sweep & Picks ------------------
def sweep_thresholds(out_df: pd.DataFrame, thresholds: Iterable[float]) -> pd.DataFrame:
    rows = []
    for thr in thresholds:
        dd = out_df[out_df["confidence"] > thr]
        n = len(dd); wins = int((dd["prediction"] == dd["target"]).sum())
        wr = (wins / n * 100) if n else 0.0
        rows.append({"thr": round(float(thr), 2), "trades": n, "win_rate_%": round(wr, 2)})
    return pd.DataFrame(rows)


def pick_by_winrate(summary: pd.DataFrame, min_trades: int, fallback: float) -> float:
    eligible = summary[summary["trades"] >= min_trades]
    if not eligible.empty:
        row = eligible.sort_values(["win_rate_%","trades"], ascending=[False, False]).iloc[0]
        return float(row["thr"])
    any_trades = summary[summary["trades"] > 0]
    if not any_trades.empty:
        row = any_trades.sort_values(["win_rate_%","trades"], ascending=[False, False]).iloc[0]
        return float(row["thr"])
    return float(fallback)


def pick_by_pnl(out_df: pd.DataFrame, thresholds: Iterable[float], cfg: Cfg) -> float:
    best_thr, best_pnl, best_trades = None, -np.inf, 0
    for thr in thresholds:
        dd = out_df[out_df["confidence"] > thr].copy()
        if dd.empty:
            continue
        dd["result"] = np.where(dd["prediction"] == dd["target"], "WIN", "LOSS")
        sim = apply_dollar_pnl(dd, cfg)
        if sim.empty:
            continue
        pnl = float(sim["pnl_$"].iloc[-1])
        if (pnl > best_pnl) or (np.isclose(pnl, best_pnl) and len(dd) > best_trades):
            best_thr, best_pnl, best_trades = float(thr), pnl, len(dd)
    return best_thr if best_thr is not None else cfg.fallback_threshold


# ------------------ $ PnL ------------------
def apply_dollar_pnl(confident_df: pd.DataFrame, cfg: Cfg) -> pd.DataFrame:
    if confident_df.empty:
        cols = sorted(set(list(confident_df.columns) + ["equity","pnl_$","drawdown_$"]))
        return pd.DataFrame(columns=cols)

    df = confident_df.copy()
    df["next_open"]  = df["open"].shift(-1)
    df["next_close"] = df["close"].shift(-1)
    df = df.iloc[:-1].copy()

    raw_ret = df["next_close"] / df["next_open"] - 1.0
    mag = (np.clip(np.abs(raw_ret), 0.0, cfg.clip_at_abs_ret)
           if cfg.clip_at_abs_ret is not None else np.abs(raw_ret))

    # Win -> +mag; loss -> -mag
    sign = np.where(df["prediction"] == df["target"], 1.0, -1.0)
    total_cost = 2.0 * cfg.fee_pct_per_side + cfg.spread_pct
    df["net_ret"] = cfg.allocation * (sign * mag - total_cost)

    equity = cfg.initial_capital
    equities = []
    for r in df["net_ret"].values:
        equity *= (1.0 + r)
        equities.append(equity)

    df["equity"] = equities
    df["pnl_$"] = df["equity"] - cfg.initial_capital
    df["equity_peak"] = df["equity"].cummax()
    df["drawdown_$"] = df["equity"] - df["equity_peak"]
    return df.drop(columns=["net_ret","equity_peak","next_open","next_close"])


def apply_dollar_pnl_with_profit_factor(confident_df: pd.DataFrame, cfg: Cfg) -> pd.DataFrame:
    """
    Apply 2:1 profit factor trading logic:
    - For each trade, risk 1% of equity for potential 2% gain
    - Stop loss at -1%, take profit at +2% (before costs)
    - Uses intrabar high/low simulation for realistic fill levels
    """
    if confident_df.empty:
        cols = sorted(
            set(list(confident_df.columns) + ["equity", "pnl_$", "drawdown_$", "trade_result", "risk_$", "reward_$"]))
        return pd.DataFrame(columns=cols)

    df = confident_df.copy()
    df["next_open"] = df["open"].shift(-1)
    df["next_high"] = df["high"].shift(-1)
    df["next_low"] = df["low"].shift(-1)
    df["next_close"] = df["close"].shift(-1)
    df = df.iloc[:-1].copy()  # Remove last row (no next bar data)

    # Trading parameters
    # Trading parameters (use cfg, not hardcoded)
    risk_per_trade_pct = cfg.risk_per_trade_pct
    profit_factor = cfg.profit_factor

    equity = cfg.initial_capital
    equities = []
    trade_results = []
    risk_amounts = []
    reward_amounts = []

    for i, row in df.iterrows():
        entry_price = row["next_open"]
        prediction = row["prediction"]

        # Calculate position size based on 1% risk
        risk_amount = equity * risk_per_trade_pct

        if prediction == 1:  # BUY signal
            # Calculate stop loss and take profit levels (before spread/fees)
            raw_stop_pct = risk_per_trade_pct  # 1% stop loss
            raw_target_pct = risk_per_trade_pct * profit_factor  # 2% take profit

            stop_loss = entry_price * (1 - raw_stop_pct)
            take_profit = entry_price * (1 + raw_target_pct)

            # Check what happened during the bar
            bar_high = row["next_high"]
            bar_low = row["next_low"]
            bar_close = row["next_close"]

            # Determine trade outcome (order matters: stop loss checked first)
            if bar_low <= stop_loss:
                # Hit stop loss
                exit_price = stop_loss
                gross_return_pct = (exit_price / entry_price) - 1
                trade_result = "LOSS"
            elif bar_high >= take_profit:
                # Hit take profit
                exit_price = take_profit
                gross_return_pct = (exit_price / entry_price) - 1
                trade_result = "WIN"
            else:
                # Neither level hit, exit at close
                exit_price = bar_close
                gross_return_pct = (exit_price / entry_price) - 1
                trade_result = "WIN" if gross_return_pct > 0 else "LOSS"

        else:  # SELL signal (prediction == 0)
            # Calculate stop loss and take profit levels for short
            raw_stop_pct = risk_per_trade_pct  # 1% stop loss
            raw_target_pct = risk_per_trade_pct * profit_factor  # 2% take profit

            stop_loss = entry_price * (1 + raw_stop_pct)
            take_profit = entry_price * (1 - raw_target_pct)

            # Check what happened during the bar
            bar_high = row["next_high"]
            bar_low = row["next_low"]
            bar_close = row["next_close"]

            # Determine trade outcome (stop loss checked first)
            if bar_high >= stop_loss:
                # Hit stop loss
                exit_price = stop_loss
                gross_return_pct = -((exit_price / entry_price) - 1)  # Invert for short
                trade_result = "LOSS"
            elif bar_low <= take_profit:
                # Hit take profit
                exit_price = take_profit
                gross_return_pct = -((exit_price / entry_price) - 1)  # Invert for short
                trade_result = "WIN"
            else:
                # Neither level hit, exit at close
                exit_price = bar_close
                gross_return_pct = -((exit_price / entry_price) - 1)  # Invert for short
                trade_result = "WIN" if gross_return_pct > 0 else "LOSS"

        # Apply costs (spread + fees)
        total_cost = 2.0 * cfg.fee_pct_per_side + cfg.spread_pct
        net_return_pct = cfg.allocation * (gross_return_pct - total_cost)

        # Calculate actual dollar amounts
        actual_risk = risk_amount if trade_result == "LOSS" else 0
        actual_reward = equity * net_return_pct if trade_result == "WIN" else 0

        # Update equity
        equity *= (1.0 + net_return_pct)

        # Store results
        equities.append(equity)
        trade_results.append(trade_result)
        risk_amounts.append(actual_risk)
        reward_amounts.append(actual_reward)

    # Add results to dataframe
    df["equity"] = equities
    df["pnl_$"] = df["equity"] - cfg.initial_capital
    df["equity_peak"] = df["equity"].cummax()
    df["drawdown_$"] = df["equity"] - df["equity_peak"]
    df["trade_result"] = trade_results
    df["risk_$"] = risk_amounts
    df["reward_$"] = reward_amounts

    # Clean up temporary columns
    return df.drop(columns=["next_open", "next_high", "next_low", "next_close"])
# ------------------ Essentials Export ------------------
def export_essentials(features: List[str], tree_importance: Optional[pd.Series],
                      top_k: int, out_path: str, meta_header: str = "") -> None:
    if tree_importance is not None and not tree_importance.empty:
        essential = list(tree_importance.sort_values(ascending=False).head(top_k).index)
    else:
        essential = features[:top_k]

    print(f"\n=== Suggested 'essential' features (top {top_k}) ===")
    for i, f in enumerate(essential, 1): print(f"{i:2d}. {f}")
    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            if meta_header:
                fh.write(f"# {meta_header}\n")
            fh.write("\n".join(essential))
        print("Saved essentials ->", out_path)
    except Exception as e:
        print(f"Could not write {out_path}: {e}")


# ------------------ Full Visual Suite ------------------
def create_comprehensive_visualizations(out_df, confident_df, summary_df,
                                        tree_importance, chosen_thr, cfg: Cfg):
    # Composite figure: equity + summary table + monthly heatmap + feature importance
    fig = plt.figure(figsize=(20, 12))

    # 1) Equity curve with wins/losses markers
    ax1 = plt.subplot(2, 3, (1, 3))
    if not confident_df.empty:
        cr = confident_df.reset_index()
        sns.lineplot(data=cr, x="time", y="pnl_$", linewidth=3, color="#2E86C1", ax=ax1)
        ax1.axhline(0, color="red", linestyle="--", alpha=0.7, linewidth=2)
        wins = cr[cr["result"] == "WIN"]; losses = cr[cr["result"] == "LOSS"]
        ax1.scatter(wins["time"], wins["pnl_$"], color="green", alpha=0.7, s=40, marker="^", label="Wins")
        ax1.scatter(losses["time"], losses["pnl_$"], color="red",   alpha=0.7, s=40, marker="v", label="Losses")
        ax1.set_title(
            f"{cfg.symbol} {tf_name(cfg.timeframe)} — {cfg.model_type.upper()} | Conf>{chosen_thr:.2f} | Mode:{cfg.feature_mode}",
            fontsize=16, fontweight="bold"
        )
        ax1.set_xlabel("Date"); ax1.set_ylabel("PnL ($)"); ax1.legend(fontsize=12); ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "No confident trades for chosen threshold", ha="center", va="center", fontsize=14)
        ax1.set_axis_off()

    # 2) Performance summary table
    ax2 = plt.subplot(2, 3, 4); ax2.axis("off")
    if not confident_df.empty:
        total = len(confident_df); wins_ct = int((confident_df["result"] == "WIN").sum())
        wr = (wins_ct / total * 100.0) if total else 0.0
        stats = pd.DataFrame({
            "Metric": ["Initial Capital ($)", "Total Trades", "Winning Trades", "Losing Trades",
                       "Win Rate (%)", "Final Equity ($)", "Total PnL ($)", "Max Drawdown ($)",
                       "Avg Confidence", "Model Type", "Profit-Factor Mode"],
            "Value": [f"{cfg.initial_capital:,.2f}", total, wins_ct, total - wins_ct,
                      f"{wr:.2f}%", f"{confident_df['equity'].iloc[-1]:,.2f}",
                      f"{confident_df['pnl_$'].iloc[-1]:,.2f}",
                      f"{confident_df['drawdown_$'].min():,.2f}",
                      f"{confident_df['confidence'].mean():.3f}", cfg.model_type.upper(),
                      "ON" if cfg.use_profit_factor else "OFF"]}
        )

        table = ax2.table(cellText=stats.values, colLabels=stats.columns,
                          cellLoc="center", loc="center", bbox=[0,0,1,1])
        table.auto_set_font_size(False); table.set_fontsize(12); table.scale(1, 2.5)
        for (i, j), cell in table.get_celld().items():
            if i == 0: cell.set_facecolor("#4CAF50"); cell.set_text_props(weight="bold", color="white")
            else:      cell.set_facecolor("#f0f0f0" if i % 2 == 0 else "white")
        ax2.set_title("Performance Summary", fontweight="bold", fontsize=16, pad=20)
    else:
        ax2.text(0.5, 0.5, "No trades to summarize", ha="center", va="center", fontsize=14)

    # 3) Monthly $PnL heatmap
    if not confident_df.empty and len(confident_df) > 30:
        ax3 = plt.subplot(2, 3, 5)
        cr = confident_df.reset_index()
        cr["year_month"] = cr["time"].dt.to_period("M")
        monthly = cr.groupby("year_month")["pnl_$"].apply(
            lambda s: s.iloc[-1] - s.iloc[0] if len(s) > 1 else s.iloc[0]
        ).reset_index(name="monthly_pnl_$")
        monthly["year"] = monthly["year_month"].dt.year
        monthly["month"] = monthly["year_month"].dt.month
        if len(monthly) > 1:
            pv = monthly.pivot(index="year", columns="month", values="monthly_pnl_$")
            sns.heatmap(pv, annot=True, cmap="RdYlGn", center=0, ax=ax3,
                        cbar_kws={"label":"Monthly PnL ($)"}, fmt=".0f")
            ax3.set_title("Monthly Strategy Performance ($ PnL)", fontweight="bold", fontsize=14)
            ax3.set_xlabel("Month"); ax3.set_ylabel("Year")

    # 4) Feature importances
    if tree_importance is not None and not tree_importance.empty:
        ax4 = plt.subplot(2, 3, 6)
        top = tree_importance.head(15)
        sns.barplot(x=top.values, y=top.index, ax=ax4, palette="viridis")
        ax4.set_title("Top 15 Feature Importances", fontweight="bold", fontsize=14)
        ax4.set_xlabel("Importance"); ax4.set_ylabel("Features")

    plt.tight_layout(); plt.show()

    # 5) Confidence distribution (separate figure)
    fig2, ax5 = plt.subplots(1, 1, figsize=(12, 6))
    if len(out_df) > 0:
        dd = out_df["confidence"].dropna()
        if len(dd) > 0:
            sns.histplot(data=dd.to_frame("confidence"), x="confidence", bins=30, kde=True, ax=ax5,
                         color="skyblue", alpha=0.7, stat="density")
            ax5.axvline(chosen_thr, color="red", linestyle="--", linewidth=3,
                        label=f"Chosen Threshold: {chosen_thr:.2f}")
            ax5.set_title("Confidence Distribution", fontweight="bold", fontsize=16)
            ax5.set_xlabel("Confidence"); ax5.set_ylabel("Density"); ax5.legend(); ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, "No confidence values to display", ha="center", va="center", fontsize=14)
            ax5.set_axis_off()
    else:
        ax5.text(0.5, 0.5, "No predictions to display", ha="center", va="center", fontsize=14)
        ax5.set_axis_off()
    plt.tight_layout(); plt.show()


# ------------------ Backtest ------------------
def backtest(cfg: Cfg):
    if not mt5.initialize():
        raise RuntimeError("MT5 initialization failed")
    ensure_symbol(cfg.symbol)
    try:
        from main import NASDAQTradingModel

        model_inst = NASDAQTradingModel(
            symbol=cfg.symbol, timeframe=cfg.timeframe,
            model_type=cfg.model_type, feature_mode=cfg.feature_mode
        )

        rates = mt5.copy_rates_from_pos(cfg.symbol, cfg.timeframe, 0, cfg.candles)
        if rates is None: raise RuntimeError("No data returned from MT5")
        df = pd.DataFrame(rates)
        if df.empty: raise RuntimeError("Empty data from MT5")
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True); df.sort_index(inplace=True)

        feats_all = model_inst.create_features(df)
        if feats_all.empty: raise RuntimeError("No features after indicator creation (insufficient history?)")

        feats = build_targets(feats_all, cfg.move_threshold, cfg.use_adaptive_targets)

        split_idx = int(len(feats) * (1 - cfg.test_ratio))  # 80% for training
        if split_idx <= 0 or split_idx >= len(feats):
            raise RuntimeError(f"Invalid split with {len(feats)} bars and test_ratio={cfg.test_ratio}")
        train, test = feats.iloc[:split_idx], feats.iloc[split_idx:]
        print(f"Train window: {train.index[0].date()} → {train.index[-1].date()}")
        print(f"Test  window: {test.index[0].date()}  → {test.index[-1].date()}")

        # Feature list sanity
        features = [c for c in model_inst.get_feature_list() if c in feats.columns]
        if not features:
            raise RuntimeError(
                "No features matched for the selected feature_mode.\n"
                "Switch feature_mode='all' or regenerate 'suggested_essential_features.txt' via backtest."
            )

        X_train, y_train = train[features], train["target"]
        X_test,  y_test  = test[features],  test["target"]




        print("Class balance (train):", y_train.value_counts(normalize=True).round(3).to_dict())
        print("Class balance (test):",  y_test.value_counts(normalize=True).round(3).to_dict())
        print("Feature count:", len(features))
        print("Top 5 feature variances:",
              train[features].var().sort_values(ascending=False).head().round(6).to_dict())

        base = make_model(cfg.model_type)
        if cfg.model_type.lower() == "xgboost":
            n_pos, n_neg = int((y_train == 1).sum()), int((y_train == 0).sum())
            spw = max(n_neg / max(1, n_pos), 1.0) if n_pos > 0 else 1.0
            if hasattr(base, "set_params"): base.set_params(scale_pos_weight=spw)
            print(f"scale_pos_weight={spw:.3f}")

        if cfg.use_calibration:
            tscv = TimeSeriesSplit(n_splits=5)
            model = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=tscv).fit(X_train, y_train)
        else:
            model = base.fit(X_train, y_train)

        feature_order = list(getattr(model, "feature_names_in_", X_train.columns))
        meta = {
            "feature_order": feature_order,
            "model_type": cfg.model_type,
            "feature_mode": cfg.feature_mode,
            "use_calibration": cfg.use_calibration,
            "use_adaptive_targets": cfg.use_adaptive_targets,
        }
        # Deterministic model filename (prevents cross-market/timeframe confusion)
        model_path = f"{cfg.symbol}_{int(cfg.timeframe)}_{cfg.feature_mode}_{cfg.model_type}.pkl"
        with open(model_path, "wb") as f: pickle.dump((model, meta), f)
        print("Saved model ->", model_path)

        # Predictions
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            classes = getattr(model, "classes_", np.array([0,1]))
            idx = {c:i for i,c in enumerate(classes)}
            p1 = proba[:, idx[1]]
            y_pred = (p1 >= 0.5).astype(int)
            conf = np.where(y_pred == 1, p1, 1 - p1)  # confidence aligned to predicted side
            print("Prediction balance:", pd.Series(y_pred).value_counts(normalize=True).round(3).to_dict())
            print("Prob UP quantiles:", [round(float(x),6) for x in np.quantile(p1, [0.1,0.25,0.5,0.75,0.9])])
        else:
            y_pred = model.predict(X_test)
            conf = np.full(len(y_pred), 0.5, float)

        out = test.copy()
        out["prediction"] = y_pred
        out["confidence"] = conf

        base_acc = (out["prediction"] == out["target"]).mean() * 100
        print(f"\nBaseline accuracy (no confidence filter): {base_acc:.2f}%")

        summary = sweep_thresholds(out, cfg.sweep_range)
        print("\n=== Confidence sweep ===")
        print(summary.to_string(index=False))

        chosen_thr = (pick_by_pnl(out, cfg.sweep_range, cfg)
                      if cfg.auto_pick_metric.lower() == "pnl"
                      else pick_by_winrate(summary, cfg.auto_min_trades, cfg.fallback_threshold))
        if chosen_thr is None:
            chosen_thr = cfg.fallback_threshold
        print(f"\nChosen threshold ({cfg.auto_pick_metric}): {chosen_thr:.2f}")

        confident = out[out["confidence"] > chosen_thr].copy()
        confident["result"] = np.where(confident["prediction"] == confident["target"], "WIN", "LOSS")

        # CSV file name reflects symbol/timeframe & adaptive
        stem = f"{cfg.symbol}_{int(cfg.timeframe)}"
        out_path = (f"{stem}_backtest_adaptive.csv" if cfg.use_adaptive_targets else f"{stem}_backtest.csv")

        if confident.empty:
            print("\nNo trades above the chosen threshold. Lower the threshold, expand test window, or switch auto_pick_metric.")
            create_comprehensive_visualizations(out, confident, summary, None, chosen_thr, cfg)
            summary.to_csv(out_path.replace(".csv", "_sweep.csv"), index=False)
            base_thr_hint = (cfg.move_threshold if not cfg.use_adaptive_targets else (cfg.move_threshold or 0.005))
            print("\n>>> Set in main.py for live:")
            print(f"    model_path = '{model_path}'")
            print(f"    confidence_threshold = {chosen_thr:.2f}")
            print(f"    use_calibration = {cfg.use_calibration}")
            print(f"    use_adaptive_targets = {cfg.use_adaptive_targets}")
            print(f"    target_threshold_factor = {('None' if base_thr_hint is None else base_thr_hint)}")
            print(f"    feature_mode = '{cfg.feature_mode}'")
            print(f"    model_type = '{cfg.model_type}'")
            print(f"    # Optional: use '{cfg.export_essential_file}' for feature_mode='essential' list")
            return

        if cfg.use_profit_factor:
            confident = apply_dollar_pnl_with_profit_factor(confident, cfg)
            confident["result"] = confident["trade_result"]  # <- normalize name
        else:
            confident = apply_dollar_pnl(confident, cfg)  # Keep original as fallback

        cr = confident.reset_index()
        if "time" not in cr.columns: cr.insert(0, "time", cr.index)
        cr[["time","close","prediction","target","confidence","result",
            "equity","pnl_$","drawdown_$"]].to_csv(out_path, index=False)

        total = len(confident); wins = int((confident["result"] == "WIN").sum())
        wr = (wins / total * 100) if total else 0.0
        print(f"\nSaved {total} confident trades -> {out_path}")
        print(f"Win rate (conf>{chosen_thr:.2f}) | mode={cfg.feature_mode} "
              f"| move_thr={cfg.move_threshold} | cal={'yes' if cfg.use_calibration else 'no'}: {wr:.2f}%")
        if total:
            print(f"Final Equity: ${confident['equity'].iloc[-1]:,.2f} | "
                  f"Total PnL: ${confident['pnl_$'].iloc[-1]:,.2f} | "
                  f"Max DD: ${confident['drawdown_$'].min():,.2f}")

        tree_importance = None
        try:
            imp = clone(make_model(cfg.model_type)).fit(X_train, y_train)
            if hasattr(imp, "feature_importances_"):
                tree_importance = pd.Series(imp.feature_importances_, index=features).sort_values(ascending=False)
                print("\n=== Tree-based feature importances ===")
                print(tree_importance.head(20).to_string())
        except Exception as e:
            print("Tree importance failed:", e)

        print("\n" + "="*50)
        print("CREATING COMPREHENSIVE VISUALIZATIONS")
        print("="*50)
        create_comprehensive_visualizations(out, confident, summary, tree_importance, chosen_thr, cfg)

        # Essentials export with header (main.py now ignores commented lines)
        if cfg.feature_mode == "all" and cfg.top_k_essential > 0:
            meta_header = f"{cfg.symbol} | {tf_name(cfg.timeframe)} | mode={cfg.feature_mode} | model={cfg.model_type}"
            export_essentials(
                features, tree_importance, cfg.top_k_essential,
                cfg.export_essential_file, meta_header
            )
        else:
            print("Skipping essentials export (only exported when feature_mode='all').")
        if len(out["confidence"]) > 0:
            q = np.quantile(out["confidence"], [0.1,0.25,0.5,0.75,0.9])
            print("Confidence quantiles:", [round(float(x),3) for x in q])

        base_thr_hint = (cfg.move_threshold if not cfg.use_adaptive_targets else (cfg.move_threshold or 0.005))
        print("\n>>> Set in main.py for live:")
        print(f"    model_path = '{model_path}'")
        print(f"    confidence_threshold = {chosen_thr:.2f}")
        print(f"    use_calibration = {cfg.use_calibration}")
        print(f"    use_adaptive_targets = {cfg.use_adaptive_targets}")
        print(f"    target_threshold_factor = {('None' if base_thr_hint is None else base_thr_hint)}")
        print(f"    feature_mode = '{cfg.feature_mode}'")
        print(f"    model_type = '{cfg.model_type}'")
        print(f"    # Optional: use '{cfg.export_essential_file}' for feature_mode='essential' list")

    finally:
        mt5.shutdown()


# ------------------ PARAMETERS (edit here) ------------------
if __name__ == "__main__":
    cfg = Cfg(
        symbol="US100Cash",
        timeframe=mt5.TIMEFRAME_D1,
        candles=2000,
        test_ratio=0.2,
        feature_mode="essential",
        model_type="randomforest",
        use_calibration=False,
        move_threshold=0.005,
        use_adaptive_targets=False,
        sweep_range=tuple(np.arange(0.50, 0.91, 0.02)),
        auto_min_trades=100,
        fallback_threshold=0.54,
        auto_pick_metric="winrate",
        top_k_essential=10,
        export_essential_file="suggested_essential_features.txt",
        initial_capital=1_000.0,
        fee_pct_per_side=0.0002,
        spread_pct=0.000,
        allocation=1.0,
        risk_per_trade_pct=0.01,  # Risk 1% per trade
        profit_factor=2.0,  # 2:1 reward/risk
        use_profit_factor=False,  # Enable new PnL method
        clip_at_abs_ret=None,
    )
    backtest(cfg)

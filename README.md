# NASDAQ Machine Learning Trading System

This repository contains a machine learning–driven trading system for the **NASDAQ-100 (US100)** index.  
It includes both **live trading functionality** (via MetaTrader5 API) and a **comprehensive backtesting framework** for model evaluation.

---

## Features

- 📈 **Backtesting (`nasdaq_backtest.py`)**
  - Train/test split using an **80/20 time-based split**
  - Support for **RandomForest, XGBoost, LightGBM** classifiers
  - **Adaptive targets** (volatility-scaled) or simple next-day directional labels
  - Automated performance analytics:
    - Accuracy, precision/recall, class balance
    - Equity curve & PnL visualization
    - Feature variance & importance plots
  - CSV export of backtest trades/results

- ⚡ **Live Trading (`main.py`)**
  - Runs on **MetaTrader5** with configurable symbol (default: `US100Cash`)
  - Predicts next-day market direction at end-of-day
  - Confidence-threshold filtering
  - Retrains model automatically (e.g., every 30 days)
  - Webhook alerts + optional beep sound notifications
  - Timezone aware (e.g., `America/Toronto`)

- 🧩 **Shared pipeline**
  - Identical feature engineering across backtest & live trading
  - Two feature modes: **all** or **essential**
  - Calibration toggle (`CalibratedClassifierCV`) for probability refinement
  - Saves trained models as `.pkl` files for reuse

---

## Project Structure

```
.
├── main.py                         # Live trading script
├── nasdaq_backtest.py              # Backtesting framework
├── suggested_essential_features.txt # Auto-exported essential feature list
├── models/                         # Saved .pkl models
├── results/                        # Backtest CSV & plots
└── README.md
```

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   **Key libraries:**
   - `MetaTrader5`
   - `pandas`, `numpy`
   - `scikit-learn`
   - `xgboost`, `lightgbm`
   - `matplotlib`, `seaborn`
   - `requests`, `pytz`

---

## Usage

### 🔹 Backtesting
Run backtest with desired config (example: 5000 candles, 20% test set, RandomForest):

```bash
python nasdaq_backtest.py
```

Generates:
- Console summary (train/test windows, accuracy, prediction balance, etc.)
- Plots (equity curve, feature importance, confidence distribution)
- CSV with trade-by-trade results

---

### 🔹 Live Trading
Run daily predictor on MetaTrader5:

```bash
python main.py
```

- Loads most recent trained model (or retrains if outdated)
- Predicts next-day direction
- Applies confidence threshold (default `0.60`)
- Sends alerts via webhook (if configured)

---

## Configuration

Both scripts use a `Cfg` dataclass for easy configuration:
- **symbol**: trading symbol (e.g., `US100-SEP25`, `BTCUSD`, etc.)
- **timeframe**: default `D1` (daily)
- **candles**: number of historical bars to fetch
- **test_ratio / test_bars**: backtest split size
- **feature_mode**: `"all"` or `"essential"`
- **model_type**: `"randomforest"`, `"xgboost"`, `"lightgbm"`
- **confidence_threshold**: filter weak predictions
- **target_threshold_factor**: optional move size filter (fixed or adaptive)

---

## Results (Example)

- Backtest (2018–2025 daily data):
  - **Prediction accuracy**: ~60%
  - **Annualized return**: ~12.6%
  - **Win/loss ratio**: ~1.2  
- Model: XGBoost with essential features  
- Confidence threshold: 0.60  

---

## License

This project is licensed under the MIT License.

---

👨‍💻 Developed by [Seungkwon Lim]  
📬 For inquiries: classkwon@gmail.com

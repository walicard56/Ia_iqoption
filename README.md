# 🧠 AI Trading Bot — IQ Option (LSTM)

> Self-improving automated trading bot using LSTM neural networks to predict price movements and execute operations on IQ Option.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white)
![IQ Option](https://img.shields.io/badge/IQ_Option_API-automation-blue?style=flat-square)
![License](https://img.shields.io/github/license/walicard56/Ia_iqoption?style=flat-square)

---

## Overview

This bot uses **Long Short-Term Memory (LSTM)** neural networks to analyze historical candlestick data from IQ Option, calculate technical indicators, and predict the direction of price movements. Based on predictions, it autonomously executes buy or sell orders.

After each trading session, the model stores results and **retrains itself in the background**, continuously adapting to market behavior over time.

---

## How It Works

```
Market Data → Feature Engineering → LSTM Model → Prediction → Execute Order
                                         ↑
                              Background Retraining (periodic)
```

1. **Data Collection** — Fetches live candle data from IQ Option API
2. **Feature Engineering** — Computes Moving Averages, RSI, and other indicators
3. **Prediction** — LSTM model outputs buy/sell signal with confidence score
4. **Execution** — Sends order to IQ Option if confidence exceeds threshold
5. **Self-improvement** — Saves results, retrains model periodically

---

## Project Structure

```
├── iq.py           # IQ Option connection and order execution
├── training.py     # LSTM model training pipeline
├── testing.py      # Model evaluation and backtesting
├── models/         # Saved model checkpoints
└── .gitignore
```

---

## Requirements

```
tensorflow
pandas
numpy
scikit-learn
iqoptionapi
```

---

## Installation

```bash
git clone https://github.com/walicard56/Ia_iqoption.git
cd Ia_iqoption
pip install -r requirements.txt
```

---

## Usage

**Train the model:**
```bash
python training.py
```

**Run live trading:**
```bash
python iq.py
```

**Evaluate performance:**
```bash
python testing.py
```

---

## Configuration

Inside `iq.py`, configure your credentials and trading parameters:

```python
EMAIL      = "your_iqoption_email"
PASSWORD   = "your_iqoption_password"
ASSET      = "EURUSD"        # Trading pair
TIMEFRAME  = 1               # Candle duration in minutes
AMOUNT     = 2               # Trade amount in USD
THRESHOLD  = 0.65            # Minimum confidence to execute trade
```

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**.  
AI-based trading does not guarantee profits. Use at your own risk.  
The author is not responsible for any financial losses.

---

## Author

**Walisson Jose** · [GitHub](https://github.com/walicard56) · [Portfolio](https://walicard56.github.io/Portifolio_wali)

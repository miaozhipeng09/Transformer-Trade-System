# ✅ Run `real_run.py` to launch a fully automated trading system (based on a simple strategy)

This project implements a lightweight **automated trading system**, using a **very simple trading logic**, designed for experimentation and rapid deployment.

## 🧠 Project Summary

Although this project is named `Transformer-Trade-System`, the current version implements a **basic rule-based trading logic**, focused on demonstrating **end-to-end automation** — from signal generation to order execution.

It is designed to be extensible to more complex models (like Transformers) later.

## 🚀 Key Features

- 📡 `real_run.py`: one-click launch of full trading loop
- 🕹️ End-to-end automation: signal generation → execution → position tracking
- 🧪 Simple decision logic (e.g., price threshold, momentum, or rule-based signals)
- 📁 Modular file structure to allow future model integration (e.g., Transformers)

## 🧱 Project Structure

/Transformer-Trade-System
├── real_run.py # 🚀 Entry point for live auto-trading
├── backtest/ # Backtest scripts and logs
├── data/ # Order book or market data
├── execution/ # Trade execution logic
├── model/ # Placeholder for future ML models
├── utils/ # Logging, config loaders, etc.
└── README.md

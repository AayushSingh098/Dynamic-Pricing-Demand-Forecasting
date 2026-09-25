# 💰 Dynamic Pricing & Demand Forecasting

An end-to-end data science project that forecasts product demand and evaluates alternative price scenarios to recommend the price with the highest estimated revenue — combining demand forecasting, price-elasticity analysis, and an interactive Streamlit dashboard.

[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-9cf)](https://lightgbm.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B)](https://streamlit.io/)

🚀 **Live Demo:** https://dynamic-pricing-demand-forecasting.streamlit.app/
---

## 🎯 Project Objective

> **If the price of a product changes, how could expected demand and revenue change?**

For a selected product and store, the application:

1. Forecasts demand using a trained LightGBM regression model.
2. Tests candidate prices from **−20% to +20%** of the current price.
3. Predicts demand and revenue for each scenario.
4. Recommends the tested price with the highest estimated revenue.
5. Displays historical price elasticity as a price-sensitivity indicator.

---

## 📊 Dataset & Feature Engineering

Trained on historical retail sales data spanning **2011-02-26 to 2016-05-22** (~13.56M processed observations). The final LightGBM model uses **17 features** across four groups:

| Group | Examples |
|---|---|
| Product & Store | `item_id`, `store_id`, `cat_id` |
| Price | `sell_price`, `previous_price`, `price_change_pct` |
| Historical Demand | `lag_1_demand`, `lag_7_demand`, `rolling_28d_demand` |
| Calendar | `wday`, `month`, `is_weekend`, `snap_active` |

Data is split **chronologically** (train / validation / test) to avoid leaking future information into the past.

---

## 📈 Model Performance

| Metric | Baseline | LightGBM |
|---|---|---|
| Validation RMSE | 3.3836 | **2.4512** |
| Test MAE | 1.8519 | **1.4207** |
| Test RMSE | 3.5531 | **2.5506** |

LightGBM reduced test RMSE by approximately **28%** over the baseline.

Historical price elasticity is estimated for **7,527 product-store combinations** (1,486 reliable individual estimates, remainder via category-level fallback), and is shown in the dashboard as an indicator — not a causal guarantee.

---

## 🖥️ Streamlit Dashboard

- Select a product and store
- Generate a pricing recommendation with current vs. recommended price
- View historical elasticity and forecasted/expected demand
- Inspect the revenue curve and full table of tested price scenarios

```bash
python -m streamlit run app/app.py
```

---

## ⚙️ Getting Started

```bash
git clone <your-repository-url>
cd "Dynamic Pricing & Demand Forecasting"
pip install -r requirements.txt
python -m streamlit run app/app.py
```

---

## 📁 Project Structure

```text
├── app/app.py                  # Streamlit dashboard
├── data/                       # raw & processed data
├── models/                     # trained LightGBM model
├── notebooks/                  # EDA to price simulation
├── src/pricing_engine.py       # scenario & revenue logic
└── requirements.txt
```

---

## 🧰 Tech Stack

Python · Pandas · NumPy · Scikit-learn · LightGBM · Joblib · PyArrow · Streamlit

---

## ⚠️ Limitation

Pricing recommendations are **scenario-based estimates**, not causal estimates. The model was trained for demand forecasting; observational data alone doesn't establish that a price change *causes* the predicted demand shift. Elasticity is presented as a sensitivity indicator, not proof of a causal effect. Future work: controlled pricing experiments, causal inference, and production monitoring.

---

## 👨‍💻 Author

**Aayush Kumar Singh** — Data Science / Data Analytics Portfolio Project
# Stock Price Predictor (Python · Flask · scikit-learn)

Interactive web app that estimates a trading day's **closing price** from its Open, High, Low and Volume using Ridge regression.

## Features
- Type a ticker (AAPL, TCS.NS, GC=F…) → loads live data from Yahoo Finance, trains a model on it and shows held-out R² / MAE / RMSE
- Live what-if sliders that re-predict instantly
- Price-history chart with your prediction, recent-predictions table
- Input validation, JSON API (`/api/predict`, `/api/stock/<ticker>`), `/health`

## Run
```bash
pip install -r requirements.txt
python app.py          # http://127.0.0.1:5000
```
Deploy: `gunicorn app:app` (see `Procfile`). Retrain built-in model: `python train_model.py --ticker GC=F`.

## Note
The model predicts the *same day's* close from that day's other values; it is not a forecast of future prices. Educational use only – not financial advice.

"""Stock Price Predictor – Flask web app.

Run locally:   python app.py
Production:    gunicorn app:app   (see Procfile)
"""
from __future__ import annotations

import logging
import os

from flask import Flask, jsonify, redirect, render_template, request, url_for

import utils

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("stock-predictor")


def _field(data, name: str):
    """Read a field from form/JSON data, accepting 'Open' or 'open'."""
    value = data.get(name)
    return value if value is not None else data.get(name.lower())


def run_prediction(data) -> dict:
    """Validate inputs, pick a model, predict. Shared by the page and the JSON API."""
    values = utils.parse_inputs(
        _field(data, "Open"), _field(data, "High"), _field(data, "Low"), _field(data, "Volume")
    )

    model, metrics, source, note = utils.load_default_model(), None, "default", None
    ticker = (data.get("ticker") or "").strip()
    if ticker:
        ticker = utils.normalize_ticker(ticker)
        try:
            bundle = utils.get_ticker_bundle(ticker)
            model, metrics, source = bundle["model"], bundle["metrics"], ticker
        except utils.DataUnavailable as exc:
            note = f"{exc} Used the built-in model instead."

    predicted = utils.predict_close(values, model)
    change = predicted - values["Open"]
    return {
        "prediction": round(predicted, 2),
        "change": round(change, 2),
        "change_pct": round(change / values["Open"] * 100, 2),
        "inputs": values,
        "model": source,
        "metrics": metrics,
        "note": note,
    }


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
@app.get("/")
def main():
    return render_template("index.html", defaults=utils.DEFAULTS)


@app.route("/predict/", methods=["GET", "POST"])
def predict():
    """Classic form submit (works without JavaScript)."""
    if request.method == "GET":
        return redirect(url_for("main"))
    try:
        result = run_prediction(request.form)
    except utils.ValidationError as exc:
        return render_template("index.html", defaults=utils.DEFAULTS, error=str(exc),
                               submitted=request.form), 400
    return render_template("prediction.html", result=result)


# --------------------------------------------------------------------------- #
# JSON API (used by the interactive page)
# --------------------------------------------------------------------------- #
@app.post("/api/predict")
def api_predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Send a JSON body with Open, High, Low and Volume."), 400
    try:
        return jsonify(run_prediction(data))
    except utils.ValidationError as exc:
        return jsonify(error=str(exc)), 400


@app.get("/api/stock/<ticker>")
def api_stock(ticker: str):
    """Latest OHLCV, recent closes for the chart, and the per-ticker model metrics."""
    try:
        bundle = utils.get_ticker_bundle(ticker)
    except utils.ValidationError as exc:
        return jsonify(error=str(exc)), 400
    except utils.DataUnavailable as exc:
        return jsonify(error=str(exc)), 503
    return jsonify(
        ticker=bundle["ticker"],
        latest=bundle["latest"],
        history=utils.history_for_chart(bundle["history"]),
        metrics=bundle["metrics"],
    )


@app.get("/health")
def health():
    return jsonify(status="ok")


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
def _wants_json() -> bool:
    return request.path.startswith("/api/")


@app.errorhandler(404)
def not_found(_e):
    if _wants_json():
        return jsonify(error="Not found."), 404
    return render_template("404.html"), 404


@app.errorhandler(405)
def method_not_allowed(_e):
    if _wants_json():
        return jsonify(error="Method not allowed."), 405
    return redirect(url_for("main"))


@app.errorhandler(Exception)
def server_error(exc):
    from werkzeug.exceptions import HTTPException

    if isinstance(exc, HTTPException):
        return exc
    log.exception("Unhandled error")
    if _wants_json():
        return jsonify(error="Something went wrong on the server."), 500
    return render_template("404.html", message="Something went wrong on the server."), 500


if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    )

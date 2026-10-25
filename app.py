"""Flask API. POST /train, POST /predict, GET /monitor, GET /logs/<train|predict>."""
import time

import pandas as pd
from flask import Flask, jsonify, request

import ingest
import logger
import model
import monitor

app = Flask(__name__)
VALID = ["all"] + ingest.COUNTRIES


@app.route("/")
def index():
    return jsonify({"service": "AAVAIL revenue forecasting", "countries": VALID,
                    "endpoints": ["/train", "/predict", "/monitor", "/logs/<train|predict>"]})


@app.route("/train", methods=["POST"])
def train():
    test = bool((request.get_json(silent=True) or {}).get("test", False))
    t0 = time.time()
    scores = model.train(test=test)
    return jsonify({"scores": scores, "runtime": time.time() - t0, "test": test})


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True) or {}
    missing = [k for k in ("country", "year", "month", "day") if k not in body]
    if missing:
        return jsonify({"error": f"missing fields: {missing}"}), 400
    country = str(body["country"]).lower()
    if country not in VALID:
        return jsonify({"error": f"unknown country '{country}'", "valid": VALID}), 400
    try:
        date = pd.Timestamp(year=int(body["year"]), month=int(body["month"]), day=int(body["day"]))
        return jsonify(model.model_predict(country, date, test=bool(body.get("test", False))))
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404


@app.route("/monitor")
def monitor_view():
    return jsonify(monitor.performance_report(test=request.args.get("test") == "1"))


@app.route("/logs/<kind>")
def logs(kind):
    if kind not in ("train", "predict"):
        return jsonify({"error": "kind must be train or predict"}), 400
    return jsonify(logger.read_log(kind, model.MODEL_VERSION, test=request.args.get("test") == "1"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

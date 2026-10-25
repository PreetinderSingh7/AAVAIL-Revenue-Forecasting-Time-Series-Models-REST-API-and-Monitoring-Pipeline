# AAVAIL Revenue Forecasting: Time-Series Models, REST API and Monitoring Pipeline

End-to-end AI workflow: data ingestion, EDA, model comparison against a baseline, a REST API,
unit tests, logging, performance monitoring and a Docker image.

> **Data note:** this repo ships with a **synthetic data generator** (`ingest.py`) so it runs anywhere.
> To use real data, put JSON files with the fields `country, invoice, price, times_viewed, year, month, day`
> (plus optional `customer_id, stream_id`) in `data/cs-train/` and re-run training. Reports in `reports/` come from synthetic data.

## Problem
Predict total revenue for the **30 days after a given date**, for one country or for all countries combined.

## Project layout
| File | Purpose |
|---|---|
| `ingest.py` | Data ingestion as a function (`fetch_data`): loads JSON, handles missing values/duplicates, builds daily time series; `python ingest.py` generates demo data |
| `eda.py` | EDA visualizations saved to `reports/eda/` |
| `model.py` | Feature engineering, **model comparison** (baseline, Ridge, Random Forest, Gradient Boosting), saving, loading, prediction |
| `logger.py` | Train and predict logging |
| `monitor.py` | Performance monitoring (model vs baseline RMSE, prediction runtime, alerts) |
| `app.py` | Flask API |
| `unittests/` | API, model, logger and ingestion tests |
| `run-tests.py` | Runs every unit test with one command |
| `Dockerfile` | Containerizes ingestion, EDA, training and the API |

## Quick start
```bash
pip install -r requirements.txt
python ingest.py      # create demo data (skip if you have your own)
python eda.py         # EDA figures -> reports/eda
python model.py       # train + compare models -> models/, logs/, reports/
python run-tests.py   # all unit tests
python app.py         # API on http://localhost:8080
```

## Docker
```bash
docker build -t aavail-forecast .
docker run -p 8080:8080 aavail-forecast
```

## API
```bash
# prediction for one country
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" \
  -d '{"country":"singapore","year":2019,"month":6,"day":1}'
# prediction for all countries combined
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" \
  -d '{"country":"all","year":2019,"month":6,"day":1}'
curl -X POST http://localhost:8080/train          # retrain (add {"test":true} to use isolated test models)
curl http://localhost:8080/monitor                # performance monitoring report
curl http://localhost:8080/logs/predict           # or /logs/train
```
Valid countries: `all, united_kingdom, singapore, france, germany, spain`.

## Testing and isolation
`python run-tests.py` runs 22 tests (API, model, logging, ingestion). Tests train and log in **test mode**:
models go to `models/test/`, logs to `logs/test/` and data to `data/test/`, all deleted afterwards, so production models and logs are never touched.

## Model comparison and monitoring
`python model.py` writes `reports/model_comparison.png` (hold-out RMSE of every model vs the baseline, per country) and
`reports/best_vs_baseline_all.png`. The baseline is the trailing 28-day revenue scaled to 30 days. Training is logged to
`logs/train-v0.1.log` and each prediction to `logs/predict-v0.1.log`. `/monitor` raises an alert when a deployed model is not better than the baseline
or mean prediction runtime exceeds 1 second. With the demo data, some countries (for example Germany) correctly raise this alert.

## Next steps
Real data, a scheduled retraining job, input-drift checks, and richer features (holidays, promotions).

"""Performance monitoring from the train and predict logs."""
import logger
from model import MODEL_VERSION


def performance_report(test=False, runtime_limit=1.0):
    """Alerts if a model is no better than the baseline or predictions get slow."""
    latest = {}
    for r in logger.read_log("train", MODEL_VERSION, test):
        latest[r["country"]] = r
    models = {}
    for c, r in latest.items():
        ratio = float(r["rmse"]) / float(r["baseline_rmse"]) if float(r["baseline_rmse"]) else None
        models[c] = {"model": r["model"], "rmse": float(r["rmse"]), "baseline_rmse": float(r["baseline_rmse"]),
                     "rmse_vs_baseline": ratio, "alert": ratio is None or ratio >= 1.0}
    preds = logger.read_log("predict", MODEL_VERSION, test)
    runtimes = [float(r["runtime"]) for r in preds]
    mean_rt = sum(runtimes) / len(runtimes) if runtimes else 0.0
    return {"models": models, "predictions_served": len(preds), "mean_runtime_seconds": mean_rt,
            "runtime_alert": mean_rt > runtime_limit,
            "any_alert": any(m["alert"] for m in models.values()) or mean_rt > runtime_limit}

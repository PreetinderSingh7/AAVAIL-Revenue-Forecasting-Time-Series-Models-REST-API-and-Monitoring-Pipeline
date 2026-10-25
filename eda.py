"""Exploratory data analysis: saves figures to reports/eda."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ingest

OUT = os.path.join(ingest.BASE, "reports", "eda")


def main():
    os.makedirs(OUT, exist_ok=True)
    if not os.path.isdir(ingest.DATA_DIR):
        ingest.generate_synthetic()
    series, df = ingest.fetch_data()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for c in ingest.COUNTRIES:
        ax.plot(series[c].set_index("date").revenue.rolling(14).mean(), label=c)
    ax.set_title("Daily revenue by country (14-day average)"); ax.set_ylabel("Revenue"); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "revenue_by_country.png"), dpi=140); plt.close()
    m = series["all"].set_index("date").revenue.groupby(lambda d: d.month).mean()
    fig, ax = plt.subplots(figsize=(8, 4)); ax.bar(m.index, m.values, color="#6C4AB6")
    ax.set_title("Average daily revenue by month (seasonality)"); ax.set_xlabel("Month"); ax.set_ylabel("Revenue")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "seasonality.png"), dpi=140); plt.close()
    tot = df.groupby("country").revenue.sum().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4)); ax.barh(tot.index, tot.values, color="#F4A259")
    ax.set_title("Total revenue by country"); ax.set_xlabel("Revenue")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "total_revenue_by_country.png"), dpi=140); plt.close()
    wk = series["all"].set_index("date").revenue.groupby(lambda d: d.dayofweek).mean()
    fig, ax = plt.subplots(figsize=(8, 4)); ax.bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], wk.values, color="#2A9D8F")
    ax.set_title("Average daily revenue by weekday"); ax.set_ylabel("Revenue")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "weekday.png"), dpi=140); plt.close()
    print("EDA figures saved to", OUT)


if __name__ == "__main__":
    main()

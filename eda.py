"""
Exploratory data analysis for the raw PM2.5 series (data/pm25_raw.csv).

Produces two figures for the README and prints a short findings summary:
  screenshots/eda_pm25_trend.png   - full hourly time series (shows gaps)
  screenshots/eda_seasonality.png  - monthly + hour-of-day (diurnal) profiles

Run: python eda.py
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: save files without opening a window
import matplotlib.pyplot as plt

RAW_PATH = "data/pm25_raw.csv"
STATION_TZ = "Asia/Bangkok"  # OpenAQ station 717 local time (UTC+7)


def load():
    df = pd.read_csv(RAW_PATH, parse_dates=["datetime"])
    df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    # Work in the station's local time so "hour of day" is physically meaningful.
    df["local"] = df["datetime"].dt.tz_convert(STATION_TZ)
    return df


def basic_report(df):
    print("=" * 60)
    print("SHAPE:", df.shape)
    print("RANGE:", df["local"].min(), "->", df["local"].max())
    print("\nPM2.5 describe:")
    print(df["pm25"].describe().round(2).to_string())
    print("\nMissing pm25 values:", int(df["pm25"].isna().sum()))

    # Gap analysis: the series is hourly, so any step > 1h is a hole.
    step_h = df["datetime"].diff().dt.total_seconds().div(3600)
    gaps = step_h[step_h > 1]
    expected = int((df["datetime"].max() - df["datetime"].min()).total_seconds() // 3600) + 1
    coverage = len(df) / expected * 100
    print(f"\nExpected hourly slots in range: {expected}")
    print(f"Actual rows: {len(df)}  ->  coverage {coverage:.1f}%")
    print(f"Number of gaps (>1h): {len(gaps)}")
    if len(gaps):
        print(f"Largest gap: {gaps.max():.0f} hours (~{gaps.max()/24:.1f} days)")
    return gaps


def plot_trend(df):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df["local"], df["pm25"], linewidth=0.6, color="#c0392b")
    # WHO 24h guideline for context (15 ug/m3).
    ax.axhline(15, color="#2c3e50", linestyle="--", linewidth=1, label="WHO 24h guideline (15)")
    ax.set_title("Hourly PM2.5 — Pluakdaeng, Thailand (OpenAQ station 717)")
    ax.set_ylabel("PM2.5 (ug/m3)")
    ax.set_xlabel("Local time (Asia/Bangkok)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig("screenshots/eda_pm25_trend.png", dpi=120)
    plt.close(fig)
    print("\nSaved screenshots/eda_pm25_trend.png")


def plot_seasonality(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    # Monthly average (calendar month) — expect dry-season (Dec-Mar) peak.
    monthly = df.groupby(df["local"].dt.month)["pm25"].mean()
    ax1.bar(monthly.index, monthly.values, color="#e67e22")
    ax1.set_title("Average PM2.5 by month")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Mean PM2.5 (ug/m3)")
    ax1.set_xticks(range(1, 13))

    # Hour-of-day average — diurnal cycle.
    hourly = df.groupby(df["local"].dt.hour)["pm25"].mean()
    ax2.plot(hourly.index, hourly.values, marker="o", color="#2980b9")
    ax2.set_title("Average PM2.5 by hour of day (local)")
    ax2.set_xlabel("Hour of day")
    ax2.set_ylabel("Mean PM2.5 (ug/m3)")
    ax2.set_xticks(range(0, 24, 3))

    fig.tight_layout()
    fig.savefig("screenshots/eda_seasonality.png", dpi=120)
    plt.close(fig)
    print("Saved screenshots/eda_seasonality.png")
    return monthly, hourly


def main():
    df = load()
    basic_report(df)
    plot_trend(df)
    monthly, hourly = plot_seasonality(df)

    print("\n" + "=" * 60)
    print("FINDINGS (for README):")
    peak_month = int(monthly.idxmax())
    low_month = int(monthly.idxmin())
    print(f"- Strongest month: {peak_month} ({monthly.max():.1f}), "
          f"cleanest: {low_month} ({monthly.min():.1f}) "
          f"-> clear seasonal cycle for the LSTM to learn.")
    print(f"- Diurnal swing: hour {int(hourly.idxmax())} highest ({hourly.max():.1f}), "
          f"hour {int(hourly.idxmin())} lowest ({hourly.min():.1f}).")


if __name__ == "__main__":
    main()

"""
Exploratory data analysis for the raw PM2.5 series (data/pm25_raw.csv).

Produces two figures for the README and prints a short findings summary:
  screenshots/eda_pm25_trend.png   - full hourly time series (shows gaps)
  screenshots/eda_seasonality.png  - monthly + hour-of-day (diurnal) profiles

Run: python eda.py
"""
import pandas as pd                      # load the CSV + group/aggregate
import matplotlib                        # plotting library
matplotlib.use("Agg")                    # headless: save files without opening a window
import matplotlib.pyplot as plt          # the pyplot drawing interface

RAW_PATH = "data/pm25_raw.csv"           # input file produced by fetch_data.py
STATION_TZ = "Asia/Bangkok"              # OpenAQ station 717 local time (UTC+7)


def load():
    """Load the CSV, tidy it, and add a local-time column for the plots."""
    df = pd.read_csv(RAW_PATH, parse_dates=["datetime"])     # read + parse dates
    # Sort by time, drop duplicate timestamps, and reset to a clean 0..N index.
    df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    # The stored time is UTC; convert to station-local time so that
    # "hour of day" lines up with real morning/evening at the station.
    df["local"] = df["datetime"].dt.tz_convert(STATION_TZ)
    return df


def basic_report(df):
    """Print shape, summary stats, and a gap analysis of the hourly series."""
    print("=" * 60)
    print("SHAPE:", df.shape)                                # (rows, columns)
    print("RANGE:", df["local"].min(), "->", df["local"].max())  # first/last time
    print("\nPM2.5 describe:")
    print(df["pm25"].describe().round(2).to_string())        # mean/std/quartiles
    print("\nMissing pm25 values:", int(df["pm25"].isna().sum()))  # NaN count

    # Gap analysis: readings are hourly, so the time step between consecutive
    # rows should be 1h. Any step larger than 1h means missing hours (a hole).
    step_h = df["datetime"].diff().dt.total_seconds().div(3600)  # gap sizes in h
    gaps = step_h[step_h > 1]                                # keep only the holes

    # "Coverage" = how many of the expected hourly slots we actually have.
    expected = int((df["datetime"].max() - df["datetime"].min()).total_seconds() // 3600) + 1
    coverage = len(df) / expected * 100
    print(f"\nExpected hourly slots in range: {expected}")
    print(f"Actual rows: {len(df)}  ->  coverage {coverage:.1f}%")
    print(f"Number of gaps (>1h): {len(gaps)}")
    if len(gaps):                                            # report the worst hole
        print(f"Largest gap: {gaps.max():.0f} hours (~{gaps.max()/24:.1f} days)")
    return gaps


def plot_trend(df):
    """Save the full hourly PM2.5 time series as a single line chart."""
    fig, ax = plt.subplots(figsize=(14, 4))                  # wide, short canvas
    ax.plot(df["local"], df["pm25"], linewidth=0.6, color="#c0392b")  # the series
    # Draw the WHO 24h guideline (15 ug/m3) as a reference line for context.
    ax.axhline(15, color="#2c3e50", linestyle="--", linewidth=1, label="WHO 24h guideline (15)")
    ax.set_title("Hourly PM2.5 — Pluakdaeng, Thailand (OpenAQ station 717)")
    ax.set_ylabel("PM2.5 (ug/m3)")
    ax.set_xlabel("Local time (Asia/Bangkok)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()                                       # trim extra margins
    fig.savefig("screenshots/eda_pm25_trend.png", dpi=120)   # write the PNG
    plt.close(fig)                                           # free the figure
    print("\nSaved screenshots/eda_pm25_trend.png")


def plot_seasonality(df):
    """Save a 2-panel figure: average PM2.5 by month and by hour of day."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))    # two side-by-side axes

    # LEFT panel: mean PM2.5 for each calendar month (1..12).
    # Expect a dry-season (Dec-Mar) peak from open burning.
    monthly = df.groupby(df["local"].dt.month)["pm25"].mean()  # group by month
    ax1.bar(monthly.index, monthly.values, color="#e67e22")
    ax1.set_title("Average PM2.5 by month")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Mean PM2.5 (ug/m3)")
    ax1.set_xticks(range(1, 13))                             # show all 12 months

    # RIGHT panel: mean PM2.5 for each hour of day (0..23) — the daily cycle.
    hourly = df.groupby(df["local"].dt.hour)["pm25"].mean()  # group by hour
    ax2.plot(hourly.index, hourly.values, marker="o", color="#2980b9")
    ax2.set_title("Average PM2.5 by hour of day (local)")
    ax2.set_xlabel("Hour of day")
    ax2.set_ylabel("Mean PM2.5 (ug/m3)")
    ax2.set_xticks(range(0, 24, 3))                          # ticks every 3 hours

    fig.tight_layout()
    fig.savefig("screenshots/eda_seasonality.png", dpi=120)
    plt.close(fig)
    print("Saved screenshots/eda_seasonality.png")
    return monthly, hourly                                   # reused for findings


def main():
    df = load()                          # load + prepare the data
    basic_report(df)                     # print stats and gap analysis
    plot_trend(df)                       # save the trend chart
    monthly, hourly = plot_seasonality(df)  # save seasonality chart + get profiles

    # Turn the two seasonal profiles into one-line, human-readable findings
    # that can be copied straight into the README.
    print("\n" + "=" * 60)
    print("FINDINGS (for README):")
    peak_month = int(monthly.idxmax())   # month with the highest average
    low_month = int(monthly.idxmin())    # month with the lowest average
    print(f"- Strongest month: {peak_month} ({monthly.max():.1f}), "
          f"cleanest: {low_month} ({monthly.min():.1f}) "
          f"-> clear seasonal cycle for the LSTM to learn.")
    print(f"- Diurnal swing: hour {int(hourly.idxmax())} highest ({hourly.max():.1f}), "
          f"hour {int(hourly.idxmin())} lowest ({hourly.min():.1f}).")


if __name__ == "__main__":
    main()

"""
Pull historical hourly PM2.5 data from the OpenAQ v3 API for a chosen
monitoring station and save it to data/pm25_raw.csv.

Requires OPENAQ_API_KEY in a .env file (see .env.example).
Get a free key at https://explore.openaq.org/register
"""
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("OPENAQ_API_KEY")
BASE_URL = "https://api.openaq.org/v3"
PM25_PARAMETER_ID = 2

COUNTRY_ISO = "TH"  # Thailand; change to "TW" for Taiwan
MONTHS_OF_HISTORY = 24  # how far back to pull


def api_get(path, params=None, max_retries=4):
    """GET a v3 endpoint with retry/backoff on transient errors (408/429/5xx)."""
    if not API_KEY:
        sys.exit(
            "Missing OPENAQ_API_KEY. Copy .env.example to .env and set your key "
            "(get one free at https://explore.openaq.org/register)."
        )
    transient = {408, 429, 500, 502, 503, 504}
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(
                f"{BASE_URL}{path}",
                headers={"X-API-Key": API_KEY},
                params=params or {},
                timeout=60,
            )
        except requests.exceptions.RequestException as exc:
            if attempt == max_retries:
                raise
            wait = 2 ** attempt
            print(f"    network error ({exc}); retry {attempt}/{max_retries} in {wait}s")
            time.sleep(wait)
            continue

        if resp.status_code in transient and attempt < max_retries:
            wait = 2 ** attempt
            print(f"    HTTP {resp.status_code}; retry {attempt}/{max_retries} in {wait}s")
            time.sleep(wait)
            continue

        resp.raise_for_status()
        return resp.json()


def find_country_id(iso_code):
    data = api_get("/countries", {"limit": 200})
    for c in data["results"]:
        if c["code"] == iso_code:
            return c["id"], c["name"]
    sys.exit(f"Could not find country with ISO code {iso_code}")


def find_pm25_location(country_id):
    """Pick the best PM2.5 station: currently active + longest history.

    LSTMs need long, continuous series to learn seasonality, so among
    stations that are still reporting we choose the one whose data spans
    the widest date range (datetimeFirst -> datetimeLast).
    """
    data = api_get(
        "/locations",
        {"countries_id": country_id, "parameters_id": PM25_PARAMETER_ID, "limit": 1000},
    )
    results = data["results"]
    if not results:
        sys.exit("No PM2.5-reporting locations found for this country.")

    def utc_of(loc, field):
        """Safely read loc[field]['utc']; field may be missing or None."""
        block = loc.get(field) or {}
        return block.get("utc") or ""

    # An "active" station reported at some point in 2026 (data pulled 2026).
    active = [loc for loc in results if utc_of(loc, "datetimeLast") >= "2026-"]
    candidates = active or results  # fall back to all if none look active

    # Longest history = earliest datetimeFirst among the most-recent-active set.
    candidates.sort(key=lambda l: utc_of(l, "datetimeFirst"))  # oldest first
    best = candidates[0]
    print(
        f"Chose station from {len(candidates)} active candidate(s); "
        f"history {utc_of(best, 'datetimeFirst')[:10]} -> "
        f"{utc_of(best, 'datetimeLast')[:10]}"
    )
    return best


def find_pm25_sensor(location):
    for sensor in location["sensors"]:
        if sensor["parameter"]["id"] == PM25_PARAMETER_ID:
            return sensor
    sys.exit("Location has no PM2.5 sensor (unexpected).")


def _fetch_window(sensor_id, win_from, win_to):
    """Fetch one date window, paging shallowly (a window holds few pages)."""
    rows = []
    page = 1
    while True:
        data = api_get(
            f"/sensors/{sensor_id}/hours",
            {
                # v3 /hours uses datetime_from/datetime_to; the date_* names
                # are silently ignored (returns the full series unfiltered).
                "datetime_from": win_from.isoformat(),
                "datetime_to": win_to.isoformat(),
                "limit": 1000,
                "page": page,
            },
        )
        results = data["results"]
        if not results:
            break
        rows.extend(results)
        if len(results) < 1000:
            break
        page += 1
        time.sleep(0.2)  # be polite to the rate limit
    return rows


def fetch_hourly_measurements(sensor_id, date_from, date_to, window_days=60):
    """Fetch hourly aggregates by walking fixed date windows.

    OpenAQ v3 degrades on deep page offsets (times out past ~18k rows), so we
    slice the full range into ~60-day windows. Each window has only ~1-2 pages,
    which keeps every request cheap and lets api_get retry transient failures.
    """
    all_rows = []
    win_from = date_from
    while win_from < date_to:
        win_to = min(win_from + timedelta(days=window_days), date_to)
        rows = _fetch_window(sensor_id, win_from, win_to)
        all_rows.extend(rows)
        print(
            f"  {win_from.date()} -> {win_to.date()}: +{len(rows)} rows "
            f"(total {len(all_rows)})"
        )
        win_from = win_to
    return all_rows


def main():
    country_id, country_name = find_country_id(COUNTRY_ISO)
    print(f"Country: {country_name} (id={country_id})")

    location = find_pm25_location(country_id)
    print(f"Location: {location['name']} (id={location['id']})")

    sensor = find_pm25_sensor(location)
    print(f"Sensor: {sensor['name']} (id={sensor['id']})")

    date_to = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=30 * MONTHS_OF_HISTORY)

    print(f"Fetching hourly PM2.5 from {date_from.date()} to {date_to.date()} ...")
    rows = fetch_hourly_measurements(sensor["id"], date_from, date_to)

    if not rows:
        sys.exit("No measurements returned. Try a different location or date range.")

    records = []
    for r in rows:
        records.append(
            {
                "datetime": r["period"]["datetimeFrom"]["utc"],
                "pm25": r["value"],
            }
        )

    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").drop_duplicates(subset="datetime")

    os.makedirs("data", exist_ok=True)
    out_path = "data/pm25_raw.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Station: {location['name']}, {country_name}")


if __name__ == "__main__":
    main()

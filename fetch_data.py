"""
Pull historical hourly PM2.5 data from the OpenAQ v3 API for a chosen
monitoring station and save it to data/pm25_raw.csv.

Flow: find country -> pick the best station -> find its PM2.5 sensor ->
download hourly values in date windows -> write a tidy CSV.

Requires OPENAQ_API_KEY in a .env file (see .env.example).
Get a free key at https://explore.openaq.org/register
"""
import os                                # read env vars + make the data folder
import sys                               # sys.exit() to stop with a message
import time                              # sleep() for polite pauses / backoff
from datetime import datetime, timedelta, timezone  # build the date range

import pandas as pd                      # assemble rows into a table + CSV
import requests                          # make the HTTP calls to the API
from dotenv import load_dotenv           # load OPENAQ_API_KEY from the .env file

load_dotenv()                            # read .env into environment variables

# ---- Configuration -------------------------------------------------------
API_KEY = os.environ.get("OPENAQ_API_KEY")   # our secret key (None if missing)
BASE_URL = "https://api.openaq.org/v3"       # root of every API request
PM25_PARAMETER_ID = 2                         # OpenAQ's numeric id for PM2.5

COUNTRY_ISO = "TH"        # Thailand; change to "TW" for Taiwan
MONTHS_OF_HISTORY = 24    # how far back to pull (24 months = 2 years)


def api_get(path, params=None, max_retries=4):
    """GET a v3 endpoint with retry/backoff on transient errors (408/429/5xx).

    Returns the parsed JSON body. Retries a few times with growing waits
    because the API occasionally times out or rate-limits under load.
    """
    # Fail fast with a helpful message if the key was never set.
    if not API_KEY:
        sys.exit(
            "Missing OPENAQ_API_KEY. Copy .env.example to .env and set your key "
            "(get one free at https://explore.openaq.org/register)."
        )
    # HTTP codes that are worth retrying (temporary, not our fault).
    transient = {408, 429, 500, 502, 503, 504}

    # Try up to max_retries times before giving up.
    for attempt in range(1, max_retries + 1):
        try:
            # The actual request: key goes in the header, filters in params.
            resp = requests.get(
                f"{BASE_URL}{path}",                 # full URL
                headers={"X-API-Key": API_KEY},      # authenticate the request
                params=params or {},                 # query-string filters
                timeout=60,                           # give up on one call after 60s
            )
        except requests.exceptions.RequestException as exc:
            # A low-level network error (DNS, connection dropped, etc.).
            if attempt == max_retries:               # out of tries -> re-raise
                raise
            wait = 2 ** attempt                      # exponential backoff: 2,4,8s
            print(f"    network error ({exc}); retry {attempt}/{max_retries} in {wait}s")
            time.sleep(wait)                         # wait, then loop to retry
            continue

        # Got a response, but it may be a temporary server-side error.
        if resp.status_code in transient and attempt < max_retries:
            wait = 2 ** attempt                      # same growing backoff
            print(f"    HTTP {resp.status_code}; retry {attempt}/{max_retries} in {wait}s")
            time.sleep(wait)
            continue

        resp.raise_for_status()                      # raise on any other 4xx/5xx
        return resp.json()                           # success -> parsed JSON


def find_country_id(iso_code):
    """Look up OpenAQ's internal numeric id for a 2-letter country code."""
    data = api_get("/countries", {"limit": 200})     # list all countries
    for c in data["results"]:                        # scan for our ISO code
        if c["code"] == iso_code:
            return c["id"], c["name"]                # return (id, display name)
    sys.exit(f"Could not find country with ISO code {iso_code}")


def find_pm25_location(country_id):
    """Pick the best PM2.5 station: currently active + longest history.

    LSTMs need long, continuous series to learn seasonality, so among
    stations that are still reporting we choose the one whose data spans
    the widest date range (datetimeFirst -> datetimeLast).
    """
    # Ask for every location in this country that reports PM2.5.
    data = api_get(
        "/locations",
        {"countries_id": country_id, "parameters_id": PM25_PARAMETER_ID, "limit": 1000},
    )
    results = data["results"]
    if not results:                                  # nothing to work with
        sys.exit("No PM2.5-reporting locations found for this country.")

    def utc_of(loc, field):
        """Safely read loc[field]['utc']; field may be missing or None."""
        block = loc.get(field) or {}                 # guard against a null field
        return block.get("utc") or ""                # "" sorts before any real date

    # Keep only stations whose most recent reading is in 2026 (i.e. still live).
    active = [loc for loc in results if utc_of(loc, "datetimeLast") >= "2026-"]
    candidates = active or results                   # fall back to all if none active

    # Sort by first-ever reading (oldest first); [0] = the longest history.
    candidates.sort(key=lambda l: utc_of(l, "datetimeFirst"))
    best = candidates[0]
    print(
        f"Chose station from {len(candidates)} active candidate(s); "
        f"history {utc_of(best, 'datetimeFirst')[:10]} -> "
        f"{utc_of(best, 'datetimeLast')[:10]}"
    )
    return best


def find_pm25_sensor(location):
    """Return the PM2.5 sensor object inside a location's sensor list."""
    for sensor in location["sensors"]:               # a station has many sensors
        if sensor["parameter"]["id"] == PM25_PARAMETER_ID:  # keep the PM2.5 one
            return sensor
    sys.exit("Location has no PM2.5 sensor (unexpected).")


def _fetch_window(sensor_id, win_from, win_to):
    """Fetch one date window, paging shallowly (a window holds few pages)."""
    rows = []                                        # collect this window's rows
    page = 1                                         # OpenAQ paging is 1-based
    while True:
        data = api_get(
            f"/sensors/{sensor_id}/hours",           # hourly-aggregate endpoint
            {
                # v3 /hours uses datetime_from/datetime_to; the date_* names
                # are silently ignored (returns the full series unfiltered).
                "datetime_from": win_from.isoformat(),
                "datetime_to": win_to.isoformat(),
                "limit": 1000,                       # max rows per page
                "page": page,                        # which page to fetch
            },
        )
        results = data["results"]
        if not results:                              # empty page -> done
            break
        rows.extend(results)                         # add this page's rows
        if len(results) < 1000:                      # partial page = last page
            break
        page += 1                                    # otherwise fetch the next page
        time.sleep(0.2)                              # be polite to the rate limit
    return rows


def fetch_hourly_measurements(sensor_id, date_from, date_to, window_days=60):
    """Fetch hourly aggregates by walking fixed date windows.

    OpenAQ v3 degrades on deep page offsets (times out past ~18k rows), so we
    slice the full range into ~60-day windows. Each window has only ~1-2 pages,
    which keeps every request cheap and lets api_get retry transient failures.
    """
    all_rows = []                                    # rows across all windows
    win_from = date_from                             # start of the first window
    while win_from < date_to:                        # walk forward until "now"
        # End of this window = 60 days later, but never past the overall end.
        win_to = min(win_from + timedelta(days=window_days), date_to)
        rows = _fetch_window(sensor_id, win_from, win_to)  # download this slice
        all_rows.extend(rows)
        print(
            f"  {win_from.date()} -> {win_to.date()}: +{len(rows)} rows "
            f"(total {len(all_rows)})"
        )
        win_from = win_to                            # next window starts where this ended
    return all_rows


def main():
    # Step 1: resolve country -> station -> PM2.5 sensor.
    country_id, country_name = find_country_id(COUNTRY_ISO)
    print(f"Country: {country_name} (id={country_id})")

    location = find_pm25_location(country_id)
    print(f"Location: {location['name']} (id={location['id']})")

    sensor = find_pm25_sensor(location)
    print(f"Sensor: {sensor['name']} (id={sensor['id']})")

    # Step 2: build the [date_from, date_to] range to download (in UTC).
    date_to = datetime.now(timezone.utc)                     # right now
    date_from = date_to - timedelta(days=30 * MONTHS_OF_HISTORY)  # ~24 months ago

    print(f"Fetching hourly PM2.5 from {date_from.date()} to {date_to.date()} ...")
    rows = fetch_hourly_measurements(sensor["id"], date_from, date_to)

    if not rows:                                             # nothing came back
        sys.exit("No measurements returned. Try a different location or date range.")

    # Step 3: keep only the two fields we care about from each API record.
    records = []
    for r in rows:
        records.append(
            {
                # Each hourly record has a period; use its start timestamp.
                "datetime": r["period"]["datetimeFrom"]["utc"],
                "pm25": r["value"],                          # the measured value
            }
        )

    # Step 4: turn the records into a clean, time-ordered table.
    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["datetime"])          # parse text -> datetime
    df = df.sort_values("datetime").drop_duplicates(subset="datetime")  # order + dedupe

    # Step 5: write the CSV that every downstream script reads.
    os.makedirs("data", exist_ok=True)                       # ensure data/ exists
    out_path = "data/pm25_raw.csv"
    df.to_csv(out_path, index=False)                         # no pandas index column
    print(f"Saved {len(df)} rows to {out_path}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Station: {location['name']}, {country_name}")


if __name__ == "__main__":
    main()

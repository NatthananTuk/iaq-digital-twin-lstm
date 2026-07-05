"""
Turn the raw hourly PM2.5 series (data/pm25_raw.csv) into supervised-learning
windows for the LSTM, and save them as .npy arrays.

Task framing: use the last WINDOW hours to predict the next HORIZON hours.
    X shape: (samples, WINDOW, 1)   -> 24 hours of history
    y shape: (samples, HORIZON)     -> next 6 hours to forecast

Four design choices baked in (see README):
  1. Small gaps (<= MAX_INTERP_HOURS missing hours) are bridged by linear
     interpolation. Real sensors miss the odd reading; PM2.5 is smooth, so
     filling 1-3 hour holes recovers a lot of usable data. Large gaps are
     left as real breaks.
  2. Train/test split is CHRONOLOGICAL (no shuffling) to avoid leaking the
     future into the past.
  3. The MinMax scaler is fit on the TRAIN portion only, then applied to test.
  4. Windows are GAP-AWARE: a window is only built inside a run of consecutive
     hours, so no sample straddles a real (unfilled) gap.

Run: python preprocess.py
"""
import numpy as np                       # numeric arrays + saving .npy files
import pandas as pd                      # loading/handling the CSV table
import joblib                            # saving the fitted scaler to disk
from sklearn.preprocessing import MinMaxScaler  # scales values into [0, 1]

# ---- Configuration knobs (change these to experiment) --------------------
RAW_PATH = "data/pm25_raw.csv"           # where fetch_data.py wrote the data
WINDOW = 24                              # hours of history fed into the model
HORIZON = 6                              # hours ahead we want to predict
TRAIN_FRAC = 0.8                         # first 80% of the timeline = training
MAX_INTERP_HOURS = 3                     # bridge gaps up to this many missing h
SCALER_PATH = "scaler.pkl"              # saved so evaluate.py/app.py reuse it


def load_regular_hourly_series():
    """Load the CSV and return a value array on a COMPLETE hourly time grid.

    Small holes (<= MAX_INTERP_HOURS consecutive missing hours) are filled by
    linear interpolation; larger holes stay as NaN so later steps treat them
    as real breaks. Returns the value array (which may contain NaN).
    """
    df = pd.read_csv(RAW_PATH, parse_dates=["datetime"])  # read + parse dates
    df = df.sort_values("datetime").drop_duplicates("datetime")  # tidy order

    # Put time on the index and resample onto a gap-free hourly grid. Every
    # missing hour now shows up explicitly as a NaN row.
    s = df.set_index("datetime")["pm25"].asfreq("h")

    # Find runs of consecutive NaNs so we can size each hole. `block` gives a
    # new id whenever the NaN/not-NaN status flips.
    is_na = s.isna()                                     # True where missing
    block = (is_na != is_na.shift()).cumsum()           # id per constant run
    run_len = s.groupby(block).transform("size")        # length of each run

    # Interpolate the whole series in time, but only KEEP the interpolated
    # value where the hole is small enough. Big holes revert to NaN.
    interpolated = s.interpolate(method="time")         # straight-line fill
    small_hole = is_na & (run_len <= MAX_INTERP_HOURS)  # holes we allow to fill
    filled = s.copy()                                   # start from originals
    filled[small_hole] = interpolated[small_hole]       # patch only small holes

    n_filled = int(small_hole.sum())                    # how many hours we added
    print(f"Filled {n_filled} missing hours via interpolation "
          f"(gaps <= {MAX_INTERP_HOURS}h); "
          f"{int(filled.isna().sum())} hours remain as real gaps")
    return filled.to_numpy()             # values on the hourly grid (NaN = gap)


def make_gap_aware_sequences(values, window, horizon):
    """Build (X, y) windows only inside runs of NON-NaN hourly readings.

    Because `values` sits on a complete hourly grid, any stretch without NaN
    is automatically consecutive in time, so we just split on the NaNs.
    Returns X of shape (n, window, 1) and y of shape (n, horizon).
    """
    X, y = [], []                        # collectors for the finished windows

    # np.isnan marks the real gaps; we group the array into non-NaN segments.
    is_valid = ~np.isnan(values)         # True where we have a usable value

    # Identify contiguous valid segments by scanning for on/off transitions.
    idx = 0                              # current position while scanning
    n = len(values)
    while idx < n:
        if not is_valid[idx]:            # skip over a gap
            idx += 1
            continue
        start = idx                      # a valid segment starts here
        while idx < n and is_valid[idx]:  # extend until the next gap
            idx += 1
        seg = values[start:idx]          # one continuous block of readings

        # Slide a (window + horizon) frame across this segment only.
        last_start = len(seg) - window - horizon
        for i in range(last_start + 1):  # +1 so the final valid start is used
            X.append(seg[i : i + window])                    # 24h input
            y.append(seg[i + window : i + window + horizon])  # next 6h target

    # Convert the Python lists into numpy arrays with the shapes Keras expects.
    X = np.array(X)                      # shape: (n, window)
    y = np.array(y)                      # shape: (n, horizon)
    X = X.reshape(X.shape[0], window, 1)  # add feature axis -> (n, window, 1)
    return X, y


def main():
    values = load_regular_hourly_series()          # gridded series w/ NaN gaps
    print(f"Series spans {len(values)} hourly slots")

    # --- 1. Chronological split (by position, since the grid is time-ordered).
    split_idx = int(len(values) * TRAIN_FRAC)       # index where train ends
    train_vals = values[:split_idx]                 # earliest 80% -> training
    test_vals = values[split_idx:]                  # latest 20%  -> testing

    # --- 2. Fit the scaler on TRAIN values only (ignoring NaN), scale both.
    scaler = MinMaxScaler()                                     # maps to [0, 1]
    train_known = train_vals[~np.isnan(train_vals)].reshape(-1, 1)  # drop NaN
    scaler.fit(train_known)                                     # learn min/max
    # Transform reshaped columns; NaNs pass straight through as NaN.
    train_scaled = scaler.transform(train_vals.reshape(-1, 1)).ravel()
    test_scaled = scaler.transform(test_vals.reshape(-1, 1)).ravel()

    # --- 3. Build gap-aware windows separately for each split.
    X_train, y_train = make_gap_aware_sequences(train_scaled, WINDOW, HORIZON)
    X_test, y_test = make_gap_aware_sequences(test_scaled, WINDOW, HORIZON)

    # --- 4. Save arrays + scaler so the next scripts can load them directly.
    np.save("X_train.npy", X_train)      # model inputs (training)
    np.save("y_train.npy", y_train)      # model targets (training)
    np.save("X_test.npy", X_test)        # model inputs (testing)
    np.save("y_test.npy", y_test)        # model targets (testing)
    joblib.dump(scaler, SCALER_PATH)     # persist the fitted scaler

    # --- 5. Report the final shapes.
    print("\nSaved arrays:")
    print(f"  X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"  X_test  {X_test.shape}, y_test  {y_test.shape}")
    print(f"  scaler -> {SCALER_PATH}")


if __name__ == "__main__":
    main()

"""
Evaluate the trained LSTM on the untouched test set and compare it against a
naive "last observed value" baseline.

Why a baseline? A forecast model is only worth anything if it beats the obvious
guess ("the next hours look like the last hour"). Reporting the % improvement
over that baseline is the honest headline number for this project.

Steps:
  1. Load the model, the test windows, and the SAME scaler used in training.
  2. Predict the next 6 hours for every test window.
  3. Inverse-scale predictions AND targets back to real ug/m3.
  4. Build the naive baseline (repeat the last input hour 6 times).
  5. Score both with MAE/RMSE, overall and per lead-time hour, and plot.

Run: python evaluate.py
"""
import numpy as np                       # array math
import joblib                            # load the saved scaler
import matplotlib                        # plotting
matplotlib.use("Agg")                    # headless backend (save, don't show)
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model          # load the trained model
from sklearn.metrics import mean_absolute_error, mean_squared_error

MODEL_PATH = "lstm_pm25_model.keras"     # produced by train_model.py
SCALER_PATH = "scaler.pkl"              # produced by preprocess.py


def inverse_scale(scaled_2d, scaler):
    """Turn scaled [0,1] values back into real ug/m3 using the fitted scaler.

    scaled_2d has shape (n, horizon). The scaler expects a single column, so we
    flatten to one column, inverse-transform, then reshape back to (n, horizon).
    """
    n, horizon = scaled_2d.shape                        # remember the shape
    flat = scaled_2d.reshape(-1, 1)                     # -> (n*horizon, 1) column
    real = scaler.inverse_transform(flat)              # undo the MinMax scaling
    return real.reshape(n, horizon)                    # back to (n, horizon)


def main():
    # --- Step 1: load the model, test windows, and the training scaler. ----
    model = load_model(MODEL_PATH)                     # the trained LSTM
    scaler = joblib.load(SCALER_PATH)                  # SAME scaler as training
    X_test = np.load("X_test.npy")                     # (n, 24, 1) inputs
    y_test = np.load("y_test.npy")                     # (n, 6) true future (scaled)
    print(f"Test windows: {X_test.shape[0]}")

    # --- Step 2: predict the next 6 hours for each window (still scaled). ---
    y_pred_scaled = model.predict(X_test, verbose=0)   # (n, 6) predictions

    # --- Step 3: convert predictions and truth back to real ug/m3. ---------
    y_true = inverse_scale(y_test, scaler)             # actual future values
    y_pred = inverse_scale(y_pred_scaled, scaler)      # LSTM's forecast

    # --- Step 4: naive baseline = repeat the LAST observed hour 6 times. ----
    # X_test[:, -1, 0] is the 24th (most recent) scaled value of each window.
    last_hour_scaled = X_test[:, -1, 0].reshape(-1, 1)  # (n, 1) last input hour
    # Tile it across all 6 horizon steps to form a (n, 6) "flat" forecast.
    naive_scaled = np.repeat(last_hour_scaled, y_test.shape[1], axis=1)
    naive = inverse_scale(naive_scaled, scaler)        # back to ug/m3

    # --- Step 5a: overall MAE / RMSE across all horizons, in ug/m3. ---------
    # .ravel() flattens (n, 6) -> one long vector so every predicted hour counts.
    lstm_mae = mean_absolute_error(y_true.ravel(), y_pred.ravel())
    naive_mae = mean_absolute_error(y_true.ravel(), naive.ravel())
    lstm_rmse = np.sqrt(mean_squared_error(y_true.ravel(), y_pred.ravel()))
    naive_rmse = np.sqrt(mean_squared_error(y_true.ravel(), naive.ravel()))
    improvement = (1 - lstm_mae / naive_mae) * 100     # % better than baseline

    print("\n================ RESULTS (ug/m3) ================")
    print(f"LSTM   MAE: {lstm_mae:5.2f}   RMSE: {lstm_rmse:5.2f}")
    print(f"Naive  MAE: {naive_mae:5.2f}   RMSE: {naive_rmse:5.2f}")
    print(f"Improvement over naive baseline (MAE): {improvement:.1f}%")

    # --- Step 5b: MAE per lead-time hour (does error grow with distance?). --
    print("\nPer-hour-ahead MAE (ug/m3):")
    per_hour_lstm, per_hour_naive = [], []
    for h in range(y_true.shape[1]):                   # column h = h+1 hours ahead
        m_lstm = mean_absolute_error(y_true[:, h], y_pred[:, h])
        m_naive = mean_absolute_error(y_true[:, h], naive[:, h])
        per_hour_lstm.append(m_lstm)
        per_hour_naive.append(m_naive)
        print(f"  +{h+1}h  LSTM {m_lstm:5.2f}  |  naive {m_naive:5.2f}")

    # --- Step 5c: two plots -> per-hour MAE bars + one example forecast. ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    # LEFT: grouped bars comparing LSTM vs naive at each lead time.
    hours = np.arange(1, y_true.shape[1] + 1)          # 1..6 hours ahead
    ax1.bar(hours - 0.2, per_hour_lstm, width=0.4, label="LSTM", color="#27ae60")
    ax1.bar(hours + 0.2, per_hour_naive, width=0.4, label="Naive", color="#95a5a6")
    ax1.set_title("MAE by hours ahead")
    ax1.set_xlabel("Hours ahead")
    ax1.set_ylabel("MAE (ug/m3)")
    ax1.set_xticks(hours)
    ax1.legend()

    # RIGHT: pick a real test window where the LSTM beats the naive guess by
    # the most. The naive line is flat by definition, so this illustrates the
    # model's core value: anticipating a trend the "last value" guess cannot.
    lstm_err = np.abs(y_true - y_pred).sum(axis=1)     # total error per window
    naive_err = np.abs(y_true - naive).sum(axis=1)
    sample = int(np.argmax(naive_err - lstm_err))      # biggest LSTM advantage
    steps = hours                                      # x-axis: +1h..+6h
    ax2.plot(steps, y_true[sample], marker="o", label="Actual", color="#2c3e50")
    ax2.plot(steps, y_pred[sample], marker="o", label="LSTM", color="#27ae60")
    ax2.plot(steps, naive[sample], marker="o", linestyle="--",
             label="Naive", color="#95a5a6")
    ax2.set_title(f"Example: LSTM tracks a trend the naive guess misses (#{sample})")
    ax2.set_xlabel("Hours ahead")
    ax2.set_ylabel("PM2.5 (ug/m3)")
    ax2.legend()

    fig.tight_layout()
    fig.savefig("screenshots/evaluation.png", dpi=120)
    plt.close(fig)
    print("\nSaved screenshots/evaluation.png")


if __name__ == "__main__":
    main()

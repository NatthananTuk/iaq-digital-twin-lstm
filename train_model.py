"""
Build and train the LSTM that forecasts the next 6 hours of PM2.5 from the
previous 24 hours, using the windows created by preprocess.py.

Architecture:  Input(24, 1) -> LSTM(64) -> Dense(32, relu) -> Dense(6)

Key training choices (things a reviewer will ask about):
  - Validation set is the LAST 15% of the training windows (chronological),
    NOT the test set. The test set stays untouched until evaluate.py, so we
    never tune the model on data we later report scores on.
  - EarlyStopping watches validation loss and restores the best weights, so
    the model stops before it starts memorising / overfitting.
  - Random seeds are fixed so the run is reproducible.

Run: python train_model.py   (writes lstm_pm25_model.keras)
"""
import numpy as np                       # load the .npy arrays
import matplotlib                        # plot the training curve
matplotlib.use("Agg")                    # headless backend (save, don't show)
import matplotlib.pyplot as plt
import tensorflow as tf                  # deep-learning framework
from tensorflow.keras.models import Sequential          # simple layer stack
from tensorflow.keras.layers import LSTM, Dense, Input  # the layers we use
from tensorflow.keras.callbacks import EarlyStopping    # stop when val stops improving

# ---- Reproducibility -----------------------------------------------------
# Fix the random seeds so weights initialise the same way every run; this
# makes results repeatable (important when reporting numbers).
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---- Hyperparameters (the knobs we can tune) -----------------------------
LSTM_UNITS = 64          # size of the LSTM's memory/summary vector
DENSE_UNITS = 32         # width of the hidden layer after the LSTM
EPOCHS = 100             # max passes over the data (EarlyStopping cuts it short)
BATCH_SIZE = 32          # how many windows per gradient update
VAL_FRACTION = 0.15      # last 15% of training windows used for validation
PATIENCE = 8             # stop if val loss doesn't improve for this many epochs
MODEL_PATH = "lstm_pm25_model.keras"     # where the trained model is saved


def load_training_data():
    """Load the training windows produced by preprocess.py."""
    X_train = np.load("X_train.npy")     # shape (n, 24, 1): 24h of history
    y_train = np.load("y_train.npy")     # shape (n, 6):    next 6h to predict
    print(f"X_train {X_train.shape}, y_train {y_train.shape}")
    return X_train, y_train


def build_model(window, horizon):
    """Assemble the LSTM network.

    window  = number of input time steps (24)
    horizon = number of outputs / future hours (6)
    """
    model = Sequential([                 # a plain feed-forward stack of layers
        # Declare the input shape: `window` time steps, each with 1 feature.
        Input(shape=(window, 1)),
        # LSTM reads the 24-step sequence and returns ONE 64-dim vector
        # (its final hidden state = a summary of the last 24 hours).
        LSTM(LSTM_UNITS),
        # A small dense layer adds non-linear mixing of those 64 features.
        # ReLU keeps positive signals and zeroes negatives (cheap + effective).
        Dense(DENSE_UNITS, activation="relu"),
        # Final layer outputs `horizon` numbers = the 6 hourly forecasts.
        # No activation: this is regression, values are continuous.
        Dense(horizon),
    ])
    # Compile = choose how to train:
    #   loss "mse"      : penalises big errors hard (good for spiky pollution)
    #   optimizer "adam": adaptive learning rate, a strong default
    #   metric "mae"    : average absolute error, easy to read in ug/m3-scale
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.summary()                      # print the layer/parameter table
    return model


def plot_history(history):
    """Save train vs. validation loss over epochs (shows over/under-fitting)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history["loss"], label="train loss (MSE)")       # training
    ax.plot(history.history["val_loss"], label="validation loss (MSE)")  # val
    ax.set_title("Training curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE, scaled)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("screenshots/training_curve.png", dpi=120)
    plt.close(fig)
    print("Saved screenshots/training_curve.png")


def main():
    X_train, y_train = load_training_data()   # get the windows
    window = X_train.shape[1]                  # 24 time steps
    horizon = y_train.shape[1]                 # 6 outputs

    model = build_model(window, horizon)       # create the network

    # EarlyStopping: monitor validation loss; if it doesn't improve for
    # PATIENCE epochs, stop and roll back to the best-scoring weights.
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
    )

    # Train. `validation_split=0.15` holds out the LAST 15% of the windows
    # (Keras takes them from the end BEFORE shuffling), so validation is the
    # most-recent training data and the test set is never touched here.
    history = model.fit(
        X_train, y_train,
        validation_split=VAL_FRACTION,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=2,                             # one line per epoch
    )

    model.save(MODEL_PATH)                      # persist the trained model
    print(f"\nSaved model -> {MODEL_PATH}")

    plot_history(history)                       # save the loss curve

    # Report where EarlyStopping settled and the best validation error.
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    best_val_mae = min(history.history["val_mae"])
    print(f"Best epoch: {best_epoch}  |  best val MAE (scaled): {best_val_mae:.4f}")


if __name__ == "__main__":
    main()

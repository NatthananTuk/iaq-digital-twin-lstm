# 🌫️ IAQ Digital Twin — PM2.5 Forecast (LSTM)

Forecasting short-horizon PM2.5 air quality using an LSTM, wrapped in a simple
dashboard. Built as a portfolio piece for an IIoT / predictive Digital Twin
research direction — the pipeline is deliberately **data-source-agnostic** so
the same code can run on a live sensor feed later.

> **Status:** 🚧 Work in progress. Data ingestion is done; modelling and
> dashboard are next. See the roadmap below.

## Data source

Real hourly PM2.5 measurements pulled live from the **[OpenAQ v3 API](https://docs.openaq.org/)**
rather than a static CSV — using a real API keeps the "live IIoT data" narrative
honest and lets reviewers verify the data themselves.

- **Station:** Pluakdaeng District Health Office, Rayong, **Thailand** (OpenAQ location `717`)
- **Signal:** PM2.5, hourly aggregates (µg/m³)
- **Window pulled:** ~24 months (Jul 2024 → Jul 2026), 12,477 hourly rows
- **Why this station:** `fetch_data.py` auto-selects, among all PM2.5 stations in
  the country, the one that is still actively reporting *and* has the longest
  history — LSTMs need long, continuous series to learn seasonality.

### Data-quality notes (from initial EDA)

| Metric | Value |
|---|---|
| Rows (hourly) | 12,477 |
| Mean / median PM2.5 | 22.2 / 18.2 µg/m³ |
| Max PM2.5 | 92.4 µg/m³ |
| Null / non-positive values | 0 / 0 |

⚠️ The series is **not perfectly continuous** — there are 783 time gaps, the
largest ~55 days. Preprocessing therefore builds training windows *within*
continuous segments only, so no input sequence straddles a large gap.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# Get a free key at https://explore.openaq.org/register
cp .env.example .env          # then paste your key into .env
python fetch_data.py          # writes data/pm25_raw.csv
```

`.env` holds your `OPENAQ_API_KEY` and is git-ignored — never commit it.

## Roadmap

- [x] `fetch_data.py` — pull real hourly PM2.5 from OpenAQ v3
- [ ] `preprocess.py` — gap-aware windowing (24h lookback → 6h horizon) + scaling
- [ ] `train_model.py` — LSTM (Keras/TensorFlow)
- [ ] `evaluate.py` — LSTM vs. naive "last value" baseline (MAE, % improvement)
- [ ] `app.py` — Streamlit dashboard (history + live 6h forecast)

## Project structure

```
IAQ_LSTM/
├── fetch_data.py        # OpenAQ v3 ingestion → data/pm25_raw.csv
├── requirements.txt
├── .env.example         # template; copy to .env and add your key
├── data/
│   └── pm25_raw.csv
└── screenshots/
```

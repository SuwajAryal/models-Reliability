# models-Reliability

This repository now includes a small baseline workflow to train and compare multiple models for **suspended sediment load** prediction.

## What is included

- `MeanBaseline`: predicts the mean training suspended load.
- `LinearRegressionGD`: gradient-descent linear regression.
- `KNNRegressor(k=3)`: nearest-neighbor regression.

Each model is evaluated with:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R2 score

## Run model comparison

```bash
python suspended_sediment_models.py --data /path/to/data.csv --target suspended_load
```

Optional: choose explicit feature columns.

```bash
python suspended_sediment_models.py \
  --data /path/to/data.csv \
  --target suspended_load \
  --features discharge rainfall temperature
```

The script prints a CSV-style leaderboard sorted by RMSE (best first).

## Run tests

```bash
python -m unittest discover -s tests -q
```

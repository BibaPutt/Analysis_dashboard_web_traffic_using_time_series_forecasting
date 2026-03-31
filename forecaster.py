"""
forecaster.py
Time-series forecasting engine with Prophet, ARIMA, seasonal decomposition,
and evaluation metrics.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------------------------------------------
# Seasonal decomposition (statsmodels)
# ---------------------------------------------------------------------------
def seasonal_decompose(series, period=7):
    """
    Decompose a time series into trend, seasonal, and residual components.
    Returns a dict with 'trend', 'seasonal', 'residual' Series.
    """
    from statsmodels.tsa.seasonal import seasonal_decompose as sm_decompose
    if len(series) < 2 * period:
        period = max(2, len(series) // 3)
    result = sm_decompose(series, model="additive", period=period, extrapolate_trend="freq")
    return {
        "trend": result.trend,
        "seasonal": result.seasonal,
        "residual": result.resid,
        "observed": result.observed,
    }


# ---------------------------------------------------------------------------
# Prophet forecasting
# ---------------------------------------------------------------------------
def forecast_prophet(df, date_col, value_col, periods=30, yearly=True, weekly=True):
    """
    Train a Prophet model and return forecast dataframe + metrics.
    df must have date_col and value_col.
    Returns (forecast_df, metrics_dict, model).
    """
    from prophet import Prophet

    prophet_df = df[[date_col, value_col]].rename(columns={date_col: "ds", value_col: "y"}).dropna()
    prophet_df = prophet_df.sort_values("ds")

    # Train / test split — last 20 % for evaluation
    split_idx = int(len(prophet_df) * 0.8)
    train = prophet_df.iloc[:split_idx]
    test = prophet_df.iloc[split_idx:]

    model = Prophet(
        yearly_seasonality=yearly,
        weekly_seasonality=weekly,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    model.fit(train)

    # Evaluate on test set
    future_test = model.make_future_dataframe(periods=len(test))
    pred_test = model.predict(future_test)
    pred_values = pred_test.iloc[split_idx:]["yhat"].values[: len(test)]
    actual_values = test["y"].values[: len(pred_values)]

    metrics = _compute_metrics(actual_values, pred_values)

    # Retrain on full data for final forecast
    model_full = Prophet(
        yearly_seasonality=yearly,
        weekly_seasonality=weekly,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    model_full.fit(prophet_df)
    future = model_full.make_future_dataframe(periods=periods)
    forecast = model_full.predict(future)

    return forecast, metrics, model_full


# ---------------------------------------------------------------------------
# ARIMA / SARIMA forecasting
# ---------------------------------------------------------------------------
def forecast_arima(df, date_col, value_col, periods=30, seasonal_order=(1, 1, 1, 7)):
    """
    Train a SARIMAX model and return forecast dataframe + metrics.
    Returns (forecast_df, metrics_dict).
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    series = df[[date_col, value_col]].dropna().sort_values(date_col)
    series = series.set_index(date_col)
    series = series.asfreq("D")
    series[value_col] = series[value_col].interpolate()

    # Train / test split
    split_idx = int(len(series) * 0.8)
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]

    try:
        model = SARIMAX(
            train[value_col],
            order=(1, 1, 1),
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        results = model.fit(disp=False, maxiter=200)
    except Exception:
        # Fallback to simpler order
        model = SARIMAX(
            train[value_col],
            order=(1, 1, 0),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        results = model.fit(disp=False, maxiter=200)

    # Evaluate on test set
    pred_test = results.forecast(steps=len(test))
    metrics = _compute_metrics(test[value_col].values, pred_test.values)

    # Retrain on full data
    try:
        model_full = SARIMAX(
            series[value_col],
            order=(1, 1, 1),
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        results_full = model_full.fit(disp=False, maxiter=200)
    except Exception:
        model_full = SARIMAX(
            series[value_col],
            order=(1, 1, 0),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        results_full = model_full.fit(disp=False, maxiter=200)

    forecast_vals = results_full.forecast(steps=periods)
    last_date = series.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq="D")

    forecast_df = pd.DataFrame({
        "ds": forecast_dates,
        "yhat": forecast_vals.values,
    })

    # Build full historical + forecast df
    hist_df = pd.DataFrame({
        "ds": series.index,
        "yhat": results_full.fittedvalues.values,
        "actual": series[value_col].values,
    })

    return forecast_df, hist_df, metrics


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _compute_metrics(actual, predicted):
    """Compute MAE, RMSE, MAPE."""
    actual = np.array(actual)
    predicted = np.array(predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    # MAPE — guard against zeros
    mask = actual != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    else:
        mape = 0.0
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE": round(mape, 2)}

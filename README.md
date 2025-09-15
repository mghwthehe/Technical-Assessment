---
editor_options: 
  markdown: 
    wrap: 72
---

# Time Series Sales Forecasting (ARIMA)

Objective: Build a predictive model to forecast sales for the next 3
months. Requirements: Apply at least one time series forecasting
algorithm (e.g., ARIMA, Prophet, LSTM); Provide evaluation metrics such
as MAE, RMSE, or MAPE; Document your feature engineering choices and
explain the rationale behind your modelling approach.

### Feature Engineering Choices:

**1. Date Parsing**

I parsed the date column as a proper datetime type, which makes it
easier to resample, split, and align time series data. This ensures
chronological order is preserved and prevents data leakage from future
dates into training.

**2. Series Separation (by store and item)**

The sales data is given for 10 stores and 50 items (500 unique series).
Each (store, item) pair was treated as an independent time series
because sales patterns differ across items and locations.

**3. Train/Validation Split**

I used a time-based split where the last 3 months of 2017 were held out
as validation (cutoff = "2017-10-01"). This reflects the requirement of
forecast sales for the next 3 months (January - March 2018), ensuring
evaluation conditions are realistic.

**forecast = forecast_arima(store, item, order = (p,d,q) = (2,1,2))**

**4. p = AutoRegressive (Lagged Dependence)**

p = 2 allows the model to capture short-term dependencies (yesterday’s
and the day before yesterday’s sales). This is important in retail where
demand on one day often depends on the previous few days.

**5. d = Differencing (Stationarity)**

ARIMA requires stationary data. I applied d = 1 (first-order
differencing) to remove linear trends as it is a simple and robust
default since retail sales usually show upward/downward drifts over
time.

**6. q = Moving Average Terms (Shock/Error Smoothing)**

q = 2 allows the model to adjust forecasts based on recent shocks (large
deviations). This helps smooth out noise, which is common in daily sales
data.

**7. Evaluation Metrics**

The predictive model is evaluated using MAE (scale-independent,
interpretable as average error in units sold) and RMSE (penalises large
deviations more). These metrics can provide complementary views of model
performance.

### Modelling Approach Rationale:

The reason I choose ARIMA time series forecasting algorithm is because
it is a strong classical baseline for time series forecasting,
especially suitable when we only have historical sales data. It
explicitly models trends, which are key drivers in daily sales. Then, to
ensure consistency across 500 (10 stores x 50 items) series, I used a
fixed ARIMA order (2,1,2) for all series instead of tuning each one.
This is efficient and provides a fair benchmark across all items.

### Technologies and tools used.

Python 3.11+ Libraries: pandas, numpy, statsmodels (SARIMAX for ARIMA),
scikit-learn (MAE/RMSE)

### Steps to reproduce the results:

Firstly, I fit on training data (till 2017-09-30) and validate on the
last 3 months of 2017 to check the MAE and RMSE. After that, I retrain
on the full data (2013–2017) to capture the maximum history. Lastly, I
forecast horizon (Jan – Mar 2018) to generate the output submission with
the 2 columns (id, sales).

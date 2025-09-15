#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[18]:


train = pd.read_csv("/Users/meganchow/Downloads/train.csv", parse_dates = ["date"])
test = pd.read_csv("/Users/meganchow/Downloads/test.csv", parse_dates = ["date"])


# In[19]:


# Evaluate RMSE
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return mae, rmse

# Forecast function for one series
def forecast_arima(store, item, order=(1,1,1)):
    series = train[(train["store"]==store) & (train["item"]==item)].set_index("date")["sales"]
    
    # Train/Validation split
    cutoff = "2017-10-01"   # last 3 months as validation
    train_series = series.loc[:cutoff]
    valid_series = series.loc[cutoff:]
    
    # Fit ARIMA
    model = SARIMAX(train_series, order=order, seasonal_order=(0,0,0,0), enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    
    # Validation forecast
    valid_pred = model_fit.get_prediction(start=valid_series.index[0], end=valid_series.index[-1]).predicted_mean
    
    mae, rmse = evaluate(valid_series, valid_pred)
    print(f"Store {store}, Item {item} → MAE={mae:.2f}, RMSE={rmse:.2f}")
    
    # Retrain on full history for final test forecast
    final_model = SARIMAX(series, order=order, seasonal_order=(0,0,0,0), enforce_stationarity=False, enforce_invertibility=False)
    final_fit = final_model.fit(disp=False)
    
    # Forecast horizon = test set for this store+item
    horizon = test[(test["store"]==store) & (test["item"]==item)].set_index("date").index
    forecast = final_fit.get_forecast(steps=len(horizon)).predicted_mean
    return forecast


# In[22]:


# Run the ARIMA (store, item)
forecast = forecast_arima(1, 1, order = (2,1,2))

# Merge with test set for store 1, item 1
test_slice = test[(test["store"]==1) & (test["item"]==1)].copy()
test_slice["sales"] = forecast.values
print(test_slice.head())
print(test_slice.tail())


# In[21]:


submission = test.copy()
submission["sales"] = 0.0  

for store in test["store"].unique():
    for item in test["item"].unique():
        forecast = forecast_arima(store, item, order=(2,1,2))
        mask = (submission["store"]==store) & (submission["item"]==item)
        submission.loc[mask, "sales"] = forecast.values

# Produce the submission.csv
final_submission = submission[["id","sales"]]
final_submission.to_csv("submission.csv", index=False)
print(final_submission)


# Feature Engineering Choices:
# 
# 1. Date Parsing
# 
# I parsed the date column as a proper datetime type, which makes it easier to resample, split, and align time series data. This ensures chronological order is preserved and prevents data leakage from future dates into training.
# 
# 2. Series Separation (by store and item)
# 
# The sales data is given for 10 stores and 50 items (500 unique series). Each (store, item) pair was treated as an independent time series because sales patterns differ across items and locations.
# 
# 
# 3. Train/Validation Split
# 
# I used a time-based split where the last 3 months of 2017 were held out as validation (cutoff = "2017-10-01"). This reflects the requirement of forecast sales for the next 3 months (January - March 2018), ensuring evaluation conditions are realistic.
# 
# 
# forecast = forecast_arima(store, item, order=(2,1,2))
# order=(p,d,q)
# 4. p = AutoRegressive (Lagged Dependence) 
# 
# p = 2 allows the model to capture short-term dependencies (yesterday’s and the day before yesterday’s sales). This is important in retail where demand on one day often depends on the previous few days.
# 
# 5. d = Differencing (Stationarity)
# 
# ARIMA requires stationary data. I applied d = 1 (first-order differencing) to remove linear trends as it is a simple and robust default since retail sales usually show upward/downward drifts over time.
# 
# 6. q = Moving Average Terms (Shock/Error Smoothing)
# 
# q = 2 allows the model to adjust forecasts based on recent shocks (large deviations). This helps smooth out noise, which is common in daily sales data.
# 
# 7. Evaluation Metrics
# 
# The predictive model is evaluated using MAE (scale-independent, interpretable as average error in units sold) and RMSE (penalises large deviations more). These metrics can provide complementary views of model performance.

# Modelling Approach Rationale:
# 
# The reason I choose ARIMA time series forecasting algorithm is because it is a strong classical baseline for time series forecasting, especially suitable when we only have historical sales data. It explicitly models trends, which are key drivers in daily sales. Then, to ensure consistency across 500 (10 stores x 50 items) series, I used a fixed ARIMA order (2,1,2) for all series instead of tuning each one. This is efficient and provides a fair benchmark across all items.
# 
# Firstly, I fit on training data (till 2017-09-30) and validate on the last 3 months of 2017 to check the MAE and RMSE. After that, I retrain on the full data (2013–2017) to capture the maximum history. Lastly, I forecast horizon (Jan – Mar 2018) to generate the output submission with the 2 columns (id, sales).

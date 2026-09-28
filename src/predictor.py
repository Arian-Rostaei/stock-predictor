import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# --- ۱. گرفتن داده ---
ticker = "AAPL"  # می‌تونی نمادهای دیگه مثل "MSFT" یا "TSLA" رو بذاری
print(f"Downloading data for {ticker}...")
data = yf.download(ticker, period="2y", interval="1d")
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# --- ۲. مهندسی ویژگی (Feature Engineering) ---
# ساخت ویژگی‌های جدید بر اساس قیمت‌های قبلی (Lag) و میانگین متحرک
data['Close_Lag1'] = data['Close'].shift(1)
data['Close_Lag2'] = data['Close'].shift(2)
data['Close_Lag3'] = data['Close'].shift(3)
data['MA_3'] = data['Close'].rolling(window=3).mean()
data['MA_5'] = data['Close'].rolling(window=5).mean()
data['MA_10'] = data['Close'].rolling(window=10).mean()
data.dropna(inplace=True)

# --- ۳. تعریف هدف (Target) ---
# هدف ما پیش‌بینی قیمت بسته شدن روز بعد است
data['Target'] = data['Close'].shift(-1)
data.dropna(inplace=True)

# --- ۴. آماده‌سازی برای مدل ---
features = ['Open', 'High', 'Low', 'Close', 'Volume',
            'Close_Lag1', 'Close_Lag2', 'Close_Lag3',
            'MA_3', 'MA_5', 'MA_10']
X = data[features]
y = data['Target']

# تقسیم داده به آموزش و تست (۸۰٪ آموزش، ۲۰٪ تست)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# --- ۵. آموزش مدل ---
print("Training model...")
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1, max_depth=12)
model.fit(X_train, y_train)

# --- ۶. پیش‌بینی و ارزیابی ---
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# --- ۷. پیش‌بینی ۵ روز آینده (مثال ساده) ---
last_data = X.tail(1)
future_predictions = []
for _ in range(5):
    prediction = model.predict(last_data)[0]
    future_predictions.append(prediction)
    new_row = last_data.copy()
    new_row['Close'] = prediction
    new_row['Close_Lag1'] = prediction
    last_data = new_row

print(f"پیش‌بینی ۵ روز آینده: {future_predictions}")

# --- ۸. ذخیره نتایج ---
data.to_csv(f"{ticker}_data.csv")
print(f"داده‌ها در {ticker}_data.csv ذخیره شد.")

# --- ۹. رسم نمودار ---
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual Price', color='blue')
plt.plot(y_pred, label='Predicted Price', color='red', alpha=0.7)
plt.legend()
plt.title(f"{ticker} - Actual vs Predicted")
plt.xlabel("Days")
plt.ylabel("Price (USD)")
plt.savefig(f"{ticker}_chart.png")
print(f"نمودار در {ticker}_chart.png ذخیره شد.")

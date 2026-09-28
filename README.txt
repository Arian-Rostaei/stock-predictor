# 📈 Stock Market Predictor | پیش‌بینی بازار سهام

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30-red?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Linux-yellow?logo=linux&logoColor=white)

**A multilingual desktop application for predicting US stock market prices using Machine Learning.**

**یک اپلیکیشن دسکتاپ چندزبانه برای پیش‌بینی قیمت سهام بازار آمریکا با استفاده از یادگیری ماشین.**

[English](#english) | [فارسی](#فارسی)

</div>

---

## 🎯 Overview

This project is a **desktop application** that fetches historical stock data from Yahoo Finance, engineers features, trains a Random Forest model, and predicts the next day's closing price. It supports **11 languages** and runs as a native Linux desktop app.

### ✨ Key Features

- 🤖 **Machine Learning**: Random Forest Regressor with 200 estimators
- 📊 **Real-time Data**: Fetches live data from Yahoo Finance via `yfinance`
- 🌍 **11 Languages**: Persian, English, Arabic, Russian, German, French, Spanish, Portuguese, Chinese, Japanese, Korean
- 🖥️ **Native Desktop App**: Runs as a standalone window using `pywebview`
- 📈 **Interactive Dashboard**: Built with Streamlit for real-time visualization
- 🎨 **Feature Engineering**: Lag features (1, 2, 3 days) + Moving Averages (MA3, MA5, MA10)
- 📉 **Model Evaluation**: MAE (Mean Absolute Error) reporting

---

## 🏗️ Architecture

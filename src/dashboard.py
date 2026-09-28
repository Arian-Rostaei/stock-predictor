import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="wide")

# --- دیکشنری ترجمه ۱۱ زبانه ---
TEXTS = {
    "fa": {
        "name": "فارسی",
        "title": "📈 پیش‌بینی بازار سهام آمریکا",
        "subtitle": "این داشبورد با استفاده از یادگیری ماشین، قیمت روز بعد سهام را پیش‌بینی می‌کند.",
        "ticker_label": "نماد سهام را وارد کنید (مثلاً AAPL, MSFT, TSLA):",
        "period_label": "بازه داده:",
        "btn": "🚀 پیش‌بینی کن",
        "spinner": "در حال دریافت داده و تحلیل...",
        "not_found": "داده‌ای برای نماد {} پیدا نشد.",
        "success": "✅ تحلیل برای {} با موفقیت انجام شد!",
        "last_price": "آخرین قیمت واقعی",
        "next_pred": "پیش‌بینی روز بعد",
        "mae": "خطای میانگین (MAE)",
        "chart_title": "📊 مقایسه قیمت واقعی و پیش‌بینی‌شده",
        "actual": "قیمت واقعی",
        "predicted": "پیش‌بینی مدل",
        "xlabel": "تاریخ",
        "ylabel": "قیمت (دلار)",
        "raw_data": "📁 مشاهده داده خام",
        "error": "خطا: {}",
        "lang_label": "🌐 زبان"
    },
    "en": {
        "name": "English",
        "title": "📈 US Stock Market Predictor",
        "subtitle": "This dashboard uses machine learning to predict the next day's stock price.",
        "ticker_label": "Enter stock ticker (e.g. AAPL, MSFT, TSLA):",
        "period_label": "Data period:",
        "btn": "🚀 Predict",
        "spinner": "Fetching data and analyzing...",
        "not_found": "No data found for ticker {}.",
        "success": "✅ Analysis for {} completed successfully!",
        "last_price": "Last Actual Price",
        "next_pred": "Next Day Prediction",
        "mae": "Mean Absolute Error (MAE)",
        "chart_title": "📊 Actual vs Predicted Price",
        "actual": "Actual Price",
        "predicted": "Predicted Price",
        "xlabel": "Date",
        "ylabel": "Price (USD)",
        "raw_data": "📁 View Raw Data",
        "error": "Error: {}",
        "lang_label": "🌐 Language"
    },
    "ar": {
        "name": "العربية",
        "title": "📈 التنبؤ بسوق الأسهم الأمريكي",
        "subtitle": "يستخدم هذا التطبيق التعلم الآلي للتنبؤ بسعر السهم في اليوم التالي.",
        "ticker_label": "أدخل رمز السهم (مثل AAPL, MSFT, TSLA):",
        "period_label": "فترة البيانات:",
        "btn": "🚀 تنبأ",
        "spinner": "جاري جلب البيانات والتحليل...",
        "not_found": "لم يتم العثور على بيانات للرمز {}.",
        "success": "✅ تم التحليل بنجاح لـ {}!",
        "last_price": "آخر سعر فعلي",
        "next_pred": "توقع اليوم التالي",
        "mae": "متوسط الخطأ المطلق (MAE)",
        "chart_title": "📊 السعر الفعلي مقابل المتوقع",
        "actual": "السعر الفعلي",
        "predicted": "السعر المتوقع",
        "xlabel": "التاريخ",
        "ylabel": "السعر (دولار)",
        "raw_data": "📁 عرض البيانات الخام",
        "error": "خطأ: {}",
        "lang_label": "🌐 اللغة"
    },
    "ru": {
        "name": "Русский",
        "title": "📈 Прогноз рынка акций США",
        "subtitle": "Эта панель использует машинное обучение для прогнозирования цены акций на следующий день.",
        "ticker_label": "Введите тикер акции (например, AAPL, MSFT, TSLA):",
        "period_label": "Период данных:",
        "btn": "🚀 Прогнозировать",
        "spinner": "Загрузка данных и анализ...",
        "not_found": "Данные для тикера {} не найдены.",
        "success": "✅ Анализ для {} успешно завершен!",
        "last_price": "Последняя фактическая цена",
        "next_pred": "Прогноз на следующий день",
        "mae": "Средняя абсолютная ошибка (MAE)",
        "chart_title": "📊 Фактическая и прогнозируемая цена",
        "actual": "Фактическая цена",
        "predicted": "Прогнозируемая цена",
        "xlabel": "Дата",
        "ylabel": "Цена (долл. США)",
        "raw_data": "📁 Просмотр необработанных данных",
        "error": "Ошибка: {}",
        "lang_label": "🌐 Язык"
    },
    "de": {
        "name": "Deutsch",
        "title": "📈 US-Aktienmarkt-Vorhersage",
        "subtitle": "Dieses Dashboard verwendet maschinelles Lernen, um den Aktienkurs des nächsten Tages vorherzusagen.",
        "ticker_label": "Aktienticker eingeben (z.B. AAPL, MSFT, TSLA):",
        "period_label": "Datenzeitraum:",
        "btn": "🚀 Vorhersagen",
        "spinner": "Daten werden abgerufen und analysiert...",
        "not_found": "Keine Daten für Ticker {} gefunden.",
        "success": "✅ Analyse für {} erfolgreich abgeschlossen!",
        "last_price": "Letzter tatsächlicher Preis",
        "next_pred": "Vorhersage für den nächsten Tag",
        "mae": "Mittlerer absoluter Fehler (MAE)",
        "chart_title": "📊 Tatsächlicher vs. vorhergesagter Preis",
        "actual": "Tatsächlicher Preis",
        "predicted": "Vorhergesagter Preis",
        "xlabel": "Datum",
        "ylabel": "Preis (USD)",
        "raw_data": "📁 Rohdaten anzeigen",
        "error": "Fehler: {}",
        "lang_label": "🌐 Sprache"
    },
    "fr": {
        "name": "Français",
        "title": "📈 Prévision du marché boursier américain",
        "subtitle": "Ce tableau de bord utilise l'apprentissage automatique pour prédire le prix de l'action du jour suivant.",
        "ticker_label": "Entrez le symbole boursier (ex: AAPL, MSFT, TSLA):",
        "period_label": "Période des données:",
        "btn": "🚀 Prédire",
        "spinner": "Récupération des données et analyse...",
        "not_found": "Aucune donnée trouvée pour le symbole {}.",
        "success": "✅ Analyse pour {} terminée avec succès!",
        "last_price": "Dernier prix réel",
        "next_pred": "Prédiction du jour suivant",
        "mae": "Erreur absolue moyenne (MAE)",
        "chart_title": "📊 Prix réel vs prédit",
        "actual": "Prix réel",
        "predicted": "Prix prédit",
        "xlabel": "Date",
        "ylabel": "Prix (USD)",
        "raw_data": "📁 Voir les données brutes",
        "error": "Erreur: {}",
        "lang_label": "🌐 Langue"
    },
    "es": {
        "name": "Español",
        "title": "📈 Predictor del mercado de valores de EE.UU.",
        "subtitle": "Este panel utiliza aprendizaje automático para predecir el precio de la acción del día siguiente.",
        "ticker_label": "Ingrese el símbolo de la acción (ej: AAPL, MSFT, TSLA):",
        "period_label": "Período de datos:",
        "btn": "🚀 Predecir",
        "spinner": "Obteniendo datos y analizando...",
        "not_found": "No se encontraron datos para el símbolo {}.",
        "success": "✅ ¡Análisis para {} completado con éxito!",
        "last_price": "Último precio real",
        "next_pred": "Predicción del día siguiente",
        "mae": "Error absoluto medio (MAE)",
        "chart_title": "📊 Precio real vs predicho",
        "actual": "Precio real",
        "predicted": "Precio predicho",
        "xlabel": "Fecha",
        "ylabel": "Precio (USD)",
        "raw_data": "📁 Ver datos sin procesar",
        "error": "Error: {}",
        "lang_label": "🌐 Idioma"
    },
    "pt": {
        "name": "Português",
        "title": "📈 Previsor do mercado de ações dos EUA",
        "subtitle": "Este painel usa aprendizado de máquina para prever o preço da ação do dia seguinte.",
        "ticker_label": "Digite o ticker da ação (ex: AAPL, MSFT, TSLA):",
        "period_label": "Período dos dados:",
        "btn": "🚀 Prever",
        "spinner": "Buscando dados e analisando...",
        "not_found": "Nenhum dado encontrado para o ticker {}.",
        "success": "✅ Análise para {} concluída com sucesso!",
        "last_price": "Último preço real",
        "next_pred": "Previsão do próximo dia",
        "mae": "Erro absoluto médio (MAE)",
        "chart_title": "📊 Preço real vs previsto",
        "actual": "Preço real",
        "predicted": "Preço previsto",
        "xlabel": "Data",
        "ylabel": "Preço (USD)",
        "raw_data": "📁 Ver dados brutos",
        "error": "Erro: {}",
        "lang_label": "🌐 Idioma"
    },
    "zh": {
        "name": "中文",
        "title": "📈 美国股市预测",
        "subtitle": "此仪表板使用机器学习来预测第二天的股票价格。",
        "ticker_label": "输入股票代码 (例如 AAPL, MSFT, TSLA):",
        "period_label": "数据周期:",
        "btn": "🚀 预测",
        "spinner": "正在获取数据并分析...",
        "not_found": "未找到代码 {} 的数据。",
        "success": "✅ {} 的分析已成功完成！",
        "last_price": "最后实际价格",
        "next_pred": "次日预测",
        "mae": "平均绝对误差 (MAE)",
        "chart_title": "📊 实际价格与预测价格",
        "actual": "实际价格",
        "predicted": "预测价格",
        "xlabel": "日期",
        "ylabel": "价格 (美元)",
        "raw_data": "📁 查看原始数据",
        "error": "错误: {}",
        "lang_label": "🌐 语言"
    },
    "ja": {
        "name": "日本語",
        "title": "📈 米国株式市場予測",
        "subtitle": "このダッシュボードは機械学習を使用して翌日の株価を予測します。",
        "ticker_label": "ティッカーを入力 (例: AAPL, MSFT, TSLA):",
        "period_label": "データ期間:",
        "btn": "🚀 予測",
        "spinner": "データを取得して分析中...",
        "not_found": "ティッカー {} のデータが見つかりません。",
        "success": "✅ {} の分析が正常に完了しました！",
        "last_price": "最後の実際の価格",
        "next_pred": "翌日の予測",
        "mae": "平均絶対誤差 (MAE)",
        "chart_title": "📊 実際の価格と予測価格",
        "actual": "実際の価格",
        "predicted": "予測価格",
        "xlabel": "日付",
        "ylabel": "価格 (USD)",
        "raw_data": "📁 生データを表示",
        "error": "エラー: {}",
        "lang_label": "🌐 言語"
    },
    "ko": {
        "name": "한국어",
        "title": "📈 미국 주식 시장 예측",
        "subtitle": "이 대시보드는 머신 러닝을 사용하여 다음 날 주가를 예측합니다.",
        "ticker_label": "티커 입력 (예: AAPL, MSFT, TSLA):",
        "period_label": "데이터 기간:",
        "btn": "🚀 예측",
        "spinner": "데이터를 가져와 분석 중...",
        "not_found": "티커 {}에 대한 데이터를 찾을 수 없습니다.",
        "success": "✅ {}에 대한 분석이 성공적으로 완료되었습니다!",
        "last_price": "마지막 실제 가격",
        "next_pred": "다음 날 예측",
        "mae": "평균 절대 오차 (MAE)",
        "chart_title": "📊 실제 가격 vs 예측 가격",
        "actual": "실제 가격",
        "predicted": "예측 가격",
        "xlabel": "날짜",
        "ylabel": "가격 (USD)",
        "raw_data": "📁 원시 데이터 보기",
        "error": "오류: {}",
        "lang_label": "🌐 언어"
    }
}

# --- انتخاب زبان ---
lang_options = {code: data["name"] for code, data in TEXTS.items()}
lang_choice = st.sidebar.selectbox(
    "🌐 Language / زبان",
    options=list(lang_options.keys()),
    format_func=lambda x: lang_options[x],
    index=0
)
t = TEXTS[lang_choice]

# --- عنوان ---
st.title(t["title"])
st.write(t["subtitle"])

# --- ورودی کاربر ---
col1, col2 = st.columns([2, 1])
with col1:
    ticker = st.text_input(t["ticker_label"], "AAPL").upper()
with col2:
    period = st.selectbox(t["period_label"], ["1y", "2y", "5y"], index=1)

# --- دکمه پیش‌بینی ---
if st.button(t["btn"], use_container_width=True):
    with st.spinner(t["spinner"]):
        try:
            data = yf.download(ticker, period=period, interval="1d")
            
            if data.empty:
                st.error(t["not_found"].format(ticker))
                st.stop()
            
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            data['Close_Lag1'] = data['Close'].shift(1)
            data['Close_Lag2'] = data['Close'].shift(2)
            data['Close_Lag3'] = data['Close'].shift(3)
            data['MA_3'] = data['Close'].rolling(window=3).mean()
            data['MA_5'] = data['Close'].rolling(window=5).mean()
            data['MA_10'] = data['Close'].rolling(window=10).mean()
            data.dropna(inplace=True)
            
            data['Target'] = data['Close'].shift(-1)
            data.dropna(inplace=True)
            
            features = ['Open', 'High', 'Low', 'Close', 'Volume',
                        'Close_Lag1', 'Close_Lag2', 'Close_Lag3',
                        'MA_3', 'MA_5', 'MA_10']
            X = data[features]
            y = data['Target']
            
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1, max_depth=12)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            last_features = X.tail(1)
            next_day_pred = model.predict(last_features)[0]
            last_actual = y_test.iloc[-1] if len(y_test) > 0 else y.iloc[-1]
            
            st.success(t["success"].format(ticker))
            
            m1, m2, m3 = st.columns(3)
            m1.metric(t["last_price"], f"${float(last_actual):.2f}")
            m2.metric(t["next_pred"], f"${float(next_day_pred):.2f}",
                      delta=f"{float(next_day_pred) - float(last_actual):.2f}")
            
            mae = mean_absolute_error(y_test, y_pred)
            m3.metric(t["mae"], f"${mae:.2f}")
            
            st.subheader(t["chart_title"])
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(y_test.index, y_test.values, label=t["actual"], color='#1f77b4', linewidth=2)
            ax.plot(y_test.index, y_pred, label=t["predicted"], color='#d62728', alpha=0.7, linewidth=2)
            ax.set_xlabel(t["xlabel"])
            ax.set_ylabel(t["ylabel"])
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            with st.expander(t["raw_data"]):
                st.dataframe(data.tail(20))
                
        except Exception as e:
            st.error(t["error"].format(str(e)))

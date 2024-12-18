import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

# Set Streamlit page configuration
st.set_page_config(page_title="SARIMA Forecasting App", layout="wide")

# Title of the app
st.title("Forecasting Metode SARIMA")
st.caption("Dibuat oleh: Agung Widodo & Rifqy Alifuddin Alfareza")
st.info("Peramalan Produksi Kemeja pada CV. Alumnistore Jombang")

# Function to check stationarity using ADF test
def check_stationarity(data):
    result = adfuller(data.dropna())
    p_value = result[1]
    st.write('ADF Statistic:', result[0])
    st.write('p-value:', p_value)
    if p_value < 0.05:
        st.success("Data stasioner (p-value < 0.05)")
        return True
    else:
        st.warning("Data tidak stasioner (p-value >= 0.05)")
        return False

# Function to automatically difference the data
def auto_difference(data, max_diff=3):
    diff_data = data.copy()
    diff_count = 0
    while not check_stationarity(diff_data) and diff_count < max_diff:
        diff_count += 1
        diff_data = diff_data.diff().dropna()
        st.write(f"Data setelah differencing ke-{diff_count}:")
        st.line_chart(diff_data)
    return diff_data, diff_count

# Function to calculate MAPE
def calculate_mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

# Function to create forecast and backcast plots
def create_forecast_plots(df, results, forecast_periods):
    """
    Plot actual values, backcasted values, and forecasted values.
    """
    # Forecast future data
    forecast = results.get_forecast(steps=forecast_periods)
    forecast_mean = forecast.predicted_mean
    forecast_index = pd.date_range(df.index[-1], periods=forecast_periods + 1, freq='M')[1:]

    # Backcast (in-sample prediction)
    backcast = results.get_prediction(start=df.index[0], end=df.index[-1])
    backcast_mean = backcast.predicted_mean

    # Combine data
    combined_data = pd.DataFrame({
        'Date': df.index.tolist() + forecast_index.tolist(),
        'Permintaan Aktual': df['Value'].tolist() + [None] * len(forecast_index),
        'Peramalan': backcast_mean.tolist() + forecast_mean.tolist()
    })

    # Calculate MAPE for backcasted data
    mape = calculate_mape(df['Value'], backcast_mean)
    st.subheader(f"Nilai MAPE: {mape:.2f}%")

    # Plot the chart
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot Permintaan Aktual (Actual Demand)
    ax.plot(combined_data['Date'], combined_data['Permintaan Aktual'], 
            marker='o', linestyle='-', color='blue', linewidth=2, label='Permintaan Aktual')

    # Plot Peramalan (Forecast + Backcast)
    ax.plot(combined_data['Date'], combined_data['Peramalan'], 
            marker='o', linestyle='--', color='orange', linewidth=2, label='Peramalan')

    # Formatting the chart
    ax.set_title("Grafik Hasil Peramalan dan Backcasting", fontsize=14)
    ax.set_xlabel("Periode", fontsize=12)
    ax.set_ylabel("Jumlah Permintaan", fontsize=12)
    ax.legend(loc="upper left")
    ax.grid(True)

    # Display the chart
    st.pyplot(fig)

    # Display forecast table
    st.subheader("Tabel Forecast 12 Bulan ke Depan")
    forecast_df = pd.DataFrame({
        'Periode': forecast_index.strftime('%Y-%m'),
        'Nilai Peramalan': forecast_mean.round(2)
    })
    st.dataframe(forecast_df)

# Main Streamlit app logic
uploaded_file = st.file_uploader("Upload File CSV", type=['csv'])

if uploaded_file is not None:
    try:
        # Read and adjust the CSV format
        df = pd.read_csv(uploaded_file)
        df['Date'] = pd.to_datetime(df['Month'], format='%b-%y')  # Convert Month to datetime
        df = df[['Date', 'Value']]  # Keep only Date and Value columns
        df.set_index('Date', inplace=True)

        # Display uploaded data
        st.subheader("Data Hasil Upload")
        st.write(df)

        # Plot initial data
        st.subheader("📉 Grafik Data Asli")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index, df['Value'], linestyle='-', marker='o', color='tab:blue')
        ax.set_title("Grafik Data Asli")
        ax.set_xlabel("Periode")
        ax.set_ylabel("Jumlah Permintaan")
        ax.grid(True)
        st.pyplot(fig)

        # Check stationarity and apply differencing
        diff_data, d = auto_difference(df['Value'])

        st.success(f"Data stasioner setelah differencing ke-{d} kali")

        # SARIMA parameter selection
        st.subheader("Parameter Model SARIMA")
        col1, col2, col3 = st.columns(3)
        with col1:
            p = st.slider('p (AR order)', 0, 3, 1)
            d = st.slider('d (Difference order)', 0, 2, 1)
            q = st.slider('q (MA order)', 0, 3, 1)
        with col2:
            P = st.slider('P (Seasonal AR order)', 0, 3, 1)
            D = st.slider('D (Seasonal difference order)', 0, 2, 1)
            Q = st.slider('Q (Seasonal MA order)', 0, 3, 1)
        with col3:
            s = st.slider('s (Seasonal period)', 1, 12, 12)

        # Build SARIMA model
        if st.button("Forecasting"):
            with st.spinner("Membangun model SARIMA..."):
                model = SARIMAX(df['Value'], order=(p, d, q), seasonal_order=(P, D, Q, s))
                results = model.fit()
                st.success("Model SARIMA berhasil dibangun!")

                # Display model summary
                st.subheader("Ringkasan Model")
                st.code(results.summary())

                # Forecast for the next 12 months
                st.subheader("Hasil Forecast dan Backcast")
                create_forecast_plots(df, results, 12)

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
        st.info("Pastikan file CSV berisi kolom 'Month' dan 'Value'.")
else:
    st.info("Silakan upload file CSV untuk memulai analisis.")

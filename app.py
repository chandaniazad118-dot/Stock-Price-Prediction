
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Price Prediction System")
st.subheader("Machine Learning Project")

st.write(
    "This dashboard displays historical test results "
    "from a Linear Regression stock price prediction model."
)

results_file = Path("prediction_results.csv")
chart_file = Path("stock_prediction.png")
errors_file = Path("prediction_errors.png")

if results_file.exists():

    results = pd.read_csv(results_file)

    st.success("Prediction results loaded successfully!")

    st.subheader("Prediction Results")
    st.dataframe(results, use_container_width=True)

    actual = results["Actual_Close"]
    predicted = results["Predicted_Close"]

    mae = (actual - predicted).abs().mean()

    st.metric("Mean Absolute Error", f"{mae:.2f}")

    chart_data = pd.DataFrame({
        "Actual Price": actual.values,
        "Predicted Price": predicted.values
    })

    st.subheader("Actual vs Predicted Prices")
    st.line_chart(chart_data)

    csv_data = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Prediction Results",
        data=csv_data,
        file_name="prediction_results.csv",
        mime="text/csv"
    )

else:
    st.warning(
        "prediction_results.csv was not found. "
        "Run main.py first."
    )

if chart_file.exists():
    st.subheader("Original Prediction Graph")
    st.image(str(chart_file), use_container_width=True)

if errors_file.exists():
    st.subheader("Prediction Error Distribution")
    st.image(str(errors_file), use_container_width=True)

st.divider()

st.write(
    "Note: This is a college project demonstration, "
    "not real financial advice."
)

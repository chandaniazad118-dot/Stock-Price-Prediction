
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Step 1: Create sample dataset
np.random.seed(42)

days = 300
dates = pd.bdate_range("2025-01-01", periods=days)

changes = np.random.normal(0.15, 1.5, days)
prices = 100 + np.cumsum(changes)

df = pd.DataFrame({
    "Date": dates,
    "Close": prices
})

df.to_csv("stock_data.csv", index=False)

# Step 2: Prepare dataset
df["Previous_Close"] = df["Close"].shift(1)
df["Moving_Average"] = (
    df["Close"].rolling(5).mean().shift(1)
)
df["Next_Close"] = df["Close"].shift(-1)
df["Target_Date"] = df["Date"].shift(-1)

df.dropna(inplace=True)

# Step 3: Select features
X = df[["Previous_Close", "Moving_Average"]]
y = df["Next_Close"]

# Step 4: Split data chronologically
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# Step 5: Train model
model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained successfully!")

# Step 6: Predict
predictions = model.predict(X_test)

# Step 7: Calculate error
mae = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", round(mae, 2))

# Step 8: Save prediction results
results = pd.DataFrame({
    "Date": df["Target_Date"].iloc[split:].values,
    "Actual_Close": y_test.values,
    "Predicted_Close": predictions
})

results.to_csv("prediction_results.csv", index=False)

# Step 9: Actual vs Predicted graph
plt.figure(figsize=(10, 5))

plt.plot(results["Date"], results["Actual_Close"],
         label="Actual Price")

plt.plot(results["Date"], results["Predicted_Close"],
         label="Predicted Price")

plt.title("Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("stock_prediction.png")
plt.close()

# Step 10: Error graph
results["Error"] = (
    results["Actual_Close"] - results["Predicted_Close"]
)

plt.figure(figsize=(10, 5))
plt.plot(results["Date"], results["Error"])

plt.title("Prediction Error Graph")
plt.xlabel("Date")
plt.ylabel("Prediction Error")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("prediction_errors.png")
plt.close()

print("All files generated successfully!")

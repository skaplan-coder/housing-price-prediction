# =========================================
# Week 1 Homework - Housing Price Analysis
# Updated Version (No 'year_built' required)
# =========================================

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# -----------------------------
# 1. Load the dataset
# -----------------------------
df = pd.read_csv("housing.csv")  # Make sure housing.csv is in the same folder

# Quick look at data
print("First 5 rows of the dataset:")
print(df.head())
print("\nColumns in dataset:", df.columns)

# -----------------------------
# 2. Calculate average price per city
# -----------------------------
if "city" in df.columns:
    average_prices = df.groupby("city")["price"].mean()
    print("\nAverage housing price per city:")
    print(average_prices)
else:
    print("\nNo 'city' column found in dataset.")

# -----------------------------
# 3. Plot housing price distribution
# -----------------------------
plt.figure(figsize=(8,5))
plt.hist(df["price"], bins=20, color="skyblue")
plt.xlabel("Price (in thousands)")
plt.ylabel("Number of Houses")
plt.title("Housing Price Distribution")
plt.show()

# -----------------------------
# 4. Build Linear Regression Model (basic)
# -----------------------------
# Select only existing numeric columns for basic model
numeric_columns = ["rooms", "size_sqm", "floor"]
X_basic = df[[col for col in numeric_columns if col in df.columns]]
y = df["price"]

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X_basic, y, test_size=0.2, random_state=42)

# Train the model
model_basic = LinearRegression()
model_basic.fit(X_train, y_train)

# Predictions
y_pred_basic = model_basic.predict(X_test)

# Plot predictions vs actual
plt.scatter(y_test, y_pred_basic)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Basic Linear Regression Predictions")
plt.show()

print("Basic Model Score:", model_basic.score(X_test, y_test))

# -----------------------------
# 5. Improve model by adding categorical features (city) if available
# -----------------------------
X_advanced = X_basic.copy()

if "city" in df.columns:
    X_advanced = df[numeric_columns + ["city"]]
    X_advanced = pd.get_dummies(X_advanced, columns=["city"], drop_first=True)

# Train/test split
X_train_adv, X_test_adv, y_train_adv, y_test_adv = train_test_split(X_advanced, y, test_size=0.2, random_state=42)

# Train improved model
model_adv = LinearRegression()
model_adv.fit(X_train_adv, y_train_adv)

# Predictions
y_pred_adv = model_adv.predict(X_test_adv)

# Plot predictions vs actual
plt.scatter(y_test_adv, y_pred_adv, color='green')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Advanced Linear Regression Predictions")
plt.show()

print("Advanced Model Score:", model_adv.score(X_test_adv, y_test_adv))

# -----------------------------
# 6. Save results to CSV (optional)
# -----------------------------
results = pd.DataFrame({
    "Actual_Price": y_test_adv,
    "Predicted_Price": y_pred_adv
})
results.to_csv("housing_predictions.csv", index=False)
print("Predictions saved to housing_predictions.csv")

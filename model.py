# ==========================================
# AI Internship - Day 12
# Project Improvement
# ==========================================

# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ------------------------------------------
# Load Dataset
# ------------------------------------------
data = pd.read_csv("data/student_scores.csv")

# Display first five rows
print("Dataset Preview")
print(data.head())

# ------------------------------------------
# Prepare Features and Target
# ------------------------------------------
X = data[["Hours"]]
y = data["Scores"]

# ------------------------------------------
# Split Dataset
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ------------------------------------------
# Train Model
# ------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ------------------------------------------
# Make Predictions
# ------------------------------------------
predictions = model.predict(X_test)

# ------------------------------------------
# Evaluate Model
# ------------------------------------------
print("\nModel Performance")

print("MAE :", round(mean_absolute_error(y_test, predictions),2))
print("MSE :", round(mean_squared_error(y_test, predictions),2))
print("R²  :", round(r2_score(y_test, predictions),2))

# ------------------------------------------
# Compare Results
# ------------------------------------------
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions.round(2)
})

print("\nPrediction Comparison")
print(results)

print("\nProject Organized Successfully!")
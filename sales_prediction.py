# Sales Prediction using Machine Learning
# CodeAlpha Internship Task 4

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("Advertising.csv")

# Remove unnecessary column
data = data.drop("Unnamed: 0", axis=1)

# Display first few rows
print("First 5 Rows:")
print(data.head())

# Input features
X = data[["TV", "Radio", "Newspaper"]]

# Target variable
y = data["Sales"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict sales
predictions = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Actual vs Predicted Sales Graph
plt.figure(figsize=(8, 5))
plt.scatter(y_test, predictions)

plt.title("Actual vs Predicted Sales")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")

plt.show()
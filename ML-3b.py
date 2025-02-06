#pip install numpy pandas scikit-learn statsmodels
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Generate synthetic dataset
np.random.seed(42)
num_samples = 100

# Features (X1, X2, X3)
X1 = 2 * np.random.rand(num_samples, 1)
X2 = 3 * np.random.rand(num_samples, 1)
X3 = 5 * np.random.rand(num_samples, 1) + 0.5 * X1  # Introduce some correlation with X1

# Target (y)
y = 4 + 3 * X1 + 2 * X2 + 1.5 * X3 + np.random.randn(num_samples, 1)

# Combine features into a DataFrame for easier manipulation
features = np.hstack([X1, X2, X3])
feature_names = ['X1', 'X2', 'X3']
data = pd.DataFrame(features, columns=feature_names)
data['y'] = y

# Split dataset into training and testing sets
X = data[feature_names]
y = data['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check for multicollinearity using Variance Inflation Factor (VIF)
def calculate_vif(X):
    vif = pd.DataFrame()
    vif['Feature'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif

vif_before = calculate_vif(X)
print("Variance Inflation Factor before feature selection:")
print(vif_before)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Extract model parameters
intercept = model.intercept_
coefficients = model.coef_

print("\nModel Coefficients:")
for name, coef in zip(feature_names, coefficients):
    print(f"{name}: {coef}")

print(f"\nIntercept: {intercept}")

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Plot predictions vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2, label='Perfect fit')
plt.title('Predicted vs Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid(True)
plt.show()

# Plot residuals
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, color='purple', alpha=0.6)
plt.axhline(0, color='red', linestyle='--', linewidth=2)
plt.title('Residuals vs Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()

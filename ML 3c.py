#pip install numpy matplotlib scikit-learn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression

# Generate synthetic dataset
X, y = make_regression(n_samples=100, n_features=1, noise=15, random_state=42)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models with regularization parameters
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)

# Train models
ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)
elastic_net.fit(X_train, y_train)

# Predict on test data
ridge_pred = ridge.predict(X_test)
lasso_pred = lasso.predict(X_test)
elastic_net_pred = elastic_net.predict(X_test)

# Evaluate models
ridge_mse = mean_squared_error(y_test, ridge_pred)
lasso_mse = mean_squared_error(y_test, lasso_pred)
elastic_net_mse = mean_squared_error(y_test, elastic_net_pred)

print(f'Ridge Regression MSE: {ridge_mse:.2f}')
print(f'Lasso Regression MSE: {lasso_mse:.2f}')
print(f'ElasticNet Regression MSE: {elastic_net_mse:.2f}')

# Plot results
plt.scatter(X_test, y_test, color='black', label="Actual Data")
plt.plot(X_test, ridge_pred, color='blue', label="Ridge Regression")
plt.plot(X_test, lasso_pred, color='red', label="Lasso Regression")
plt.plot(X_test, elastic_net_pred, color='green', label="ElasticNet Regression")
plt.legend()
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Regularized Regression Models")
plt.show()

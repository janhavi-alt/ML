# pip install scikit-learn

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Sample dataset (features and labels)
# Features: [Age, Salary]
# Labels: 0 - Not Purchased, 1 - Purchased
X = np.array([[22, 50000], [25, 60000], [27, 70000], [35, 80000], [40, 120000],
              [50, 150000], [22, 40000], [23, 45000], [30, 95000], [33, 100000]])
y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Naive Bayes classifier
model = GaussianNB()
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Output the results
print("Predicted labels: ", y_pred)
print("Actual labels: ", y_test)
print("Accuracy of the model: ", accuracy)

# Test with a custom sample
custom_sample = np.array([[28,75000]])  # Example: Age 28, Salary 75000
custom_prediction = model.predict(custom_sample)
print(f"Prediction for sample [28, 75000]: {'Purchased' if custom_prediction == 1 else 'Not Purchased'}")

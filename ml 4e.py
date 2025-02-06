#pip install scikit-learn matplotlib numpy

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load Digits dataset
data = load_digits()
X = data.data
y = data.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a single decision tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Predict and evaluate the performance of the decision tree
y_pred_dt = dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Accuracy of single Decision Tree: {accuracy_dt:.4f}")

# Experiment with Random Forest Classifier with different numbers of trees
tree_counts = [10, 50, 100, 200]
accuracies_rf = []

for n_trees in tree_counts:
    rf = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    rf.fit(X_train, y_train)
    
    # Predict and evaluate the performance of the random forest
    y_pred_rf = rf.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    accuracies_rf.append(accuracy_rf)
    print(f"Accuracy of Random Forest with {n_trees} trees: {accuracy_rf:.4f}")

# Plotting performance comparison
plt.figure(figsize=(10, 6))
plt.plot(tree_counts, accuracies_rf, label="Random Forest", marker='o')
plt.axhline(accuracy_dt, color='r', linestyle='--', label="Single Decision Tree")
plt.xlabel("Number of Trees in Random Forest")
plt.ylabel("Accuracy")
plt.title("Performance Comparison: Decision Tree vs Random Forest on Digits Dataset")
plt.legend()
plt.grid(True)
plt.show()

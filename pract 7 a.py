# pip install scikit-learn numpy
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset (Iris dataset as an example)
data = load_iris()
X = data.data
y = data.target

# Define a model (Logistic Regression as an example)
model = LogisticRegression(max_iter=200)

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
kf_accuracies = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    kf_accuracies.append(accuracy)

print(f'K-Fold Cross-Validation Accuracies: {kf_accuracies}')
print(f'Mean Accuracy: {np.mean(kf_accuracies)}')

# Stratified K-Fold Cross-Validation (useful for imbalanced datasets)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skf_accuracies = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    skf_accuracies.append(accuracy)

print(f'Stratified K-Fold Cross-Validation Accuracies: {skf_accuracies}')
print(f'Mean Accuracy: {np.mean(skf_accuracies)}')

# Use scikit-learn's cross_val_score function for an easier implementation
cv_scores = cross_val_score(model, X, y, cv=5)
print(f'Cross-Validation Scores using cross_val_score: {cv_scores}')
print(f'Mean Score using cross_val_score: {np.mean(cv_scores)}')

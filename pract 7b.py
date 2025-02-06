# pip install scikit-learn numpy scipy
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from scipy.stats import randint

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the model
model = RandomForestClassifier(random_state=42)

# 1. Grid Search
grid_param = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(estimator=model, param_grid=grid_param, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Print best parameters and score for Grid Search
print("Grid Search Best Parameters:", grid_search.best_params_)
print("Grid Search Best Score:", grid_search.best_score_)

# 2. Randomized Search
random_param = {
    'n_estimators': randint(50, 200),
    'max_depth': [5, 10, None],
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5),
    'bootstrap': [True, False]
}

random_search = RandomizedSearchCV(estimator=model, param_distributions=random_param, n_iter=100, cv=3, n_jobs=-1, verbose=2, random_state=42)
random_search.fit(X_train, y_train)

# Print best parameters and score for Randomized Search
print("Randomized Search Best Parameters:", random_search.best_params_)
print("Randomized Search Best Score:", random_search.best_score_)

# Evaluate on the test set
grid_test_score = grid_search.score(X_test, y_test)
random_test_score = random_search.score(X_test, y_test)

print(f"Grid Search Test Score: {grid_test_score}")
print(f"Randomized Search Test Score: {random_test_score}")

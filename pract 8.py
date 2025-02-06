# pip install pandas numpy

import pandas as pd
import numpy as np

# Sample dataset
data = {
    'Weather': ['Sunny', 'Sunny', 'Rainy', 'Sunny', 'Rainy', 'Rainy', 'Sunny', 'Sunny', 'Rainy', 'Sunny'],
    'Age': ['Young', 'Old', 'Middle-aged', 'Middle-aged', 'Old', 'Young', 'Old', 'Middle-aged', 'Young', 'Old'],
    'Income': ['High', 'Low', 'Low', 'High', 'Low', 'High', 'High', 'Low', 'Low', 'High'],
    'Buy': ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes']
}

# Convert the dataset into a pandas DataFrame
df = pd.DataFrame(data)

# Function to calculate prior probabilities P(Y)
def calculate_prior(df):
    total = len(df)
    prior = df['Buy'].value_counts() / total
    return prior

# Function to calculate likelihood probabilities P(X|Y)
def calculate_likelihood(df, feature, target_class):
    feature_values = df[feature].unique()
    likelihood = {}
    
    for value in feature_values:
        # Probability of feature value given the target class
        likelihood[value] = len(df[(df[feature] == value) & (df['Buy'] == target_class)]) / len(df[df['Buy'] == target_class])
    
    return likelihood

# Function to predict the class (Buy/Not Buy) using Bayesian Inference
def predict(df, sample):
    # Step 1: Calculate Prior Probabilities P(Y)
    prior = calculate_prior(df)
    
    # Step 2: Calculate Likelihoods P(X|Y)
    likelihoods = {}
    for feature in ['Weather', 'Age', 'Income']:
        likelihoods[feature] = {}
        for target_class in df['Buy'].unique():
            likelihoods[feature][target_class] = calculate_likelihood(df, feature, target_class)
    
    # Step 3: Calculate Posterior Probabilities P(Y|X) using Bayes' Theorem
    posterior = {}
    for target_class in df['Buy'].unique():
        posterior[target_class] = prior[target_class]
        for feature, value in sample.items():
            # Multiply the likelihood of each feature value given the class
            if value in likelihoods[feature][target_class]:
                posterior[target_class] *= likelihoods[feature][target_class].get(value, 0)
            else:
                posterior[target_class] *= 0  # Assign zero if the value wasn't found in likelihood
        
    # Normalize the posterior probabilities
    total_posterior = sum(posterior.values())
    for target_class in posterior:
        posterior[target_class] /= total_posterior
    
    return posterior

# Sample data to predict: Weather = Sunny, Age = Young, Income = High
sample = {
    'Weather': 'Sunny',
    'Age': 'Young',
    'Income': 'High'
}

# Prediction
posterior = predict(df, sample)

# Display the results
print(sample)
print("Posterior probabilities for each class:")
for target_class, prob in posterior.items():
    print(f"Class {target_class}: {prob:.4f}")

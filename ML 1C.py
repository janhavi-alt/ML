# pip install pandas numpy scikit-learn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, Binarizer
from sklearn.datasets import make_classification

# Create a synthetic dataset with adjusted parameters
X, y = make_classification(n_samples=100, n_features=5, n_classes=3, n_clusters_per_class=1, random_state=42)

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
df['target'] = y

print("First few rows of the dataset:")
print(df.head())

# Example of Label Encoding (assuming target is categorical)
# Convert target to categorical string values
df['target'] = df['target'].astype('category')
df['target'] = df['target'].cat.codes  # Convert categories to codes

print("\nAfter Label Encoding:")
print(df.head())

# Apply Label Encoding using sklearn (as an alternative method)
label_encoder = LabelEncoder()
df['target'] = label_encoder.fit_transform(df['target'])
print("\nAfter Label Encoding using sklearn:")
print(df.head())

# Scaling features using StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop('target', axis=1))
df_scaled = pd.DataFrame(scaled_features, columns=df.columns[:-1])
df_scaled['target'] = df['target']

print("\nAfter Scaling:")
print(df_scaled.head())

# Binarization of features
binarizer = Binarizer(threshold=0.0)  # threshold can be adjusted based on your needs
binarized_features = binarizer.fit_transform(df_scaled.drop('target', axis=1))
df_binarized = pd.DataFrame(binarized_features, columns=df_scaled.columns[:-1])
df_binarized['target'] = df_scaled['target']

print("\nAfter Binarization:")
print(df_binarized.head())

# Display statistics of the transformed dataset
print("\nDescriptive statistics after preprocessing:")
print(df_binarized.describe())

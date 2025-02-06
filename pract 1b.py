# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
url = 'https://raw.githubusercontent.com/plotly/datasets/master/iris-data.csv'
df = pd.read_csv(url)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Calculate descriptive summary statistics
print("\nDescriptive summary statistics:")
print(df.describe())

# Univariate analysis using histograms
plt.figure(figsize=(14, 7))

for i, column in enumerate(df.columns[1:5], 1):
    plt.subplot(2, 2, i)
    sns.histplot(df[column], kde=True)
    plt.title(f'Histogram of {column}')

plt.tight_layout()
plt.show()

# Univariate analysis using boxplots
plt.figure(figsize=(14, 7))

for i, column in enumerate(df.columns[1:5], 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x='species', y=column, data=df)
    plt.title(f'Boxplot of {column} by Species')

plt.tight_layout()
plt.show()

# Bivariate analysis using scatterplots
plt.figure(figsize=(14, 7))

for i, (x, y) in enumerate([(df.columns[1], df.columns[2]), 
                             (df.columns[1], df.columns[3]),
                             (df.columns[1], df.columns[4]),
                             (df.columns[2], df.columns[3]),
                             (df.columns[2], df.columns[4]),
                             (df.columns[3], df.columns[4])], 1):
    plt.subplot(2, 3, i)
    sns.scatterplot(x=x, y=y, hue='species', data=df, palette='Set2')
    plt.title(f'Scatterplot of {x} vs {y}')

plt.tight_layout()
plt.show()

# Pairplot for bivariate analysis
sns.pairplot(df, hue='species')
plt.title('Pairplot of the Iris dataset')
plt.show()

# Identify potential features and target variables
features = df.columns[1:5].tolist()  # Petal and Sepal measurements
target = 'species'  # Species of the iris plant

print("\nPotential features:")
print(features)
print("\nPotential target variable:")
print(target)

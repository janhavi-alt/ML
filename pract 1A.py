import pandas as pd
import numpy as np

# Load the dataset
file_path = 'C:/Users/Janhavi/Downloads/sample_data.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
print("Initial Data:")
print(df.head())

# Check for missing values
print("\nMissing Values:")
missing_values = df.isna().sum()
print(missing_values)

# Ensure missing values are correctly recognized
# Convert empty strings to NaN if they exist
df.replace("", np.nan, inplace=True)

# Separate numerical and categorical columns
numerical_columns = df.select_dtypes(include=np.number).columns
categorical_columns = df.select_dtypes(include='object').columns

# Fill missing values for numerical columns with mean
df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())

# Fill missing values for categorical columns with mode
df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])

# Ensure all column names are lowercase and have no leading/trailing spaces
df.columns = df.columns.str.lower().str.strip()

#Handling inconsistent formatting (e.g., strip whitespace)
for col in categorical_columns:
    df[col] = df[col].str.strip().str.lower()

# Define a function to detect and display outliers using IQR (Interquartile Range)
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Detect and display outliers for each numerical column
for col in numerical_columns:
    outliers, lower_bound, upper_bound = detect_outliers_iqr(df, col)
    if not outliers.empty:
        print(f"\nOutliers in column '{col}':")
        print(outliers)
        print(f"Lower Bound: {lower_bound}")
        print(f"Upper Bound: {upper_bound}")

# Define a function to remove outliers using IQR (Interquartile Range)
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[column] >= (Q1 - 1.5 * IQR)) & (df[column] <= (Q3 + 1.5 * IQR))]

# Apply the function to remove outliers from numerical columns
for col in numerical_columns:
    df = remove_outliers_iqr(df, col)

# Display the cleaned data
print("\nCleaned Data:")
print(df.head())

# Save the cleaned data to a new CSV file
df.to_csv('C:/Users/Janhavi/Downloads/cleaned_dataset.csv', index=False)

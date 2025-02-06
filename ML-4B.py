#pip install pandas scikit-learn numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the expanded data from CSV file
data = pd.read_csv(r'C:\Users\Janhavi\Downloads\data.csv')  # Update with your correct file path

# Check class distribution to see if there is an imbalance
print("Class distribution:\n", data.iloc[:, -1].value_counts())

# Split the data into features (X) and labels (y)
X = data.iloc[:, :-1]  # Features: all columns except the last one
y = data.iloc[:, -1]   # Labels: the last column

# Stratified split to ensure equal distribution of classes in both train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling (standardizing the features)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit and transform on the training data
X_test = scaler.transform(X_test)        # Only transform on the testing data

# Try different values of k (neighbors)
k_values = [3, 5, 7, 9]
best_k = 3  # Start with a default value of 3
best_accuracy = 0

# Manually tune the k value and check the accuracy
for k in k_values:
    # Create the KNN classifier with the current k
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Train the model on the training data
    knn.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = knn.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy with k={k}: {accuracy * 100:.2f}%")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

# Print the best k value and the corresponding accuracy
print(f"\nBest k value: {best_k}")
print(f"Best accuracy: {best_accuracy * 100:.2f}%")

# Final predictions with the best k value
final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train, y_train)
y_pred_final = final_knn.predict(X_test)

# Print final predictions
print(f"\nTrue labels: {y_test.tolist()}")
print(f"Predicted labels: {y_pred_final.tolist()}")

# Compare the predictions to the true labels and print correct/incorrect predictions
correct = 0
incorrect = 0

for true, pred in zip(y_test, y_pred_final):
    if true == pred:
        correct += 1
        print(f"Correct prediction: True={true}, Predicted={pred}")
    else:
        incorrect += 1
        print(f"Incorrect prediction: True={true}, Predicted={pred}")

# Print final accuracy
final_accuracy = accuracy_score(y_test, y_pred_final)
print(f"\nFinal Accuracy: {final_accuracy * 100:.2f}%")
print(f"Correct predictions: {correct}")
print(f"Incorrect predictions: {incorrect}")

import csv

# Read the CSV file and load data into a list
a = []
with open(r'C:\Users\Janhavi\Downloads\training_data.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        a.append(row)

# Assuming the last column contains the target variable (PlayTennis)
num_attribute = len(a[0]) - 1  # Exclude the target column
print("\nThe initial hypothesis is: ")
hypothesis = ['?'] * num_attribute  # Initialize with '?' instead of '0'
print(hypothesis)

print("\nThe total number of training instances are: ", len(a))

# Iterate through the data and apply the FIND-S algorithm
for i in range(1, len(a)):  # Start from 1 to skip header
    # Check if the current instance is a positive sample (PlayTennis == 'Yes')
    if a[i][num_attribute].strip().lower() == 'yes':  # Case insensitive check
        print(f"\nProcessing positive instance {i}: {a[i]}")
        for j in range(num_attribute):
            # Update the hypothesis for matching attributes, generalize for mismatches
            if hypothesis[j] == '?' or hypothesis[j] == a[i][j]:
                hypothesis[j] = a[i][j]
            else:
                hypothesis[j] = '?'
        
        print(f"Updated hypothesis after instance {i}: {hypothesis}")

# Final maximally specific hypothesis
print("\nThe Maximally specific hypothesis for the training instances is: ")
print(hypothesis)

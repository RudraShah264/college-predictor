import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset into a DataFrame
df = pd.read_csv('openmerit1.xls')

# Identify and replace non-numeric values
df['OPEN_HS'] = df['OPEN_HS'].replace('-----', np.nan)
df['OPEN_HS'] = pd.to_numeric(df['OPEN_HS'], errors='coerce')

# Drop rows with NaN values in 'OPEN_HS' and 'Program'
df.dropna(subset=['OPEN_HS', 'Program'], inplace=True)

# Drop rows where 'Name of Institute' is missing
df.dropna(subset=['Name of Institute'], inplace=True)

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Fit the LabelEncoder on the 'Program' column and transform it
df['Program'] = label_encoder.fit_transform(df['Program'])

# Define features (X) and target (y)
X = df[['Program', 'OPEN_HS']]  # Features
y = df['Name of Institute']  # Target variable

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the decision tree classifier
classifier = DecisionTreeClassifier(random_state=42)

# Train the classifier on the training data
classifier.fit(X_train, y_train)

# Predict on the test data
y_pred = classifier.predict(X_test)

# Calculate accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy:.2f}')
# Save the trained model and the label encoder to disk
joblib.dump(classifier, 'ml_model/classifier.joblib')
joblib.dump(label_encoder, 'ml_model/label_encoder.joblib')

# Example of new data
# new_data = pd.DataFrame({
#     'Program': ['COMPUTER ENGINEERING'],
#     'OPEN_HS': [15400]
# })

# # Transform the 'Program' column using the fitted label encoder
# new_data['Program'] = label_encoder.transform(new_data['Program'])

# # Predict the 'Name of Institute' for the new data
# predictions = classifier.predict(new_data)

# # Display the predictions
# print(predictions)


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Load the Dataset
# (Ensure 'Iris.csv' is in your working directory)
df = pd.read_csv('Iris.csv')

# Quick inspection of the dataset
print("Dataset Head:")
print(df.head())
print("\nDataset Info:")
print(df.info())

# 2. Data Preprocessing
# Drop the 'Id' column if it exists in the dataset
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

# Separate features (X) and target label (y)
X = df.drop(columns=['Species'])
y = df['Species']

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling for optimal performance with distance-based models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Training (Using K-Nearest Neighbors Classifier)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_scaled, y_train)

# 4. Model Evaluation
y_pred = model.predict(X_test_scaled)

print("\n--- Model Evaluation ---")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='d',
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix - Iris Classification')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

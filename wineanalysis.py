import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=';')

# Basic Info
print("Basic Info")
print(data.info())

# Summary Statistics
print("\nSummary Statistics of the Dataset:")
print(data.describe())

# Missing Values
print("\nMissing Values:")
print(data.isnull().sum())

# Data Visualization
plt.figure(figsize=(8, 5))
sns.set(style="whitegrid")
sns.countplot(x='quality', data=data)
plt.xlabel('Wine Quality')
plt.ylabel('Count')
plt.title('Distribution of Wine Quality Scores')
plt.show()

# Correlation Matrix Heatmap
plt.figure(figsize=(10, 6))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title('Correlation Matrix Heatmap')
plt.show()

# Feature vs. Wine Quality
features = data.columns[:-1]
for feature in features:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='quality', y=feature, data=data)
    plt.xlabel('Wine Quality')
    plt.ylabel(feature)
    plt.title(f'{feature} vs. Wine Quality')
    plt.show()

# Data Analysis
X = data.drop('quality', axis=1)
y = data['quality']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Feature Importances
feature_importances = pd.Series(clf.feature_importances_, index=X.columns)
sorted_importances = feature_importances.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=sorted_importances, y=sorted_importances.index, palette="viridis")
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()

# Model Evaluation
y_pred = clf.predict(X_test)

print("Model Evaluation:")
print(classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print("Feature Importances:")
print(sorted_importances)

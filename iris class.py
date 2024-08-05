
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import joblib

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Data exploration
print("Dataset shape:", df.shape)
print("First 5 rows:\n", df.head())
print("Class distribution:\n", df['species'].value_counts())

# Data visualization
sns.pairplot(df, hue='species', markers=["o", "s", "D"])
plt.show()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Data preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define classifiers
classifiers = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=200)
}

# Hyperparameters for Grid Search
param_grid = {
    'Decision Tree': {'clf__max_depth': [3, 5, 7]},
    'Random Forest': {'clf__n_estimators': [50, 100, 200]},
    'Support Vector Machine': {'clf__C': [0.1, 1, 10], 'clf__kernel': ['linear', 'rbf']},
    'K-Nearest Neighbors': {'clf__n_neighbors': [3, 5, 7]},
    'Logistic Regression': {'clf__C': [0.1, 1, 10]}
}

# Train, tune, and evaluate each classifier
best_classifiers = {}
for name, clf in classifiers.items():
    pipeline = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
    grid_search = GridSearchCV(pipeline, param_grid[name], cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_classifiers[name] = grid_search.best_estimator_
    y_pred = grid_search.predict(X_test)
    
    print(f"Classifier: {name}")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))
    print("\n" + "="*60 + "\n")

# Save the best model
best_model_name = max(best_classifiers, key=lambda name: accuracy_score(y_test, best_classifiers[name].predict(X_test)))
best_model = best_classifiers[best_model_name]
joblib.dump(best_model, 'best_model.pkl')

print(f"The best model is {best_model_name} and has been saved as 'best_model.pkl'")

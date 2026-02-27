import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

os.makedirs("model", exist_ok=True)

# Load dataset
data = pd.read_csv("pima_diabetes.csv")

# Columns where 0 is invalid
cols_with_zero = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']

# Replace 0 with median
for col in cols_with_zero:
    data[col] = data[col].replace(0, np.nan)
    data[col] = data[col].fillna(data[col].median())

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Random Forest with tuned hyperparameters
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=7,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)

# Cross-validation to check accuracy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
print("CV Accuracy:", np.mean(cv_scores))

# Train model
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)

# Save model and scaler
pickle.dump(model, open("model/diabetes_model.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))
pickle.dump(accuracy, open("model/accuracy.pkl", "wb"))

print("Model saved successfully!")
"""
PyGesture Studio
train.py

Train and compare multiple gesture classifiers.
"""

import os
import time
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.neural_network import MLPClassifier

# ==========================================
# Load Dataset
# ==========================================

csv_file = "dataset/gestures.csv"

if not os.path.exists(csv_file):
    raise FileNotFoundError(csv_file)

df = pd.read_csv(csv_file)

print("="*60)
print("PyGesture Studio")
print("="*60)

print()
print("Samples :", len(df))
print("Classes :")
print(df["class"].value_counts().sort_index())
print()

# ==========================================
# Prepare data
# ==========================================

X = df.drop(columns=["class"])
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================
# Models
# ==========================================

models = {

    "KNN":
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier())
        ]),

    "SVM":
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC())
        ]),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            random_state=42
        ),

    "Extra Trees":
        ExtraTreesClassifier(
            n_estimators=300,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            random_state=42
        ),

    "MLP":
        Pipeline([
            ("scaler", StandardScaler()),
            ("model",
             MLPClassifier(
                 hidden_layer_sizes=(128,64),
                 max_iter=1000,
                 random_state=42
             ))
        ])
}

# ==========================================
# Train
# ==========================================

results = {}

best_name = None
best_model = None
best_acc = 0

print("Training...\n")

start = time.time()

for name, model in models.items():

    print(f"Training {name}...")

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    results[name] = acc

    print(f"Accuracy = {acc:.4f}\n")

    if acc > best_acc:

        best_acc = acc
        best_model = model
        best_name = name

elapsed = time.time() - start

# ==========================================
# Save model
# ==========================================

os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/gesture_model.joblib")

# ==========================================
# Ranking
# ==========================================

print("="*60)
print("RESULTS")
print("="*60)

ranking = sorted(
    results.items(),
    key=lambda x: x[1],
    reverse=True
)

for i, (name, acc) in enumerate(ranking, start=1):

    print(f"{i}. {name:20s} {acc:.4f}")

print()

print("="*60)
print("BEST MODEL")
print("="*60)

print("Model    :", best_name)
print("Accuracy :", round(best_acc*100,2), "%")
print("Time     :", round(elapsed,2), "seconds")
print()

print("Saved as:")
print("models/gesture_model.joblib")
print("="*60)
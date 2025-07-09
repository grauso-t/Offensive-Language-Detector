import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import joblib
import json
import os

# Load dataset
df = pd.read_csv("dataset/cleaned_balanced_dataset.csv")

# Split dataset in training and test set
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["category"], test_size=0.2, random_state=42)

# Create a pipeline with TF-IDF vectorization and logistic regression
from sklearn.pipeline import Pipeline

model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000))
])

# Fit the model
model.fit(X_train, y_train)

def evaluate_model(model, X_test, y_test, output_dir="dataset"):
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    precision_weighted = precision_score(y_test, y_pred, average='weighted')
    recall_weighted = recall_score(y_test, y_pred, average='weighted')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    # Print metrics
    print("== Classification Report ==")
    print(classification_report(y_test, y_pred))
    print("== Metriche aggiuntive ==")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Recall (macro): {recall_macro:.4f}")
    print(f"F1-score (macro): {f1_macro:.4f}")
    print(f"Precision (weighted): {precision_weighted:.4f}")
    print(f"Recall (weighted): {recall_weighted:.4f}")
    print(f"F1-score (weighted): {f1_weighted:.4f}")

    # Save metrics
    results = {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "model_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Custom labels for confusion matrix
    class_labels = ["Omofobia", "Sessismo", "Razzismo"]

    # Confusion matrix with custom labels
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title("Matrice di Confusione")
    plt.tight_layout()

    # Save confusion matrix image
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Matrice di confusione salvata in: {cm_path}")
    print(f"Metriche salvate in: {os.path.join(output_dir, 'model_metrics.json')}")

# Evaluate model
evaluate_model(model, X_test, y_test)

# Save model
joblib.dump(model, 'dataset/logistic_regression_model.pkl')
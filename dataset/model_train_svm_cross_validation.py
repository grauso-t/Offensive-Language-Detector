import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
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

# Output directory for all results
output_dir = "dataset/svm_cross_validation_results"

# Load dataset
df = pd.read_csv("dataset/cleaned_balanced_dataset.csv")

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["category"], test_size=0.2, random_state=42)

# Define a pipeline with TF-IDF vectorization and SVM classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('svc', SVC())
])

# Define hyperparameter grid for GridSearchCV
param_grid = {
    'svc__C': [0.1, 1, 10],
    'svc__kernel': ['linear', 'rbf'],
    'svc__gamma': ['scale', 'auto']
}

# Perform grid search with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='f1_macro',
    cv=5,
    verbose=1,
    n_jobs=-1,
    return_train_score=True
)

# Fit the model
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_
print(f"Best hyperparameters: {grid_search.best_params_}")

# Define function to evaluate the model and save metrics
def evaluate_model(model, X_test, y_test, output_dir):
    """
    Function to evaluate the model using different metrics.
    """
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    precision_weighted = precision_score(y_test, y_pred, average='weighted')
    recall_weighted = recall_score(y_test, y_pred, average='weighted')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    # Generate classification report as dict
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    # Print metrics
    print("== Classification Report ==")
    print(classification_report(y_test, y_pred))
    print("== Additional Metrics ==")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Recall (macro): {recall_macro:.4f}")
    print(f"F1-score (macro): {f1_macro:.4f}")
    print(f"Precision (weighted): {precision_weighted:.4f}")
    print(f"Recall (weighted): {recall_weighted:.4f}")
    print(f"F1-score (weighted): {f1_weighted:.4f}")

    # Save metrics
    os.makedirs(output_dir, exist_ok=True)
    results = {
        "best_hyperparameters": grid_search.best_params_,
        "best_validation_score": grid_search.best_score_,
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "classification_report": report_dict
    }

    with open(os.path.join(output_dir, "svm_model_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Custom class labels
    class_labels = ["Homophobia", "Sexism", "Racism"]

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()

    cm_path = os.path.join(output_dir, "svm_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    print(f"Confusion matrix saved to: {cm_path}")
    print(f"Metrics saved to: {os.path.join(output_dir, 'svm_model_metrics.json')}")

# Evaluate model
evaluate_model(best_model, X_test, y_test, output_dir=output_dir)

# Save the best model
model_path = os.path.join(output_dir, "svm_model.pkl")
joblib.dump(best_model, model_path)
print(f"Model saved to: {model_path}")

# Overfitting/underfitting plot
cv_results = pd.DataFrame(grid_search.cv_results_)

plt.figure(figsize=(12, 6))
plt.plot(cv_results['mean_train_score'], label='Mean Train F1 (macro)', marker='o')
plt.plot(cv_results['mean_test_score'], label='Mean Validation F1 (macro)', marker='o')
plt.xlabel("Hyperparameter Combination Index")
plt.ylabel("F1 Score (macro)")
plt.title("Training vs Validation Score - Overfitting Analysis")
plt.legend()
plt.grid(True)

overfit_path = os.path.join(output_dir, "overfitting_plot.png")
plt.savefig(overfit_path)
plt.close()

print(f"Overfitting plot saved to: {overfit_path}")

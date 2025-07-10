import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import joblib
import json
import os
import shap
import numpy as np

# Output directory
output_dir = "dataset/svm_linear_results"
os.makedirs(output_dir, exist_ok=True)

# Load dataset
df = pd.read_csv("dataset/cleaned_balanced_dataset.csv")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["category"], test_size=0.2, random_state=42)

# Define pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('clf', LinearSVC(max_iter=2000))
])

# Hyperparameter grid
param_grid = {
    'clf__C': [0.01, 0.1, 1, 10]
}

# Grid search
grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    scoring='f1_macro',
    cv=5,
    verbose=1,
    n_jobs=-1,
    return_train_score=True
)

# Train
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print(f"Best hyperparameters: {grid_search.best_params_}")

# Evaluation
def evaluate_model(model, X_test, y_test, output_dir):
    """
    Function to evaluate the model using different metrics.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "best_hyperparameters": grid_search.best_params_,
        "best_validation_score": grid_search.best_score_,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average='macro'),
        "recall_macro": recall_score(y_test, y_pred, average='macro'),
        "f1_macro": f1_score(y_test, y_pred, average='macro'),
        "precision_weighted": precision_score(y_test, y_pred, average='weighted'),
        "recall_weighted": recall_score(y_test, y_pred, average='weighted'),
        "f1_weighted": f1_score(y_test, y_pred, average='weighted'),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }

    # Save metrics
    with open(os.path.join(output_dir, "svm_linear_model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    class_labels = ["Omofobia", "Sessismo", "Razzismo"]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap='Purples', xticks_rotation=45)
    plt.title("Confusion Matrix - SVM Linear")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "svm_linear_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    print(f"Confusion matrix saved to: {cm_path}")

# SHAP
def generate_shap_plots(model, X_test, output_dir, num_samples=100):
    """
    Function to plot SHAP diagrams.
    """
    tfidf = model.named_steps['tfidf']
    clf = model.named_steps['clf']

    X_sample = X_test.sample(n=num_samples, random_state=42)
    X_sample_tfidf = tfidf.transform(X_sample)
    X_sample_dense = X_sample_tfidf.toarray()

    explainer = shap.LinearExplainer(clf, X_sample_dense, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_sample_dense)

    if isinstance(shap_values, list):
        shap_values_per_class = shap_values
    else:
        shap_values_per_class = [shap_values[:, :, i] for i in range(shap_values.shape[2])]

    feature_names = tfidf.get_feature_names_out()
    class_labels = ["Omofobia", "Sessismo", "Razzismo"]

    for i, class_label in enumerate(class_labels):
        print(f"Plotting SHAP summary for class '{class_label}'")
        shap.summary_plot(
            shap_values_per_class[i],
            X_sample_dense,
            feature_names=feature_names,
            show=False
        )
        shap_path = os.path.join(output_dir, f"svm_linear_shap_summary_class_{class_label}.png")
        plt.savefig(shap_path)
        plt.close()
        print(f"Saved SHAP plot to: {shap_path}")

# Save model
model_path = os.path.join(output_dir, "svm_linear_model.pkl")
joblib.dump(best_model, model_path)
print(f"SVM model saved to {model_path}")

# Evaluate model
evaluate_model(best_model, X_test, y_test, output_dir)

# Overfitting analysis
cv_results = pd.DataFrame(grid_search.cv_results_)
plt.figure(figsize=(12, 6))
plt.plot(cv_results['mean_train_score'], label='Mean Train F1 (macro)', marker='o')
plt.plot(cv_results['mean_test_score'], label='Mean Validation F1 (macro)', marker='o')
plt.xlabel("Hyperparameter Combination Index")
plt.ylabel("F1 Score (macro)")
plt.title("Training vs Validation Score - SVM Linear Overfitting Analysis")
plt.legend()
plt.grid(True)
overfit_path = os.path.join(output_dir, "svm_linear_overfitting_plot.png")
plt.savefig(overfit_path)
plt.close()
print(f"Overfitting plot saved to: {overfit_path}")

# SHAP
generate_shap_plots(best_model, X_test, output_dir)
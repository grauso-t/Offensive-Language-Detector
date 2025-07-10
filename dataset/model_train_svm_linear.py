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

from sklearn.decomposition import PCA

def plot_pca_2d(model, X, y, output_dir, model_name):
    """
    Plot 2D PCA projection of TF-IDF features from the given pipeline model.

    Args:
        model: Trained sklearn pipeline with 'tfidf' step.
        X: Input raw texts (pandas Series or list).
        y: Numeric class labels corresponding to X.
        output_dir: Directory to save the plot.
        model_name: String name of the model (for title and filename).
    """
    from sklearn.decomposition import PCA

    tfidf = model.named_steps['tfidf']
    X_tfidf = tfidf.transform(X)
    X_dense = X_tfidf.toarray() if hasattr(X_tfidf, "toarray") else X_tfidf

    print(f"Number of samples: {len(y)}")
    print(f"TF-IDF dense shape: {X_dense.shape}")

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_dense)
    print("Explained variance ratio by PCA components:", pca.explained_variance_ratio_)

    label_map = {0: "Omofobia", 1: "Sessismo", 2: "Razzismo"}

    # Filter valid labels and map to names
    mask = y.isin(label_map.keys())
    if mask.sum() == 0:
        print("ERROR: No valid labels found after mapping. Cannot plot PCA.")
        return

    X_pca_filtered = X_pca[mask.values]
    y_filtered = y[mask].map(label_map)

    # Define colors explicitly per class (must be in same order as labels)
    class_labels = sorted(label_map.values())  # e.g. ["Omofobia", "Razzismo", "Sessismo"]
    # Map class labels to colors manually
    colors_map = {
        "Omofobia": "#E41A1C",  # red
        "Sessismo": "#377EB8",  # blue
        "Razzismo": "#4DAF4A"   # green
    }

    plt.figure(figsize=(8, 6))

    # Plot each class separately for correct color and label
    for class_label in class_labels:
        indices = y_filtered == class_label
        plt.scatter(
            X_pca_filtered[indices, 0],
            X_pca_filtered[indices, 1],
            c=colors_map[class_label],
            label=class_label,
            alpha=0.7,
            s=60
        )

    plt.title(f"PCA 2D Projection of TF-IDF Space ({model_name})")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Classe", loc='best')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"pca_2d_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"PCA 2D plot saved at: {plot_path}")

plot_pca_2d(best_model, X_test, y_test, output_dir, model_name="SVM Linear")
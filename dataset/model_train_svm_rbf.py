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
from scipy.sparse import lil_matrix
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
from sklearn.decomposition import PCA

# Output directory for all results
output_dir = "dataset/svm_rbf_results"

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

    with open(os.path.join(output_dir, "svm_rbf_model_metrics.json"), "w") as f:
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

    cm_path = os.path.join(output_dir, "svm_rbf_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    print(f"Confusion matrix saved to: {cm_path}")
    print(f"Metrics saved to: {os.path.join(output_dir, 'svm_rbf_model_metrics.json')}")

# Evaluate model
evaluate_model(best_model, X_test, y_test, output_dir=output_dir)

# Save the best model
model_path = os.path.join(output_dir, "svm_rbf_model.pkl")
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

def analyze_permutation_importance_by_class(model, X_test, y_test, output_dir):
    """
    Computes permutation importance separately for each class (Homophobia, Sexism, Racism)
    and plots the top features.
    """

    print("\n== Class-wise Permutation Feature Importance ==")

    feature_names = model.named_steps['tfidf'].get_feature_names_out()
    X_test_vect = model.named_steps['tfidf'].transform(X_test)
    classifier = model.named_steps['svc']

    # Get original predictions
    original_preds = classifier.predict(X_test_vect)
    base_scores = f1_score(y_test, original_preds, average=None)

    class_labels = ["Omofobia", "Sessismo", "Razzismo"]
    num_classes = len(class_labels)
    top_k = 15

    # Initialize importances matrix
    class_importances = np.zeros((num_classes, X_test_vect.shape[1]))

    print("Computing permutation importance... (this may take a bit)")
    rng = np.random.RandomState(42)

    for feature_idx in tqdm(range(X_test_vect.shape[1]), desc="Permutation Importance"):
        # Copy test features
        X_permuted = lil_matrix(X_test_vect)

        # Permute one feature across all samples
        column = X_permuted[:, feature_idx].toarray().flatten()
        rng.shuffle(column)
        X_permuted[:, feature_idx] = column.reshape(-1, 1)

        # Get predictions with permuted data
        permuted_preds = classifier.predict(X_permuted)

        # Calculate class-wise F1
        permuted_scores = f1_score(y_test, permuted_preds, average=None)

        # Importance = drop in performance
        score_diff = base_scores - permuted_scores
        class_importances[:, feature_idx] = score_diff

    # Plot top features for each class
    for class_idx, class_name in enumerate(class_labels):
        sorted_idx = np.argsort(class_importances[class_idx])[::-1][:top_k]
        top_features = [feature_names[i] for i in sorted_idx]
        importances = class_importances[class_idx][sorted_idx]

        plt.figure(figsize=(10, 6))
        plt.barh(range(top_k), importances[::-1], align='center')
        plt.yticks(range(top_k), top_features[::-1])
        plt.xlabel("F1-score Drop (Permutation Importance)")
        plt.title(f"Top {top_k} Important Words for Class: {class_name}")
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f"permutation_importance_{class_name.lower()}.png")
        plt.savefig(plot_path)
        plt.close()

        print(f"[{class_name}] Permutation plot saved to: {plot_path}")

analyze_permutation_importance_by_class(best_model, X_test, y_test, output_dir=output_dir)

def plot_pca_2d(model, X, y, output_dir, model_name):
    """
    Plot 2D PCA projection of TF-IDF features from the given pipeline model,
    using only matplotlib (no seaborn).
    """
    # Extract TF-IDF transformer from pipeline
    tfidf = model.named_steps['tfidf']
    # Transform raw texts to TF-IDF vectors
    X_tfidf = tfidf.transform(X)

    # Convert sparse matrix to dense for PCA
    X_dense = X_tfidf.toarray() if hasattr(X_tfidf, "toarray") else X_tfidf

    # Fit PCA to reduce to 2D
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_dense)

    # Map numeric labels to meaningful names and colors
    label_map = {0: "Omofobia", 1: "Sessismo", 2: "Razzismo"}
    color_map = {0: 'red', 1: 'green', 2: 'blue'}

    plt.figure(figsize=(8, 6))

    # Plot each class separately for legend and colors
    for label in np.unique(y):
        indices = (y == label)
        plt.scatter(
            X_pca[indices, 0],
            X_pca[indices, 1],
            c=color_map[label],
            label=label_map[label],
            alpha=0.7,
            s=60,
            edgecolors='w',
            linewidth=0.5
        )

    plt.title(f"PCA 2D Projection of TF-IDF Space ({model_name})")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Classe", loc='best')
    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"pca_2d_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"PCA 2D plot saved at: {plot_path}")

plot_pca_2d(best_model, X_test, y_test.values, output_dir="dataset/svm_rbf_results", model_name="SVM RBF")

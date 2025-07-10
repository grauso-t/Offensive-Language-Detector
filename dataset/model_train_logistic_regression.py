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
import shap

# Load dataset
df = pd.read_csv("dataset/cleaned_balanced_dataset.csv")

# Split dataset in training and test set
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["category"], test_size=0.2, random_state=42)

# Create a pipeline with TF-IDF vectorization and logistic regression
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs'))
])

# Fit the model
model.fit(X_train, y_train)

def evaluate_model(model, X_test, y_test, output_dir="dataset/logistic_regression_results"):
    """
    Function to evaluate a trained model, save classification metrics, confusion matrix,
    and generate SHAP summary plots for model interpretability.
    """
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    precision_weighted = precision_score(y_test, y_pred, average='weighted')
    recall_weighted = recall_score(y_test, y_pred, average='weighted')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    # Generate classification report as dictionary
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    # Print all metrics to console
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

    # Save metrics to file
    os.makedirs(output_dir, exist_ok=True)
    results = {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "classification_report": report_dict
    }
    with open(os.path.join(output_dir, "model_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Define Italian labels for confusion matrix and SHAP
    class_labels = ["Omofobia", "Sessismo", "Razzismo"]
    class_indices = [0, 1, 2]

    # Plot and save confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=class_indices)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title("Matrice di Confusione")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to: {cm_path}")
    print(f"Metrics saved to: {os.path.join(output_dir, 'model_metrics.json')}")

def generate_shap_plots(model, X_test, output_dir="dataset/logistic_regression_results", num_samples=100):

    tfidf = model.named_steps['tfidf']
    clf = model.named_steps['clf']

    X_sample = X_test.sample(n=num_samples, random_state=42)
    X_sample_tfidf = tfidf.transform(X_sample)
    X_sample_dense = X_sample_tfidf.toarray()
    
    print(f"Shape dati (X_sample_dense): {X_sample_dense.shape}")  # (100, 5000)

    explainer = shap.LinearExplainer(clf, X_sample_dense, feature_perturbation="interventional")
    
    shap_values = explainer.shap_values(X_sample_dense)

    # Check type and shape
    print(f"Tipo shap_values: {type(shap_values)}")
    print(f"Shape shap_values: {shap_values.shape}")

    # Transform shap_values in class list
    shap_values_per_class = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
    
    class_labels = ["Omofobia", "Sessismo", "Razzismo"]

    os.makedirs(output_dir, exist_ok=True)
    for i, class_label in enumerate(class_labels):
        print(f"Plotting SHAP summary for class '{class_label}'")
        shap.summary_plot(
            shap_values_per_class[i],       # (num_samples, num_features)
            X_sample_dense,
            feature_names=tfidf.get_feature_names_out(),
            show=False
        )
        plt.savefig(os.path.join(output_dir, f"shap_summary_class_{class_label}.png"))
        plt.close()
    print("SHAP plots saved.")

# Evaluate model
evaluate_model(model, X_test, y_test)

# Save model
joblib.dump(model, 'dataset/logistic_regression_results/logistic_regression_model.pkl')

generate_shap_plots(model, X_test, output_dir="dataset/logistic_regression_results", num_samples=100)


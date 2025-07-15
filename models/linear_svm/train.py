import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import joblib
import os
import json

# Load dataset and filter
df = pd.read_csv('dataset/cleaned_balanced_dataset.csv')
df = df[df["category"] != 3]

print(f"Dataset loaded: {len(df)} rows")

# Plot class distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='category', palette='viridis')
plt.title("Category Distribution")
plt.xlabel("Category")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Prepare features and labels
X = df['text']
y = df['category']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    max_features=1000,
    stop_words='english',
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"Number of features: {X_train_vec.shape[1]}")

# Train Linear SVM model
model = LinearSVC(random_state=42, max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=model.classes_, cmap="Blues", xticks_rotation=45
)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Create directories for saving
os.makedirs('models/linear_svm', exist_ok=True)
results_dir = 'models/linear_svm/results'
os.makedirs(results_dir, exist_ok=True)

# Save model and vectorizer
joblib.dump(model, 'models/linear_svm/model.pkl')
joblib.dump(vectorizer, 'models/linear_svm/vectorizer.pkl')

# Save evaluation results to JSON
results = {
    "accuracy": accuracy,
    "classification_report": classification_report(y_test, y_pred, output_dict=True)
}

with open(os.path.join(results_dir, 'evaluation_results.json'), 'w') as f:
    json.dump(results, f, indent=4)

# Show and save top features for each class
feature_names = vectorizer.get_feature_names_out()

for i, category in enumerate(model.classes_):
    top_indices = model.coef_[i].argsort()[-10:][::-1]
    top_words = [feature_names[idx] for idx in top_indices]
    weights = model.coef_[i][top_indices]

    plt.figure(figsize=(8, 4))
    sns.barplot(x=weights, y=top_words, palette="rocket")
    plt.title(f"Top Words for Category {category}")
    plt.xlabel("Coefficient Weight")
    plt.ylabel("Word")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'top_words_category_{category}.png'), format='png')
    plt.close()
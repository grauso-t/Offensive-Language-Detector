import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

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

# Evaluate the model on the test set
y_pred = model.predict(X_test)
print("Risultati sul set di test:")
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(model, 'dataset/logistic_regression_model.pkl')
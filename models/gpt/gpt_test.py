import pandas as pd
import random
import os
import json
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from openai import OpenAI
from dotenv import load_dotenv
import math

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=api_key, base_url="https://api.gpt4-all.xyz/v1")

# Load the dataset
df = pd.read_csv("dataset/cleaned_balanced_dataset.csv")

# Map class numbers to class names
class_map = {
    0: "homophobia",
    1: "sexism",
    2: "racism",
    3: "no offensive content"
}

# Reverse mapping from class names to numbers
inv_class_map = {v: k for k, v in class_map.items()}

# Take 20 random examples from each category
sampled_df = df.groupby("category").apply(lambda x: x.sample(20, random_state=42)).reset_index(drop=True)

def is_offensive_gpt(text):
    """
    Function to classify a sentence using GPT
    Returns one of the following categories:
    'no offensive content', 'sexism', 'racism', 'homophobia'
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an offensive sentence classifier. Respond only with one of these words: 'no offensive content', 'sexism', 'racism', 'homophobia'."
                },
                {"role": "user", "content": f"Sentence: {text}"}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        print(f"Error during API call: {e}")
        return "error"
    finally:
        time.sleep(5)

# Classify each sentence in batches
texts = sampled_df["text"].tolist()
true_labels = sampled_df["category"].tolist()

predicted_labels_text = []
batch_size = 5
pause_duration = 80
num_batches = math.ceil(len(texts) / batch_size)

for batch_index in range(num_batches):
    start = batch_index * batch_size
    end = min(start + batch_size, len(texts))
    print(f"\n--- Starting batch {batch_index + 1} of {num_batches} ---")

    for idx in range(start, end):
        text = texts[idx]
        true_label = class_map[true_labels[idx]]
        print(f"\nAnalyzing sentence {idx + 1} of {len(texts)}")
        print(f"Sentence: {text}")
        print(f"True label: {true_label}")

        label = is_offensive_gpt(text)
        predicted_labels_text.append(label)

        print(f"Model response: {label}")

    if batch_index < num_batches - 1:
        print(f"\nWaiting {pause_duration} seconds before next batch...")
        time.sleep(pause_duration)

# Convert predicted labels to numbers
predicted_labels = [inv_class_map.get(label, -1) for label in predicted_labels_text]

# Remove invalid predictions
filtered_true, filtered_pred = [], []
for t, p in zip(true_labels, predicted_labels):
    if p != -1:
        filtered_true.append(t)
        filtered_pred.append(p)

# Calculate evaluation metrics
accuracy = accuracy_score(filtered_true, filtered_pred)
report_dict = classification_report(filtered_true, filtered_pred, target_names=list(class_map.values()), output_dict=True)
conf_matrix = confusion_matrix(filtered_true, filtered_pred).tolist()

# Prepare results dictionary
results = {
    "accuracy": accuracy,
    "classification_report": report_dict,
    "confusion_matrix": conf_matrix,
    "labels": list(class_map.values())
}

# Create result folder if it does not exist
os.makedirs("models/gpt/result", exist_ok=True)

# Save results to JSON file
with open("models/gpt/result/evaluation_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nResults saved to models/gpt/result/evaluation_results.json")

# Save confusion matrix as image
conf_matrix_np = confusion_matrix(filtered_true, filtered_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_np, annot=True, fmt="d", cmap="Blues", xticklabels=list(class_map.values()), yticklabels=list(class_map.values()))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - GPT Classification")
plt.tight_layout()
plt.savefig("models/gpt/result/confusion_matrix.png")
plt.close()

print("Confusion matrix image saved to models/gpt/result/confusion_matrix.png")
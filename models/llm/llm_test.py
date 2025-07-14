import pandas as pd
import random
import os
import json
import time
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from ctransformers import AutoModelForCausalLM
import torch

# Use GPU if available
gpu = 20 if not torch.cuda.is_available() else 0

# Load the language model
llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Mistral-7B-v0.1-GGUF",
    model_file="mistral-7b-v0.1.Q4_K_M.gguf",
    model_type="mistral",
    gpu_layers=gpu
)

# Load the dataset
df = pd.read_csv("dataset/cleaned_balanced_dataset.csv")

# Class mapping
class_map = {
    0: "homophobia",
    1: "sexism",
    2: "racism",
    3: "no offensive content"
}

inv_class_map = {v: k for k, v in class_map.items()}
valid_labels = set(inv_class_map.keys())

# Sample 5 examples per class
sampled_df = df.groupby("category").apply(lambda x: x.sample(5, random_state=42)).reset_index(drop=True)

def clean_output(output):
    """
    Extract a clean class label from the model's output.
    Returns lowercase label string if valid, otherwise returns None.
    """
    output = output.strip().lower()

    # Extract only valid class labels
    for label in inv_class_map:
        if re.search(rf"\b{re.escape(label)}\b", output):
            return label
    return None

def is_offensive_llm(text):
    """
    Use the language model to classify the sentence.
    """
    prompt = f"""You are an offensive sentence classifier.
                Classify the following sentence into one of the following categories:
                'no offensive content', 'sexism', 'racism', or 'homophobia'.

                Respond with just one of these categories.

                Sentence: {text}
                Category:"""

    try:
        output = llm(prompt, max_new_tokens=10)
        return clean_output(output)
    except Exception as e:
        print(f"Model error: {e}")
        return None

# Classify sentences
texts = sampled_df["text"].tolist()
true_labels = sampled_df["category"].tolist()

predicted_labels_text = []
for idx, text in enumerate(texts):
    true_label = class_map[true_labels[idx]]
    print(f"\nAnalyzing sentence {idx + 1} of {len(texts)}")
    print(f"Sentence: {text}")
    print(f"True label: {true_label}")

    label = is_offensive_llm(text)
    print(f"Model response: {label}")
    predicted_labels_text.append(label if label in inv_class_map else None)
    time.sleep(5)

# Convert to numeric labels
predicted_labels = [inv_class_map.get(label, -1) if label else -1 for label in predicted_labels_text]

# Filter out invalid predictions
filtered_true, filtered_pred = [], []
for t, p in zip(true_labels, predicted_labels):
    if p != -1:
        filtered_true.append(t)
        filtered_pred.append(p)

# Metrics
accuracy = accuracy_score(filtered_true, filtered_pred)
report_dict = classification_report(filtered_true, filtered_pred, target_names=list(class_map.values()), output_dict=True)
conf_matrix = confusion_matrix(filtered_true, filtered_pred).tolist()

# Save results
results = {
    "accuracy": accuracy,
    "classification_report": report_dict,
    "confusion_matrix": conf_matrix,
    "labels": list(class_map.values())
}

output_dir = "models/llm/result"
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
    json.dump(results, f, indent=4)

print(f"\nResults saved to {os.path.join(output_dir, 'evaluation_results.json')}")

# Save confusion matrix as image
conf_matrix_np = confusion_matrix(filtered_true, filtered_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_np, annot=True, fmt="d", cmap="Blues", xticklabels=list(class_map.values()), yticklabels=list(class_map.values()))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - GPT Classification")
plt.tight_layout()
plt.savefig("models/llm/result/confusion_matrix.png")
plt.close()

print("Confusion matrix image saved to models/gpt/result/confusion_matrix.png")
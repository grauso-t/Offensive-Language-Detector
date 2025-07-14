import pandas as pd
import random
import os
import json
import time
from sklearn.metrics import accuracy_score, classification_report
from ctransformers import AutoModelForCausalLM
import torch

# use GPU if available
gpu = 20 if not torch.cuda.is_available() else 0

# load the language model
llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Mistral-7B-v0.1-GGUF",
    model_file="mistral-7b-v0.1.Q4_K_M.gguf",
    model_type="mistral",
    gpu_layers=gpu
)

# load the dataset
df = pd.read_csv("dataset/cleaned_balanced_dataset.csv")

# map class numbers to class names
class_map = {
    0: "homophobia",
    1: "sexism",
    2: "racism",
    3: "no offensive content"
}

# create a reverse map from class names to class numbers
inv_class_map = {v: k for k, v in class_map.items()}

# take 5 random examples from each category
sampled_df = df.groupby("category").apply(lambda x: x.sample(5, random_state=42)).reset_index(drop=True)

def is_offensive_llm(text):
    """
    Function to classify a sentence using the language model
    returns one of the following categories: no offensive content, sexism, racism, or homophobia
    """

    prompt = f"""You are an offensive sentence classifier.
    Classify the following sentence into one of the following categories:
    'no offensive content', 'sexism', 'racism', or 'homophobia'.

    Respond with just one of these categories.

    Sentence: {text}
    Category:"""

    output = llm(prompt, max_new_tokens=10)
    return output.strip().lower()

# classify each sampled sentence
texts = sampled_df["text"].tolist()
true_labels = sampled_df["category"].tolist()

predicted_labels_text = []
for idx, text in enumerate(texts):
    true_label = class_map[true_labels[idx]]
    print(f"\nAnalyzing sentence {idx + 1} of {len(texts)}")
    print(f"Sentence: {text}")
    print(f"True label: {true_label}")
    label = is_offensive_llm(text)
    predicted_labels_text.append(label)
    print(f"Model response: {label}")
    time.sleep(10)

# convert predicted labels from text to numbers
predicted_labels = [inv_class_map.get(label, -1) for label in predicted_labels_text]

# remove invalid predictions
filtered_true, filtered_pred = [], []
for t, p in zip(true_labels, predicted_labels):
    if p != -1:
        filtered_true.append(t)
        filtered_pred.append(p)

# calculate accuracy and classification report
accuracy = accuracy_score(filtered_true, filtered_pred)
report_dict = classification_report(filtered_true, filtered_pred, output_dict=True)

# save the results to a file
results = {
    "accuracy": accuracy,
    "classification_report": report_dict
}

output_path = "models/llm/result/evaluation_results.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"\nResults saved to {output_path}")
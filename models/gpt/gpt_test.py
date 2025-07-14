import pandas as pd
import random
import os
import json
import time
from sklearn.metrics import accuracy_score, classification_report
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=api_key, base_url="https://api.gpt4-all.xyz/v1")

# load the dataset
df = pd.read_csv("dataset/cleaned_balanced_dataset.csv")

# map class numbers to class names
class_map = {
    0: "homophobia",
    1: "sexism",
    2: "racism",
    3: "no offensive content"
}
# reverse mapping from class names to numbers
inv_class_map = {v: k for k, v in class_map.items()}

# take 5 random examples from each category
sampled_df = df.groupby("category").apply(lambda x: x.sample(5, random_state=42)).reset_index(drop=True)

def is_offensive_gpt(text):
    """
    Function to classify a sentence using GPT
    returns one of the following categories: no offensive content, sexism, racism, homophobia
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

# classify each sentence
texts = sampled_df["text"].tolist()
true_labels = sampled_df["category"].tolist()

predicted_labels_text = []
for idx, text in enumerate(texts):
    true_label = class_map[true_labels[idx]]
    print(f"\nAnalyzing sentence {idx + 1} of {len(texts)}")
    print(f"Sentence: {text}")
    print(f"True label: {true_label}")
    
    label = is_offensive_gpt(text)
    predicted_labels_text.append(label)
    
    print(f"Model response: {label}")


# convert predicted labels to numbers
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

# prepare results dictionary
results = {
    "accuracy": accuracy,
    "classification_report": report_dict
}

# create result folder if it does not exist
os.makedirs("models/gpt/result", exist_ok=True)

# save results to JSON file
with open("models/gpt/result/evaluation_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nResults saved to models/gpt/result/evaluation_results.json")
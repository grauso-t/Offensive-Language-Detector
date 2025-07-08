import pandas as pd
import os
import re
import emoji
import unicodedata

# Define base path for dataset folders
base_path = 'dataset'

# Function to clean the text: remove mentions, emojis, weird characters, and extra spaces
def clean_text(text):
    if pd.isnull(text):
        return ""
    
    # Remove mentions like @user
    text = re.sub(r"@\w+", "", text)

    # Remove links
    text = re.sub(r"http\S+|www.\S+", "", text)

    # Remove emojis
    text = emoji.replace_emoji(text, replace='')

    # Remove non-ASCII characters (e.g. Arabic, symbols)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    # Remove special characters, keeping basic punctuation
    text = re.sub(r"[^a-zA-Z0-9\s.,!?'\"]+", "", text)

    # Collapse multiple spaces/newlines into one space
    text = re.sub(r"\s+", " ", text).strip()

    return text

# Load Homophobia dataset
homophobia_path = os.path.join(base_path, 'archive_homophobia', 'homophobiaDatasetAnonymous.csv')
homophobia_df = pd.read_csv(homophobia_path, header=None, names=["label", "text", "lang"])
homophobia_df = homophobia_df[homophobia_df["lang"] == "en"]
homophobia_df = homophobia_df[["label", "text"]]
homophobia_df["category"] = "homophobia"

# Load Racism dataset
racism_path = os.path.join(base_path, 'archive_racism', 'twitter_racism_parsed_dataset.csv')
racism_df = pd.read_csv(racism_path)
racism_df = racism_df[["oh_label", "Text"]].rename(columns={"oh_label": "label", "Text": "text"})
racism_df["category"] = "racism"

# Load Sexism dataset
sexism_path = os.path.join(base_path, 'archive_sexism', 'dev.csv')
sexism_df = pd.read_csv(sexism_path)
sexism_df = sexism_df[["label_sexist", "text"]]
sexism_df["label"] = sexism_df["label_sexist"].map({"sexist": 1, "not sexist": 0})
sexism_df = sexism_df.dropna(subset=["label"])
sexism_df = sexism_df[["label", "text"]]
sexism_df["category"] = "sexism"

# Merge all datasets
df = pd.concat([homophobia_df, racism_df, sexism_df], ignore_index=True)

# Remove rows with missing or duplicate text
df = df.dropna(subset=["text", "label"])
df = df.drop_duplicates(subset=["text"])
df["label"] = df["label"].astype(int)

# Find the smallest group size across all (category, label) combinations
group_counts = df.groupby(["category", "label"]).size()
min_per_group = group_counts.min()
print("Minimo comune per ciascun gruppo (categoria + label):", min_per_group)

# Balance the dataset by sampling equally from all (category, label) groups
balanced_df = (
    df.groupby(["category", "label"], group_keys=False)
    .apply(lambda x: x.sample(min_per_group, random_state=42))
    .reset_index(drop=True)
)

# Clean text column using the cleaning function
balanced_df["text"] = balanced_df["text"].apply(clean_text)

# Save the cleaned and balanced dataset
output_path = os.path.join(base_path, 'merged_fully_balanced.csv')
balanced_df.to_csv(output_path, index=False)

# Print some simple stats to verify the balance
print(f"\nDataset unificato e bilanciato salvato in: {output_path}\n")

print("Distribuzione per categoria:")
print(balanced_df["category"].value_counts())

print("\nDistribuzione per etichetta:")
print(balanced_df["label"].value_counts())

print("\nDistribuzione per categoria + etichetta:")
print(balanced_df.groupby(['category', 'label']).size().unstack())
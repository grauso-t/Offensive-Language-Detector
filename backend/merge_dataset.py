import pandas as pd
import os

# Define base path
base_path = 'dataset'

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

# Merge all three datasets into a single dataframe
df = pd.concat([homophobia_df, racism_df, sexism_df], ignore_index=True)

# Clean up by removing rows with missing text or labels
df = df.dropna(subset=["text", "label"])
# Remove duplicate texts to avoid repetition
df = df.drop_duplicates(subset=["text"])
# Make sure the labels are integers
df["label"] = df["label"].astype(int)

# Balance the dataset so that each combination of category and label (0 or 1) has the same number of examples
group_counts = df.groupby(["category", "label"]).size()
# Check the smallest group size among all category-label groups
min_per_group = group_counts.min()

print("Minimo comune per ciascun gruppo (categoria + label):", min_per_group)

# For each group, we randomly sample exactly that number of examples

balanced_df = (
    df.groupby(["category", "label"], group_keys=False)
    .apply(lambda x: x.sample(min_per_group, random_state=42))
    .reset_index(drop=True)
)

# Save the balanced combined dataset to a CSV file
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
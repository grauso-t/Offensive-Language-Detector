import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# Set save path
save_directory = "models/bert_trainer"
os.makedirs(save_directory, exist_ok=True)

# Load and filter dataset
df = pd.read_csv("dataset/cleaned_balanced_dataset.csv")
df = df[df['category'].isin([0, 1, 2, 3])]

print(f"Dataset loaded with {len(df)} rows")
print("Category distribution:\n", df['category'].value_counts(), "\n")

# Split dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(),
    df['category'].tolist(),
    test_size=0.2,
    random_state=42
)

print(f"Training samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}\n")

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# Custom torch Dataset
class HateDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = HateDataset(train_encodings, train_labels)
val_dataset = HateDataset(val_encodings, val_labels)

# Load model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)

# Training arguments
training_args = TrainingArguments(
    output_dir=f"{save_directory}/results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir=f"{save_directory}/logs",
    logging_steps=10,
    evaluation_strategy="epoch"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
print("Starting training...\n" + "="*50)
trainer.train()
print("\nTraining completed.\n" + "="*50)

# Save the model and tokenizer
trainer.save_model(save_directory)
tokenizer.save_pretrained(save_directory)
print(f"\nModel and tokenizer saved to '{save_directory}'")
import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import seaborn as sns

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'})")

# Paths
save_directory = "models/bert_trainer"
os.makedirs(save_directory, exist_ok=True)
results_dir = os.path.join(save_directory, "results")
os.makedirs(results_dir, exist_ok=True)

# Load and prepare data

df = pd.read_csv("dataset/cleaned_balanced_dataset.csv")
df = df[df['category'].isin([0,1,2,3])]
print(f"Dataset loaded with {len(df)} rows")
print("Category distribution:\n", df['category'].value_counts(), "\n")

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(),
    df['category'].tolist(),
    test_size=0.2,
    random_state=42
)

print(f"Training samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}\n")

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

class HateDataset(torch.utils.data.Dataset):
    '''
    Custom dataset class for tokenized texts and labels to be used with PyTorch DataLoader.
    '''
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        '''
        Returns the tokenized item and its label at index idx as tensors.
        '''
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        '''
        Returns the number of samples in the dataset.
        '''
        return len(self.labels)

train_dataset = HateDataset(train_encodings, train_labels)
val_dataset = HateDataset(val_encodings, val_labels)

# Model

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)
model.to(device)

# Metrics function

def compute_metrics(eval_pred):
    '''
    Compute accuracy, precision, recall and F1-score given predictions and labels.
    '''
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# TrainingArguments

training_args = TrainingArguments(
    output_dir=results_dir,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir=os.path.join(save_directory, "logs"),
    logging_steps=10,
    save_strategy="no",
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Training with manual loss tracking

train_loss_values = []
eval_loss_values = []

print("Starting training...\n" + "="*50)
for epoch in range(int(training_args.num_train_epochs)):
    print(f"Epoch {epoch+1}/{int(training_args.num_train_epochs)}")
    train_output = trainer.train()
    train_loss = train_output.training_loss
    train_loss_values.append(train_loss)
    
    eval_metrics = trainer.evaluate()
    eval_loss_values.append(eval_metrics["eval_loss"])
    
    print(f"Train loss: {train_loss:.4f} | Eval loss: {eval_metrics['eval_loss']:.4f}")
    print(f"Eval metrics: Accuracy: {eval_metrics['eval_accuracy']:.4f}, F1: {eval_metrics['eval_f1']:.4f}\n")

print("Training completed.\n" + "="*50)

# Manual saving of model and tokenizer

print("Evaluating best model on validation set manually...")

# Final evaluation on validation set
predictions_output = trainer.predict(val_dataset)
preds = np.argmax(predictions_output.predictions, axis=-1)
labels = predictions_output.label_ids

# Calculate final metrics
precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
acc = accuracy_score(labels, preds)
class_report = classification_report(labels, preds, digits=4)
conf_matrix = confusion_matrix(labels, preds)

print(f"Final Evaluation Metrics:\nAccuracy: {acc:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-score: {f1:.4f}\n")
print("Classification Report:\n", class_report)

# Save metrics to file
with open(os.path.join(results_dir, "evaluation_metrics.txt"), "w") as f:
    f.write(f"Accuracy: {acc:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-score: {f1:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(class_report)

# Save confusion matrix as image
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
plt.close()

# Save model and tokenizer
trainer.save_model(save_directory)
tokenizer.save_pretrained(save_directory)
print(f"Model and tokenizer saved to '{save_directory}'")

# Save training and validation loss plot

plt.figure(figsize=(10,6))
plt.plot(range(1, len(train_loss_values)+1), train_loss_values, label="Training Loss")
plt.plot(range(1, len(eval_loss_values)+1), eval_loss_values, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.savefig(os.path.join(results_dir, "loss_plot.png"))
plt.close()

print("Loss plot saved.")
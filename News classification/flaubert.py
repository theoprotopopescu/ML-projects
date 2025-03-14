# Required Libraries
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Load FlauBERT Model
model = AutoModelForSequenceClassification.from_pretrained("flaubert/flaubert_large_cased", num_labels=3)
# Load FlauBERT Tokenizer
tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_large_cased")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load and Preprocess Data
def load_and_preprocess_data(train_path):
    # Load data
    df = pd.read_csv(train_path)

    # Encode labels
    label_mapping = {"fake": 0, "biased": 1, "true": 2}
    df["Label"] = df["Label"].map(label_mapping)

    return df, label_mapping

train_file_path = "./sii/train.csv"
train_df, label_mapping = load_and_preprocess_data(train_file_path)

# Train-validation split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df["Text"].tolist(),
    train_df["Label"].tolist(),
    test_size=0.1,
    random_state=42
)

# Preprocessing Function
def preprocess_function(texts, labels=None):
    encodings = tokenizer(texts, padding="max_length", truncation=True, max_length=512)
    return Dataset.from_dict({"input_ids": encodings["input_ids"], "attention_mask": encodings["attention_mask"], "labels": labels})

# Preprocess Data
train_dataset = preprocess_function(train_texts, train_labels)
val_dataset = preprocess_function(val_texts, val_labels)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./flaubert",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=10,
    weight_decay=0.01,
    learning_rate=2e-5,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# Define Trainer
def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    labels = pred.label_ids
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the Model
trainer.train()

# Load and Predict on Test Data
def predict_test_data(test_path, label_mapping):
    test_df = pd.read_csv(test_path)
    test_dataset = preprocess_function(test_df["Text"].tolist())

    predictions = trainer.predict(test_dataset)
    predicted_labels = predictions.predictions.argmax(-1)

    reverse_label_mapping = {v: k for k, v in label_mapping.items()}
    test_df["Label"] = [reverse_label_mapping[label] for label in predicted_labels]

    test_df.to_csv("flaubert_results.csv", index=False)
    print("Predictions saved to flaubert_results.csv")

test_file_path = "./sii/test.csv"
predict_test_data(test_file_path, model, tokenizer, label_mapping)
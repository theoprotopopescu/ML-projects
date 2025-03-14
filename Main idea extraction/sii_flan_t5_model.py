import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pandas as pd

# 1. Load the Model and Tokenizer
model_name = "./flan-t5-cnn-dailymail/checkpoint-17945" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 2. Load the CNN/DailyMail Dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]
test_dataset = dataset["test"]

train_dataset = train_dataset.select(range(len(train_dataset) // 2, len(train_dataset)))

# 3. Define the Prompt and Preprocessing Function
def preprocess_function(examples):
    inputs = ["extract main idea: " + doc for doc in examples["article"]]
    targets = examples["highlights"]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
    labels = tokenizer(targets, max_length=150, truncation=True, padding="max_length", return_tensors="pt")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 4. Apply Preprocessing to the Dataset
train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=["article", "highlights", "id"])
eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=["article", "highlights", "id"])
test_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=["article", "highlights", "id"])

# 5. Define Training Arguments
training_args = TrainingArguments(
    output_dir="./flan-t5-cnn-dailymail",
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
    report_to="none",
)

# 6. Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=None, # We will compute metrics manually
)

# 7. Train the Model
trainer.train()

# 8. Save the Model
trainer.save_model("./sii_flan_t5")
tokenizer.save_pretrained("./sii_flan_t5")

print("Model saved")

# 9. Generate Summaries for the Entire Test Set
def generate_summaries_for_dataset(dataset):
    all_predictions = []
    all_labels = []
    all_articles = []
    for i in tqdm(range(len(dataset)), desc="Generating Summaries"):
        article = dataset[i]["article"]
        input_text = "extract main idea: " + article
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        outputs = model.generate(input_ids, max_length=150, num_beams=5, early_stopping=True)
        predicted_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        true_summary = tokenizer.decode(dataset[i]["labels"], skip_special_tokens=True)
        all_predictions.append(predicted_summary)
        all_labels.append(true_summary)
        all_articles.append(article)
    return all_articles, all_predictions, all_labels

all_articles, all_predictions, all_labels = generate_summaries_for_dataset(test_dataset)

# 10. Create DataFrame and Save to Excel
df = pd.DataFrame({
    "Article": all_articles,
    "Predicted Summary": all_predictions,
    "True Summary": all_labels,
})

df.to_csv("flan_t5_large_results.csv", index=False)
print("\nResults saved to flan_t5_large_results.csv")
import torch
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from trl import DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model, TaskType

# 1. Model & Tokenizer Setup
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# Load model in 8-bit precision for memory efficiency (requires bitsandbytes)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,             # LoRA rank
    lora_alpha=32,
    lora_dropout=0.1
)

# 2. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

model = get_peft_model(model, peft_config)
print("PEFT model loaded successfully.")

# output_dir = "./sii_llama3"

# model = AutoModelForCausalLM.from_pretrained(
#     output_dir,
#     device_map="auto",
# )
# tokenizer = AutoTokenizer.from_pretrained(output_dir)

# model = get_peft_model(model, peft_config)


# 3. Load CNN/DailyMail Dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")

train_dataset = dataset["train"]
eval_dataset  = dataset["validation"]

train_dataset = train_dataset.select(range(len(train_dataset) // 2, len(train_dataset)))
# eval_dataset = eval_dataset.select(range(100))

# 4. Prompt Template
QUERY_PROMPT = """### Instruction:
Extract the main idea from the following article:

### Article:
{article}
"""

ANSWER_PROMPT = """### Main idea:
{highlights}
"""


# 5. Preprocessing Function
def preprocess_function(example):
    messages = [
        {"role": "system", "content": "Answer only with the main idea."},
        {"role": "user", "content": QUERY_PROMPT.format(article=example["article"])},
        {"role": "assistant", "content": ANSWER_PROMPT.format(highlights=example["highlights"])},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    return {"input_ids": input_ids}


# 6. Map the preprocessing over training and validation sets
train_dataset = train_dataset.map(preprocess_function, batched=False, remove_columns=["article", "highlights", "id"])
eval_dataset = eval_dataset.map(preprocess_function, batched=False, remove_columns=["article", "highlights", "id"])

data_collator = DataCollatorForCompletionOnlyLM(
    tokenizer=tokenizer,
    mlm=False, 
    return_tensors="pt",
    response_template="<|start_header_id|>assistant<|end_header_id|>"
)

# 7. Training Arguments
training_args = TrainingArguments(
    output_dir="./llama3-1B-cnn-dailymail",
    per_device_train_batch_size=2,   
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=1,
    save_total_limit=1,             
    learning_rate=2e-5,
    weight_decay=0.01,
    optim="adamw_8bit",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    fp16=True,                      
    report_to="none",
)

# 8. Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)


# 9. Fine-Tune
train_result = trainer.train()

# 10. Save the Model
model.save_pretrained("./sii_llama3")
tokenizer.save_pretrained("./sii_llama3")
print("\nModel and tokenizer saved to ./sii_llama3")


test_dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")
# test_dataset = test_dataset.select(range(100))

# 11. Function for generation
def generate_summaries_for_dataset(dataset):
    all_articles = []
    all_predictions = []
    all_references = []

    for i in tqdm(range(len(dataset)), desc="Generating Summaries"):
        article = dataset[i]["article"]

        messages = [
            {"role": "user", "content": QUERY_PROMPT.format(article=article)},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        input_ids = tokenizer(prompt, max_length=1850, truncation=True, return_tensors='pt').to("cuda")

        outputs = model.generate(
            **input_ids, 
            max_new_tokens=150, 
            num_beams=5, 
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id
        )
        predicted_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

        all_predictions.append(predicted_summary)
        all_articles.append(article)
        all_references.append(dataset[i]["highlights"])

    return all_articles, all_predictions, all_references

all_articles, all_predictions, all_labels = generate_summaries_for_dataset(test_dataset)

# 12. Create DataFrame & Save to CSV
df = pd.DataFrame({
    "Article": all_articles,
    "Predicted Summary": all_predictions,
    "True Summary": all_labels,
})

df.to_csv("llama3_results.csv", index=False)
print("\nResults saved to llama3_results.csv")
import pandas as pd
import evaluate
from nltk.tokenize import sent_tokenize
from keybert import KeyBERT

import nltk
nltk.download('punkt_tab')

# Load the Excel file
df = pd.read_csv("flan_t5_large_results.csv")

# Extract columns for evaluation
all_articles = df["Article"].tolist()
all_predictions = df["Predicted Summary"].tolist()
all_labels = df["True Summary"].tolist()

# Load ROUGE metric
rouge = evaluate.load("rouge")

# Helper function to preprocess text for ROUGE
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # ROUGE expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels

# Post-process predictions and labels
processed_predictions, processed_labels = postprocess_text(all_predictions, all_labels)

# Compute ROUGE metrics
results = rouge.compute(predictions=processed_predictions, references=processed_labels, use_stemmer=True)

# Print ROUGE results
results = {k: round(v * 100, 2) for k, v in results.items()}  # Convert to percentages
print("ROUGE Scores:")
for metric, score in results.items():
    print(f"{metric}: {score}")


# Keyword extraction and evaluation using KeyBERT
kw_model = KeyBERT()

# Helper function to extract keywords
def extract_keywords(text, model, top_n=5):
    if isinstance(text, str) and text.strip():
        keywords = model.extract_keywords(text, keyphrase_ngram_range=(1, 2), top_n=top_n)
        return [kw[0] for kw in keywords]
    return []

# Extract keywords for each predicted and true summary
predicted_keywords = [extract_keywords(pred, kw_model) for pred in all_predictions]
label_keywords = [extract_keywords(label, kw_model) for label in all_labels]

# Compute keyword match scores
keyword_match_scores = []
for pred_keywords, true_keywords in zip(predicted_keywords, label_keywords):
    if pred_keywords and true_keywords:
        matches = len(set(pred_keywords) & set(true_keywords))
        keyword_match_scores.append(matches / len(set(true_keywords)))
    else:
        keyword_match_scores.append(0.0)

# Calculate average keyword match score
average_keyword_match = sum(keyword_match_scores) / len(keyword_match_scores)

print("\nKeyword Match Evaluation:")
print(f"Average Keyword Match Score: {round(average_keyword_match * 100, 2)}%")

# Save results to a DataFrame
df["Predicted Keywords"] = predicted_keywords
df["True Keywords"] = label_keywords
df["Keyword Match Score"] = keyword_match_scores

# Save the updated DataFrame to a new .csv file
output_path = "flan_t5_results_with_keywords.csv"
df.to_csv(output_path, index=False)

print(f"\nResults saved to: {output_path}")
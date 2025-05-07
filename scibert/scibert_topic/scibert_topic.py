import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import nltk
from tqdm import tqdm
import sys, os.path

nltk.download('punkt')

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from glimpse.glimpse.data_loading.Glimpse_tokenizer import glimpse_tokenizer

# === CONFIGURATION ===

MODEL_DIR = BASE_DIR / "scibert" / "scibert_topic" / "checkpoints" / "checkpoint-1600"
DATA_DIR = BASE_DIR / "glimpse" / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "topic_scored"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === Load model and tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === Tokenize like GLIMPSE ===
# def tokenize_sentences(text: str) -> list:
#     # same tokenization as in the original glimpse code
#     text = text.replace('-----', '\n')
#     sentences = nltk.sent_tokenize(text)
#     sentences = [sentence for sentence in sentences if sentence != ""]
#     return sentences 


# === Label map (optional: for human-readable output) ===
id2label = {
    0: "asp_substance",
    1: "asp_clarity",
    2: "asp_soundness-correctness",
    3: "asp_originality",
    4: "asp_impact",
    5: "asp_comparison",
    6: "asp_replicability",
    7: "arg-structuring_summary"
}

def predict_topic(sentences):
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1).cpu().tolist()
    # Convert predictions to human-readable labels
    predictions = [id2label[pred] for pred in predictions]
    return predictions


def find_topic(start_year=2017, end_year=2021):
    for year in range(start_year, end_year + 1):
        print(f"Processing {year}...")
        input_path = DATA_DIR / f"all_reviews_{year}.csv"
        output_path = OUTPUT_DIR / f"topic_scored_reviews_{year}.csv"

        df = pd.read_csv(input_path)

        all_rows = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            review_id = row["id"]
            text = row["text"]
            sentences = glimpse_tokenizer(text)
            if not sentences:
                continue
            labels = predict_topic(sentences)
            for sentence, topic in zip(sentences, labels):
                all_rows.append({"id": review_id, "sentence": sentence, "topic": topic})

        output_df = pd.DataFrame(all_rows)
        output_df.to_csv(output_path, index=False)
        print(f"Saved topic-scored data to {output_path}")


if __name__ == "__main__":
    find_topic()
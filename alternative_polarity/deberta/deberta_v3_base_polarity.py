import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import nltk
from tqdm import tqdm
import sys, os.path
from torch.nn import functional as F

nltk.download('punkt')

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from glimpse.glimpse.data_loading.Glimpse_tokenizer import glimpse_tokenizer

# === CONFIGURATION ===

MODEL_DIR = BASE_DIR / "alternative_polarity" / "deberta" / "deberta_v3_base_polarity_final_model"
DATA_DIR = BASE_DIR / "glimpse" / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "polarity_scored"
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


# def predict_polarity(sentences):
#     inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits
#         temperature = 2.7  # Adjust temperature for scaling logits
#         probs = F.softmax(logits / temperature, dim=-1)
#         # Get probability of positive class
#         polarity_scores = probs[:, 1]
#         # Rescale: 0 → -1 (very negative), 1 → +1 (very positive)
#         polarity_scores = (polarity_scores * 2) - 1
#     return polarity_scores.cpu().tolist()

def predict_polarity(sentences):
    inputs = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits            # (batch, 2)
        logit_diff = logits[:,1] - logits[:,0]
        alpha = 2.1                               # tweak
        scores = torch.tanh(alpha * logit_diff)   # in [-1,1]
    return scores.cpu().tolist()


def find_polarity(start_year=2017, end_year=2021):
    for year in range(start_year, end_year + 1):
        print(f"Processing {year}...")
        input_path = DATA_DIR / f"all_reviews_{year}.csv"
        output_path = OUTPUT_DIR / f"polarity_scored_reviews_{year}.csv"

        df = pd.read_csv(input_path)

        all_rows = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            review_id = row["id"]
            text = row["text"]
            sentences = glimpse_tokenizer(text)
            if not sentences:
                continue
            labels = predict_polarity(sentences)
            for sentence, polarity in zip(sentences, labels):
                all_rows.append({"id": review_id, "sentence": sentence, "polarity": polarity})

        output_df = pd.DataFrame(all_rows)
        output_df.to_csv(output_path, index=False)
        print(f"Saved polarity-scored data to {output_path}")


if __name__ == "__main__":
    find_polarity()
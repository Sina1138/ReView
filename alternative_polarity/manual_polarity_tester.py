import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from glimpse.glimpse.data_loading.Glimpse_tokenizer import glimpse_tokenizer

# === CONFIGURATION ===
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "alternative_polarity" / "deberta" / "deberta_v3_large_polarity_final_model"
# MODEL_DIR = BASE_DIR / "alternative_polarity" / "llama" / "final_model"
# MODEL_DIR = BASE_DIR / "alternative_polarity" / "scideberta" / "scideberta_full_polarity_final_model"

# --> Best so far: deberta_v3 (passes "pros" test)


# === Load model and tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === Prediction function with confidence ===
def predict_polarity(sentences):
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        confidences, preds = torch.max(probs, dim=1)

    results = []
    for sentence, pred, conf, prob in zip(sentences, preds, confidences, probs):
        results.append({
            "sentence": sentence,
            "label": "Positive" if pred.item() == 1 else "Negative",
            "confidence": conf.item(),
            "probs": prob.cpu().numpy().tolist()
        })
    return results

# === Example: test a multi-sentence peer review ===
if __name__ == "__main__":
    # Replace this with your review
    full_review = """
    Pros: 
    Con: The experiments lack comparison with prior work. 
    The authors clearly explain their methodology, which is a strong point.
    """

    # Use glimpse tokenizer to split into sentences
    sentences = glimpse_tokenizer(full_review)

    # Run polarity prediction
    results = predict_polarity(sentences)

    # Display results
    for res in results:
        print(f"\nSentence: {res['sentence']}")
        print(f" → Prediction: {res['label']} (Confidence: {res['confidence']:.3f})")
        print(f"   Probabilities: [Negative: {res['probs'][0]:.3f}, Positive: {res['probs'][1]:.3f}]")

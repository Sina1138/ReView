"""
Shared utilities for polarity and topic scoring pipelines.
Provides common functions for model loading, prediction, and result saving.
"""

import re
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def find_available_years(data_dir: Path) -> list:
    """
    Auto-detect years by scanning data directory for all_reviews_*.csv files.
    
    Args:
        data_dir: Path to directory containing processed review data
        
    Returns:
        Sorted list of years found
    """
    years = []
    if data_dir.exists():
        for file in data_dir.glob("all_reviews_*.csv"):
            match = re.search(r'all_reviews_(\d{4})\.csv', file.name)
            if match:
                years.append(int(match.group(1)))
    
    return sorted(years)


def _local_model_available(model_dir: Path) -> bool:
    """Check if a local model directory has the required files."""
    if not model_dir.exists():
        return False
    # Accept either pytorch_model.bin or safetensors
    has_weights = (model_dir / "pytorch_model.bin").exists() or (model_dir / "model.safetensors").exists()
    return has_weights and (model_dir / "config.json").exists()


def load_model_and_tokenizer(model_dir: Path, device: str = "cuda", hub_fallback: str = None):
    """
    Load a model and tokenizer from a local directory, or fall back to HuggingFace Hub.

    Args:
        model_dir: Path to local model directory
        device: Device to load model onto ("cuda" or "cpu")
        hub_fallback: HuggingFace Hub model ID to use if local files are missing

    Returns:
        Tuple of (tokenizer, model, device_obj)
    """
    model_source = str(model_dir)

    if not _local_model_available(model_dir):
        if hub_fallback:
            print(f"  Local model not found at {model_dir}")
            print(f"  Falling back to HuggingFace Hub: {hub_fallback}")
            model_source = hub_fallback
        else:
            raise FileNotFoundError(f"Model not found at {model_dir} and no hub fallback configured")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_source)
        model = AutoModelForSequenceClassification.from_pretrained(model_source)
        model.eval()

        # Move to device
        device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
        model.to(device_obj)

        return tokenizer, model, device_obj

    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_source}: {e}")


def predict_batch(sentences: list, tokenizer, model, device, max_length: int = 512) -> list:
    """
    Run batch predictions on a list of sentences.
    
    Args:
        sentences: List of sentence strings to predict
        tokenizer: Tokenizer instance
        model: Model instance
        device: Device object for computation
        max_length: Maximum token length (default: 512 for BERT-like models)
        
    Returns:
        List of predicted class IDs (integers)
    """
    if not sentences:
        return []
    
    try:
        inputs = tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1).cpu().tolist()
        
        return predictions
    
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")


def save_polarity_results(output_path: Path, results: list) -> None:
    """
    Save polarity scoring results to CSV.
    
    Expected result format:
    [
        {"id": review_id, "sentence": sentence_text, "score": float, "label": int},
        ...
    ]
    
    Args:
        output_path: Path to output CSV file
        results: List of result dictionaries
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)


def save_topic_results(output_path: Path, results: list) -> None:
    """
    Save topic scoring results to CSV.
    
    Expected result format:
    [
        {"id": review_id, "sentence": sentence_text, "topic_id": int, "topic_label": str},
        ...
    ]
    
    Args:
        output_path: Path to output CSV file
        results: List of result dictionaries
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)


def validate_input_file(input_path: Path, required_columns: list) -> pd.DataFrame:
    """
    Validate that input CSV file exists and has required columns.
    
    Args:
        input_path: Path to CSV file
        required_columns: List of column names that must exist
        
    Returns:
        Loaded DataFrame
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV {input_path}: {e}")
    
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df


def load_polarity_model(model_variant: str, base_dir: Path, device: str = "cuda"):
    """
    Factory function to load polarity model by variant name.

    Supported variants:
      - "scibert": scibert/scibert_polarity/final_model (F1=0.724 baseline)
      - "deberta": training/outputs/deberta_polarity/final_model (F1=0.764, +5.5% - RECOMMENDED)
      - "deberta_v3_small": training/outputs/deberta_v3_small_polarity/final_model (F1=0.754)
      - "modernbert": training/outputs/modernbert_polarity/final_model (F1=0.741)
      - "scideberta": training/outputs/scideberta_polarity/final_model (F1=0.737)

    Args:
        model_variant: Name of model variant
        base_dir: Base directory of project
        device: Device to load onto

    Returns:
        Tuple of (tokenizer, model, device_obj)

    Raises:
        ValueError: If model_variant not supported
        FileNotFoundError: If model directory doesn't exist
    """
    # Feb 2026: New trained models from training/outputs/ (standardized comparison)
    variant_map = {
        "scibert": base_dir / "training" / "outputs" / "scibert_polarity" / "final_model",
        "deberta": base_dir / "training" / "outputs" / "deberta_polarity" / "final_model",  # BEST: F1=0.764
        "deberta_v3_small": base_dir / "training" / "outputs" / "deberta_v3_small_polarity" / "final_model",
        "modernbert": base_dir / "training" / "outputs" / "modernbert_polarity" / "final_model",
        "scideberta": base_dir / "training" / "outputs" / "scideberta_polarity" / "final_model",
        # Legacy models (pre-Feb 2026, kept for backwards compatibility)
        "scibert_legacy": base_dir / "scibert" / "scibert_polarity" / "final_model",
        "deberta_legacy": base_dir / "alternative_polarity" / "deberta" / "deberta_v3_base_polarity_final_model",
        "scideberta_legacy": base_dir / "alternative_polarity" / "scideberta" / "scideberta_full_polarity_final_model",
    }
    hub_fallback_map = {
        "scibert": "Sina1138/Scibert_polarity_Review",
        "scideberta": "KISTI-AI/Scideberta-full",  # Needs fine-tuning
        "modernbert": "answerdotai/ModernBERT-base",  # Needs fine-tuning
        "deberta": "Sina1138/deberta_polarity_Review",  # DeBERTa-v3-base (F1=0.764)
        "deberta_v3_small": "microsoft/deberta-v3-small",  # Needs fine-tuning
    }

    if model_variant not in variant_map:
        raise ValueError(
            f"Unknown polarity model variant: {model_variant}. "
            f"Supported: {list(variant_map.keys())}"
        )

    model_dir = variant_map[model_variant]
    return load_model_and_tokenizer(model_dir, device, hub_fallback=hub_fallback_map.get(model_variant))


def load_topic_model(model_variant: str, base_dir: Path, device: str = "cuda"):
    """
    Factory function to load topic model by variant name.

    Supported variants:
      - "scideberta": training/outputs/scideberta_topic/final_model (F1=0.478 - BEST, RECOMMENDED)
      - "deberta": training/outputs/deberta_topic/final_model (F1=0.450)
      - "scibert": training/outputs/scibert_topic/final_model (F1=0.442)
      - "deberta_v3_small": training/outputs/deberta_v3_small_topic/final_model (F1=0.381)
      - "modernbert": training/outputs/modernbert_topic/final_model (F1=0.376)

    Args:
        model_variant: Name of model variant
        base_dir: Base directory of project
        device: Device to load onto

    Returns:
        Tuple of (tokenizer, model, device_obj)

    Raises:
        ValueError: If model_variant not supported
        FileNotFoundError: If model directory doesn't exist
    """
    # Feb 2026: New trained models from training/outputs/ (standardized comparison)
    variant_map = {
        "scideberta": base_dir / "training" / "outputs" / "scideberta_topic" / "final_model",  # BEST: F1=0.478
        "deberta": base_dir / "training" / "outputs" / "deberta_topic" / "final_model",
        "scibert": base_dir / "training" / "outputs" / "scibert_topic" / "final_model",
        "deberta_v3_small": base_dir / "training" / "outputs" / "deberta_v3_small_topic" / "final_model",
        "modernbert": base_dir / "training" / "outputs" / "modernbert_topic" / "final_model",
        # Legacy models (pre-Feb 2026, kept for backwards compatibility)
        "scibert_legacy": base_dir / "scibert" / "scibert_topic" / "final_model",
        "deberta_legacy": base_dir / "alternative_topic" / "deberta" / "final_model",
        "scideberta_legacy": base_dir / "alternative_topic" / "scideberta" / "final_model",
    }
    hub_fallback_map = {
        "scideberta": "Sina1138/scideberta_topic_Review",  # SciDeBERTa (F1=0.478)
        "scibert": "allenai/scibert_scivocab_uncased",  # Needs fine-tuning
        "deberta": "microsoft/deberta-v3-base",  # Needs fine-tuning
        "deberta_v3_small": "microsoft/deberta-v3-small",  # Needs fine-tuning
        "modernbert": "answerdotai/ModernBERT-base",  # Needs fine-tuning
    }

    if model_variant not in variant_map:
        raise ValueError(
            f"Unknown topic model variant: {model_variant}. "
            f"Supported: {list(variant_map.keys())}"
        )

    model_dir = variant_map[model_variant]
    return load_model_and_tokenizer(model_dir, device, hub_fallback=hub_fallback_map.get(model_variant))


# Topic label mapping
TOPIC_ID_TO_LABEL = {
    0: "Substance",
    1: "Clarity",
    2: "Soundness/Correctness",
    3: "Originality",
    4: "Motivation/Impact",
    5: "Meaningful Comparison",
    6: "Replicability",
    7: "NONE",
}


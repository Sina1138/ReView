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


def load_model_and_tokenizer(model_dir: Path, device: str = "cuda"):
    """
    Load a model and tokenizer from a local directory.
    
    Args:
        model_dir: Path to directory containing model (config.json, pytorch_model.bin, etc.)
        device: Device to load model onto ("cuda" or "cpu")
        
    Returns:
        Tuple of (tokenizer, model)
        
    Raises:
        FileNotFoundError: If model directory doesn't exist or is missing model files
    """
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Check for required files
    required_files = ["config.json", "pytorch_model.bin"]
    for required_file in required_files:
        if not (model_dir / required_file).exists():
            raise FileNotFoundError(f"Missing {required_file} in {model_dir}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        model.eval()
        
        # Move to device
        device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
        model.to(device_obj)
        
        return tokenizer, model, device_obj
    
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_dir}: {e}")


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
      - "scibert": scibert/scibert_polarity/final_model
      - "deberta": alternative_polarity/deberta/final_model
      - "scideberta": alternative_polarity/scideberta/final_model
    
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
    variant_map = {
        "scibert": base_dir / "scibert" / "scibert_polarity" / "final_model",
        "deberta": base_dir / "alternative_polarity" / "deberta" / "deberta_v3_base_polarity_final_model",
        "scideberta": base_dir / "alternative_polarity" / "scideberta" / "scideberta_full_polarity_final_model",
    }
    
    if model_variant not in variant_map:
        raise ValueError(
            f"Unknown polarity model variant: {model_variant}. "
            f"Supported: {list(variant_map.keys())}"
        )
    
    model_dir = variant_map[model_variant]
    return load_model_and_tokenizer(model_dir, device)


def load_topic_model(model_variant: str, base_dir: Path, device: str = "cuda"):
    """
    Factory function to load topic model by variant name.
    
    Supported variants:
      - "scibert": scibert/scibert_topic/final_model
      - "deberta": alternative_topic/deberta/final_model
      - "scideberta": alternative_topic/scideberta/final_model
    
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
    variant_map = {
        "scibert": base_dir / "scibert" / "scibert_topic" / "final_model",
        "deberta": base_dir / "alternative_topic" / "deberta" / "final_model",
        "scideberta": base_dir / "alternative_topic" / "scideberta" / "final_model",
    }
    
    if model_variant not in variant_map:
        raise ValueError(
            f"Unknown topic model variant: {model_variant}. "
            f"Supported: {list(variant_map.keys())}"
        )
    
    model_dir = variant_map[model_variant]
    return load_model_and_tokenizer(model_dir, device)


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

TOPIC_LABEL_TO_ID = {v: k for k, v in TOPIC_ID_TO_LABEL.items()}

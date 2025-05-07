import os
import json
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
base_path = BASE_DIR / "data" / "DISAPERE-main" / "DISAPERE" / "final_dataset"
output_path = BASE_DIR / "data" / "DISAPERE-main" / "SELFExtractedData"

###################################################################################
###################################################################################

# EXTRACTING POLARITY SENTENCES FROM DISAPERE DATASET

# def extract_polarity_sentences(json_dir):
#     data = []
#     for filename in os.listdir(json_dir):
#         if filename.endswith(".json"):
#             with open(os.path.join(json_dir, filename), "r") as f:
#                 thread = json.load(f)
#                 for sentence in thread.get("review_sentences", []):
#                     text = sentence.get("text", "").strip()
#                     polarity = sentence.get("polarity")
#                     if text and polarity in ["pol_positive", "pol_negative"]:
#                         label = 1 if polarity == "pol_positive" else 0
#                         data.append({"text": text, "label": label})
#     return pd.DataFrame(data)

# # Extract and save each split
# for split in ["train", "dev", "test"]:
#     df = extract_polarity_sentences(os.path.join(base_path, split))
#     out_file = os.path.join(output_path, f"disapere_polarity_{split}.csv")
#     df.to_csv(out_file, index=False)
#     print(f"{split.capitalize()} saved to {out_file}: {len(df)} samples")


###################################################################################
###################################################################################

# 2. EXTRACTING TOPIC SENTENCES FROM DISAPERE DATASET
#
# === Topic Label Mapping ===
# 0: asp_substance              -> Are there substantial experiments and/or detailed analyses?
# 1: asp_clarity                -> Is the paper clear, well-written and well-structured?
# 2: asp_soundness-correctness -> Is the approach sound? Are the claims supported?
# 3: asp_originality            -> Are there new topics, technique, methodology, or insights?
# 4: asp_impact                 -> Does the paper address an important problem?
# 5: asp_comparison             -> Are the comparisons to prior work sufficient and fair?
# 6: asp_replicability          -> Is it easy to reproduce and verify the correctness of the results?
# 7: arg-structuring_summary    -> Reviewer's summary of the manuscript

# Final topic classes
topic_classes = [
    "asp_substance",
    "asp_clarity",
    "asp_soundness-correctness",
    "asp_originality",
    "asp_impact",
    "asp_comparison",
    "asp_replicability",
    "arg-structuring_summary"
]

label_map = {label: idx for idx, label in enumerate(topic_classes)}

def extract_topic_sentences(json_dir):
    data = []
    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            with open(os.path.join(json_dir, filename), "r") as f:
                thread = json.load(f)
                for sentence in thread.get("review_sentences", []):
                    text = sentence.get("text", "").strip()
                    aspect = sentence.get("aspect", "")
                    fine_action = sentence.get("fine_review_action", "")
                    
                    # Decide label source
                    topic = aspect if aspect in label_map else (
                        fine_action if fine_action in label_map else None
                    )

                    if text and topic in label_map:
                        label = label_map[topic]
                        data.append({"text": text, "label": label})
    return pd.DataFrame(data)

# Extract and save each split
for split in ["train", "dev", "test"]:
    df = extract_topic_sentences(os.path.join(base_path, split))
    out_file = os.path.join(output_path, f"disapere_topic_{split}.csv")
    df.to_csv(out_file, index=False)
    print(f"{split.capitalize()} saved to {out_file}: {len(df)} samples")

###################################################################################
###################################################################################



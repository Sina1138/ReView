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
#                     if text:
#                         if polarity == "pol_positive":
#                             label = 2
#                         elif polarity == "pol_negative":
#                             label = 0
#                         else:
#                             label = 1
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
# 1: "Structuring"
# 0: "Evaluative"
# 2: "Request"
# 3: "Fact"
# 4: "Social"
# 5: "Other"
# 6: "Substance"
# 7: "Clarity"
# 8: "Soundness/Correctness"
# 9: "Originality"
# 10: "Motivation/Impact"
# 11: "Meaningful Comparison"
# 12: "Replicability"

# Final topic classes
topic_classes = [
    "asp_substance",
    "asp_clarity",
    "asp_soundness-correctness",
    "asp_originality",
    "asp_impact",
    "asp_comparison",
    "asp_replicability",
    "None",  # This is used for sentences that do not match any specific topic
    # "arg-structuring_summary"
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
                    # fine_action = sentence.get("fine_review_action", "")
                    
                    # Decide label source
                    topic = aspect if aspect in label_map else "None"

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



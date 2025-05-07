import pickle
import pandas as pd
from pathlib import Path
import os

def process_pickle_results(pickle_path: Path, output_path: Path):
    # === Load Pickle File ===
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    # === Extract Metadata ===
    reranking_model = data.get('metadata/reranking_model')
    rsa_iterations = data.get('metadata/rsa_iterations')
    results = data.get('results')

    print(f"Reranking model: {reranking_model}, RSA iterations: {rsa_iterations}")

    # === Validate Results ===
    if not isinstance(results, list):
        raise ValueError("The 'results' key is not a list. Please check the pickle file structure.")

    # === Process and Flatten Results ===
    csv_data = []
    for index, result in enumerate(results):
        row = {
            'index': index,
            'id': str(result.get('id')[0]),
            'consensuality_scores': result.get('consensuality_scores').to_dict()
                if isinstance(result.get('consensuality_scores'), pd.Series) else None,

            # Optional fields — uncomment as needed
            # 'best_base': result.get('best_base').tolist() if isinstance(result.get('best_base'), np.ndarray) else None,
            # 'best_rsa': result.get('best_rsa').tolist() if isinstance(result.get('best_rsa'), np.ndarray) else None,
            # 'speaker_df': result.get('speaker_df').to_json() if isinstance(result.get('speaker_df'), pd.DataFrame) else None,
            # 'listener_df': result.get('listener_df').to_json() if isinstance(result.get('listener_df'), pd.DataFrame) else None,
            # 'initial_listener': result.get('initial_listener').to_json() if isinstance(result.get('initial_listener'), pd.DataFrame) else None,
            # 'language_model_proba_df': result.get('language_model_proba_df').to_json() if isinstance(result.get('language_model_proba_df'), pd.DataFrame) else None,
            # 'initial_consensuality_scores': result.get('initial_consensuality_scores').to_dict() if isinstance(result.get('initial_consensuality_scores'), pd.Series) else None,
            # 'gold': result.get('gold'),
            # 'rationality': result.get('rationality'),
            # 'text_candidates': result.get('text_candidates').to_json() if isinstance(result.get('text_candidates'), pd.DataFrame) else None,
        }
        csv_data.append(row)

    # === Save to CSV ===
    df = pd.DataFrame(csv_data)
    df.to_csv(output_path, index=False)
    print(f"Results saved to '{output_path}'.")


if __name__ == "__main__":
    
    BASE_DIR = Path(__file__).resolve().parent
    
    # Set the path to the pickle file and the output CSV file
    pickle_file = BASE_DIR / "glimpse" / "output" / "extractive_sentences-_-all_reviews_2017-_-none-_-2025-03-26-18-42-15-_-r3-_-rsa_reranked-google-pegasus-arxiv.pk"
    output_file = BASE_DIR / "data" / "GLIMPSE_results_from_pk.csv"
    
    process_pickle_results(pickle_file, output_file)
import pandas as pd
import nltk
import ast
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

from glimpse.glimpse.data_loading.Glimpse_tokenizer import glimpse_tokenizer

# def tokenize_sentences(text: str) -> list:
#     # same tokenization as in the original glimpse code
#     text = text.replace('-----', '\n')
#     sentences = nltk.sent_tokenize(text)
#     sentences = [sentence for sentence in sentences if sentence != ""]
#     return sentences        
        
        
def preprocessed_scores(
        original_csv_path: Path,
        scored_csv_path: Path,
        polarity_csv_path: Path,
        topic_csv_path: Path,
    ) -> dict:
    
    original_df = pd.read_csv(original_csv_path)
    scored_df = pd.read_csv(scored_csv_path)
    polarity_df = pd.read_csv(polarity_csv_path)
    topic_df = pd.read_csv(topic_csv_path)

    scored_reviews = {}

    for _, row in original_df.iterrows():
        review_id = row["id"]
        review_text = row["text"]

        if review_id not in scored_df["id"].values or review_id not in polarity_df["id"].values:
            continue

        if review_id not in scored_reviews:
            scored_reviews[review_id] = []

        # Get consensuality scores
        consensuality_scores_str = scored_df[scored_df["id"] == review_id]["consensuality_scores"].iloc[0]
        consensuality_scores_dict = ast.literal_eval(consensuality_scores_str)

        # Get polarity scores
        polarity_rows = polarity_df[polarity_df["id"] == review_id]
        polarity_dict = dict(zip(polarity_rows["sentence"], polarity_rows["polarity"]))
        
        # Get topic scores
        topic_rows = topic_df[topic_df["id"] == review_id]
        topic_dict = dict(zip(topic_rows["sentence"], topic_rows["topic"]))

        scored_sentences = {}
        for sentence in glimpse_tokenizer(review_text):
            sentence_data = {}
            if sentence in consensuality_scores_dict:
                sentence_data["consensuality"] = consensuality_scores_dict[sentence]
            if sentence in polarity_dict:
                sentence_data["polarity"] = polarity_dict[sentence]
            if sentence in topic_dict:
                sentence_data["topic"] = topic_dict[sentence]
            if sentence_data:
                scored_sentences[sentence] = sentence_data

        scored_reviews[review_id].append(scored_sentences)

    return scored_reviews


def save_all_scored_reviews(
        start_year: int = 2017,
        end_year: int = 2021,
        input_dir: Path = BASE_DIR / "glimpse" / "data" / "processed",
        scored_csv_path: Path = BASE_DIR / "data" / "GLIMPSE_results_from_pk.csv",
        polarity_dir: Path = BASE_DIR / "data" / "polarity_scored",
        topic_dir: Path = BASE_DIR / "data" / "topic_scored",
        output_csv_path: Path = BASE_DIR / "data" / "preprocessed_scored_reviews.csv",
    ):
    
    all_scored_reviews = []

    for year in range(start_year, end_year + 1):
        print(f"Processing {year}...")
        try:
            original_csv_path = input_dir / f"all_reviews_{year}.csv"
            polarity_csv_path = polarity_dir / f"polarity_scored_reviews_{year}.csv"
            topic_csv_path = topic_dir / f"topic_scored_reviews_{year}.csv"
            scored_reviews = preprocessed_scores(
                original_csv_path,
                scored_csv_path,
                polarity_csv_path,
                topic_csv_path
            )
            all_scored_reviews.append({
                "year": year,
                "scored_dict": scored_reviews
            })

        except Exception as e:
            print(f"Skipped {year} due to error: {e}")

    df = pd.DataFrame(all_scored_reviews)
    df.to_csv(output_csv_path, index=False)
    print(f"All scored reviews saved to '{output_csv_path}'.")


def load_scored_reviews(csv_path: Path = BASE_DIR / "data" / "preprocessed_scored_reviews.csv") -> tuple:
    df = pd.read_csv(csv_path)
    df["scored_dict"] = df["scored_dict"].apply(ast.literal_eval)
    years = df["year"].tolist()
    
    return years, df


if __name__ == "__main__":
    save_all_scored_reviews()
    years, all_scored_reviews_df = load_scored_reviews()
    
    # Debugging sample output
    sample_year = 2017

    sample_df = all_scored_reviews_df[all_scored_reviews_df["year"] == sample_year]
    review_dict = sample_df["scored_dict"].iloc[0]

    print(f"\n=== Sample Review from {sample_year} ===")
    for review_id, sentence_data_list in review_dict.items():
        print(f"\nReview ID: {review_id}")
        for sentence_dict in sentence_data_list:
            for sentence, data in sentence_dict.items():
                print(f"  Sentence: {sentence}")
                for key, value in data.items():
                    print(f"    → {key}: {value}")
            break  # print only the first review's sentences
        break  # only one review

        
    # --- Testing code ---
    # scored_reviews_2017 = all_scored_reviews_df[all_scored_reviews_df["year"] == 2017]
    # print(scored_reviews_2017)
    # scored_reviews_2017 = scored_reviews_2017["scored_dict"].iloc[0]
    # # scored_reviews_2017 = ast.literal_eval(scored_reviews_2017)
    # print(type(scored_reviews_2017))
    # print(scored_reviews_2017.keys())
    # sample = scored_reviews_2017["https://openreview.net/forum?id=r1rhWnZkg"]
    # print(sample[0])
    
    # print(years)
    # for id in scored_reviews_2017.keys():
    #     print(len(scored_reviews_2017[id]))

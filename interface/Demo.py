import math

import sys, os.path

import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from glimpse.rsasumm.rsa_reranker import RSAReranking
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
import pandas as pd

from scored_reviews_builder import load_scored_reviews
from glimpse.glimpse.data_loading.Glimpse_tokenizer import glimpse_tokenizer
# from scibert.scibert_polarity.scibert_polarity import predict_polarity

# Load scored reviews
years, all_scored_reviews_df = load_scored_reviews()

# -----------------------------------
# Pre-processed Tab
# -----------------------------------

def get_preprocessed_scores(year):
    scored_reviews = all_scored_reviews_df[all_scored_reviews_df["year"] == year]["scored_dict"].iloc[0]
    return scored_reviews


# -----------------------------------
# Interactive Tab
# -----------------------------------

# RSA_model = "facebook/bart-large-cnn"
RSA_model = "sshleifer/distilbart-cnn-12-3"

model = AutoModelForSeq2SeqLM.from_pretrained(RSA_model)
tokenizer = AutoTokenizer.from_pretrained(RSA_model)

# Define the manual color map for topics
topic_color_map = {
    "Substance": "#cce0ff",             # lighter blue
    "Clarity": "#e6ee9c",               # lighter yellow-green
    "Soundness/Correctness": "#ffcccc", # lighter red
    "Originality": "#d1c4e9",           # lighter purple
    "Motivation/Impact": "#b2ebf2",     # lighter teal
    "Meaningful Comparison": "#fff9c4", # lighter yellow
    "Replicability": "#c8e6c9",         # lighter green
}


# GLIMPSE Home/Description Page
glimpse_description = """
# ReView: A Tool for Visualizing and Analyzing Scientific Reviews
## **Overview**
ReView is a visualization tool designed to assist **area chairs** and **researchers** in efficiently analyzing scholarly reviews. The interface offers two main ways to explore scholarly reviews:
- Pre-Processed Reviews: Explore real peer reviews from ICLR (2017–2021) with structured visualizations of sentiment, topics, and reviewer agreement.
- Interactive Tab: Enter your own reviews and view them analyzed in real time using the same NLP-powered highlighting options.
All reviews are shown in their original, unaltered form, with visual overlays to help identify key insights such as disagreements, sentiment and common themes—reducing cognitive load and scrolling effort.
---
## **Key Features**
- *Traceability and Transparency:* The tool preserves the original text of each review and overlays highlights for key aspects (e.g., sentiment, topic, agreement), allowing area chairs to trace back every insight to its source without modifying or summarizing the content.
- *Structured Overview*: All reviews are displayed in one interface and with radio buttons, one can navigate from one highlighting option to the other.
- *Interactive*: The tool allows users to input their own reviews and, within seconds, view them annotated with highlighted aspects
---
## **Highlighting Options**
- *Agreement:* Identifies both shared and conflicting points across reviews, helping to surface consensus and disagreement.
- *Polarity:* Highlights positive and negative sentiments within the reviews to reveal tone and stance.
- *Topic:* Organizes the review sentences by their discussed topics, ensuring coverage of diverse reviewer perspectives and improving clarity. 
---
### How to Use ReView
ReView offers two main ways to explore peer reviews: using pre-processed reviews or by entering your own.
#### 🗂️ Pre-Processed Reviews Tab
Use this tab to explore reviews from ICLR (2017–2021):
1. **Select a conference year** from the dropdown menu on the right.
2. **Navigate between submissions** using the *Next* and *Previous* buttons on the left.
3. **Choose a highlighting view** using the radio buttons:
   - **Original**: Displays unmodified review text.
   - **Agreement**: Highlights consensus points in **red** and disagreements in **purple**.
   - **Polarity**: Highlights **positive** sentiment in **green** and **negative** sentiment in **red**.
   - **Topic**: Highlights comments by discussion topic using color-coded labels.
#### ✍️ Interactive Tab
Use this tab to analyze your own review text:
1. **Enter up to three reviews** in the input fields labeled *Review 1*, *Review 2*, and *Review 3*.
2. **Click "Process"** to analyze the input (average processing time: ~42 seconds).
3. **Explore the results** using the same highlighting options as above (Agreement, Polarity, Topic).
"""


EXAMPLES = [
    "The paper gives really interesting insights on the topic of transfer learning. It is well presented and the experiment are extensive. I believe the authors missed Jane and al 2021. In addition, I think, there is a mistake in the math.",
    "The paper gives really interesting insights on the topic of transfer learning. It is well presented and the experiment are extensive. Some parts remain really unclear and I would like to see a more detailed explanation of the proposed method.",
    "The paper gives really interesting insights on the topic of transfer learning. It is not well presented and lack experiments. Some parts remain really unclear and I would like to see a more detailed explanation of the proposed method.",
]

# Function to summarize the input texts using the RSAReranking model in interactive mode
def summarize(text1, text2, text3, focus, mode, rationality=1.0, iterations=1):
    
    # print(focus, mode, rationality, iterations)
    
    # get sentences for each text
    text2_sentences = glimpse_tokenizer(text2)
    text1_sentences = glimpse_tokenizer(text1)
    text3_sentences = glimpse_tokenizer(text3)


    # remove empty sentences
    text1_sentences = [sentence for sentence in text1_sentences if sentence != ""]
    text2_sentences = [sentence for sentence in text2_sentences if sentence != ""]
    text3_sentences = [sentence for sentence in text3_sentences if sentence != ""]

    sentences = list(set(text1_sentences + text2_sentences + text3_sentences))
    
    # Load polarity model and tokenizer (SciBERT)
    polarity_model_path = "Sina1138/Scibert_polarity_Review"
    polarity_tokenizer = AutoTokenizer.from_pretrained(polarity_model_path)
    polarity_model = AutoModelForSequenceClassification.from_pretrained(polarity_model_path)
    polarity_model.eval()
    polarity_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    polarity_model.to(polarity_device)

    def predict_polarity(sent_list):
        inputs = polarity_tokenizer(
            sent_list, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(polarity_device)
        with torch.no_grad():
            logits = polarity_model(**inputs).logits
            preds = torch.argmax(logits, dim=1).cpu().tolist()
        emoji_map = {0: "➖", 1: None, 2: "➕"}
        return dict(zip(sent_list, [emoji_map[p] for p in preds]))


    # Run polarity prediction
    polarity_map = predict_polarity(sentences)


    # Load topic model and tokenizer (SciBERT)
    topic_model_path = "Sina1138/SciDeberta_Review"
    topic_tokenizer = AutoTokenizer.from_pretrained(topic_model_path)
    topic_model = AutoModelForSequenceClassification.from_pretrained(topic_model_path)
    topic_model.eval()
    topic_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    topic_model.to(topic_device)
    
    def predict_topic(sent_list):
        inputs = topic_tokenizer(
            sent_list, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(topic_device)
        with torch.no_grad():
            logits = topic_model(**inputs).logits
            preds = torch.argmax(logits, dim=1).cpu().tolist()
        
        # Topic ID to label and emoji
        id2label = {
            0: "Substance",
            1: "Clarity",
            2: "Correctness",
            3: "Originality",
            4: "Impact",
            5: "Comparison",
            6: "Replicability",
            7: None  # This is used for sentences that do not match any specific topic,
        }
        return dict(zip(sent_list, [id2label[p] for p in preds]))
    
    # Run topic prediction
    topic_map = predict_topic(sentences)
    


    rsa_reranker = RSAReranking(
        model,
        tokenizer,
        candidates=sentences,
        source_texts=[text1, text2, text3],
        device="cpu",
        rationality=rationality,
    )
    (
        best_rsa,
        best_base,
        speaker_df,
        listener_df,
        initial_listener,
        language_model_proba_df,
        initial_consensuality_scores,
        consensuality_scores,
    ) = rsa_reranker.rerank(t=iterations)

    # apply exp to the probabilities
    speaker_df = speaker_df.applymap(lambda x: math.exp(x))

    text_1_summaries = speaker_df.loc[text1][text1_sentences]
    text_1_summaries = text_1_summaries / text_1_summaries.sum()

    text_2_summaries = speaker_df.loc[text2][text2_sentences]
    text_2_summaries = text_2_summaries / text_2_summaries.sum()

    text_3_summaries = speaker_df.loc[text3][text3_sentences]
    text_3_summaries = text_3_summaries / text_3_summaries.sum()

    # make list of tuples
    text_1_summaries = [(sentence, text_1_summaries[sentence]) for sentence in text1_sentences]
    text_2_summaries = [(sentence, text_2_summaries[sentence]) for sentence in text2_sentences]
    text_3_summaries = [(sentence, text_3_summaries[sentence]) for sentence in text3_sentences]

    # normalize consensuality scores between -1 and 1
    consensuality_scores = (consensuality_scores - (consensuality_scores.max() - consensuality_scores.min()) / 2) / (consensuality_scores.max() - consensuality_scores.min()) / 2

    # get most and least consensual sentences
    # most consensual --> most common; least consensual --> most unique
    most_consensual = consensuality_scores.sort_values(ascending=True).head(3).index.tolist()
    least_consensual = consensuality_scores.sort_values(ascending=False).head(3).index.tolist()
    
    # Convert lists to strings
    most_consensual = " ".join(most_consensual)
    least_consensual = " ".join(least_consensual)

    text_1_consensuality = consensuality_scores.loc[text1_sentences]
    text_2_consensuality = consensuality_scores.loc[text2_sentences]
    text_3_consensuality = consensuality_scores.loc[text3_sentences]

    text_1_consensuality = [(sentence, text_1_consensuality[sentence]) for sentence in text1_sentences]
    text_2_consensuality = [(sentence, text_2_consensuality[sentence]) for sentence in text2_sentences]
    text_3_consensuality = [(sentence, text_3_consensuality[sentence]) for sentence in text3_sentences]


    def highlight_reviews(text_sentences, consensuality_scores, threshold_common=0.0, threshold_unique=0.0):
        highlighted = []
        for sentence in text_sentences:
            # print(f"Processing sentence: {sentence}", "score:", consensuality_scores.loc[sentence])
            score = consensuality_scores.loc[sentence]
            score = score*2 if score > 0 else score  # amplify unique scores for better visibility
            
            # common sentences --> positive consensuality scores
            # unique sentences --> negative consensuality scores
            
            score *= -1 # invert the score for highlighting
            
            highlighted.append((sentence, score))
        return highlighted

    # Apply highlighting to each review
    text_1_agreement = highlight_reviews(text1_sentences, consensuality_scores)
    text_2_agreement = highlight_reviews(text2_sentences, consensuality_scores)
    text_3_agreement = highlight_reviews(text3_sentences, consensuality_scores)
    
    # Add polarity outputs
    text_1_polarity = [(s, polarity_map[s]) for s in text1_sentences]
    text_2_polarity = [(s, polarity_map[s]) for s in text2_sentences]
    text_3_polarity = [(s, polarity_map[s]) for s in text3_sentences]
    
    # Add topic outputs
    text_1_topic = [(s, topic_map[s]) for s in text1_sentences]
    text_2_topic = [(s, topic_map[s]) for s in text2_sentences]
    text_3_topic = [(s, topic_map[s]) for s in text3_sentences]
    
    # print(type(text_1_consensuality))
    return (
        # text_1_summaries, text_2_summaries, text_3_summaries,
        # text_1_consensuality, text_2_consensuality, text_3_consensuality,
        text_1_agreement, text_2_agreement, text_3_agreement,
        most_consensual, least_consensual,
        text_1_polarity, text_2_polarity, text_3_polarity,
        text_1_topic, text_2_topic, text_3_topic,
    )




with gr.Blocks(title="ReView") as demo:
    # gr.Markdown("# ReView Interface")
    
    with gr.Tab("Introduction"):
        gr.Markdown(glimpse_description)
        
    # -----------------------------------
    # Pre-processed Tab
    # -----------------------------------
    with gr.Tab("Pre-processed Reviews"):
        # Initialize state for this session.
        initial_year = 2017
        initial_scored_reviews = get_preprocessed_scores(initial_year)
        initial_review_ids = list(initial_scored_reviews.keys())
        initial_review = initial_scored_reviews[initial_review_ids[0]]
        number_of_displayed_reviews = len(initial_scored_reviews[initial_review_ids[0]])
        initial_state = {
            "year_choice": initial_year,
            "scored_reviews_for_year": initial_scored_reviews,
            "review_ids": initial_review_ids,
            "current_review_index": 0,
            "current_review": initial_review,
            "number_of_displayed_reviews": number_of_displayed_reviews,
        }
        state = gr.State(initial_state)

        def update_review_display(state, score_type):

            review_ids = state["review_ids"]
            current_index = state["current_review_index"]
            current_review = state["scored_reviews_for_year"][review_ids[current_index]]

            show_polarity = score_type == "Polarity"
            show_consensuality = score_type == "Agreement"
            show_topic = score_type == "Topic"
            
            
            if show_polarity:
                color_map = {"➕": "#d4fcd6", "➖": "#fcd6d6"}
                legend = False
            elif show_topic:
                color_map = topic_color_map  # No color map for topics
                legend = False
            elif show_consensuality:
                color_map = None  # Continuous scale, no predefined colors
                legend = True
            else:
                color_map = {}  # Default to empty map
                legend = False

            new_review_id = (
                f"### Submission Link:\n\n{review_ids[current_index]}<br>"
                f"(Showing {current_index + 1} of {len(state['review_ids'])} reviews)"
            )

            number_of_displayed_reviews = len(current_review)
            review_updates = []
            consensuality_dict = {}

            for i in range(8):
                if i < number_of_displayed_reviews:
                    review_item = list(current_review[i].items())

                    if show_polarity:
                        highlighted = []
                        for sentence, metadata in review_item:
                            polarity = metadata.get("polarity", None)
                            if polarity >= 0.995:
                                label = "➕"  # positive
                            elif polarity <= -0.99:
                                label = "➖"  # negative
                            else:
                                label = None  # ignore neutral (1)
                            highlighted.append((sentence, label))
                    elif show_consensuality:
                        highlighted = []
                        for sentence, metadata in review_item:
                            score = metadata.get("consensuality", 0.0)
                            score = score * 2 - 1  # Normalize to [-1, 1]
                            score = score/2.5 if score > 0 else score  # Amplify unique scores for better visibility
                            score *= -1  # Invert the score for highlighting
                            
                            consensuality_dict[sentence] = score
                            highlighted.append((sentence, score))
                        
                    elif show_topic:
                        highlighted = []
                        for sentence, metadata in review_item:
                            topic = metadata.get("topic", None)
                            if topic != "NONE":
                                highlighted.append((sentence, topic))
                            else:
                                highlighted.append((sentence, None))
                    else:
                        highlighted = [
                            (sentence, None)
                            for sentence, metadata in review_item
                        ]

                    review_updates.append(
                        gr.update(
                            visible=True,
                            value=highlighted,
                            color_map=color_map,
                            show_legend=legend,
                            key=f"updated_{score_type}_{i}"
                        )
                    )
                else:
                    review_updates.append(
                        gr.update(
                            visible=False,
                            value=[],
                            show_legend=False,
                            color_map=color_map,
                            key=f"updated_{score_type}_{i}"
                        )
                    )

            # Set most consensual / unique sentences
            if show_consensuality and consensuality_dict:
                scores = pd.Series(consensuality_dict)
                most_unique = scores.sort_values(ascending=True).head(3).index.tolist()
                most_common = scores.sort_values(ascending=False).head(3).index.tolist()
                most_common_text = "\n".join(most_common)
                most_unique_text = "\n".join(most_unique)

                most_common_visibility = gr.update(visible=True, value=most_common_text)
                most_unique_visibility = gr.update(visible=True, value=most_unique_text)
            else:
                # Debugging statements to check visibility settings
                # print("Hiding most common and unique sentences")

                most_common_visibility = gr.update(visible=False, value=[])
                most_unique_visibility = gr.update(visible=False, value=[])
                
            # update topic color map
            if show_topic:
                topic_color_map_visibility = gr.update(
                    visible=True,
                    color_map=topic_color_map,
                    value=[
                        ("", "Substance"),
                        ("", "Clarity"),
                        ("", "Soundness/Correctness"),
                        ("", "Originality"),
                        ("", "Motivation/Impact"),
                        ("", "Meaningful Comparison"),
                        ("", "Replicability"),
                    ]
                )
            else:
                topic_color_map_visibility = gr.update(visible=False, value=[])

            return (
                new_review_id,
                *review_updates,
                most_common_visibility,
                most_unique_visibility,
                topic_color_map_visibility,
                state
            )



        # Precompute the initial outputs so something is shown on load.
        init_display = update_review_display(initial_state, score_type="Original")
        # init_display returns: (review_id, review1, review2, review3, review4, review5, review6, review7, review8, state)

        with gr.Row():
            
            with gr.Column(scale=1):
                review_id = gr.Markdown(value=init_display[0], container=True)
                with gr.Row():
                    previous_button = gr.Button("Previous", variant="secondary", interactive=True)
                    next_button = gr.Button("Next", variant="primary", interactive=True)
                    
                    
            with gr.Column(scale=1):
                # Input controls.
                year = gr.Dropdown(choices=years, label="Select Year", interactive=True, value=initial_year)
                score_type = gr.Radio(
                    choices=["Original", "Agreement", "Polarity", "Topic"],
                    label="Score Type to Display",
                    value="Original",
                    interactive=True
                )

        # Output display.
        with gr.Row():
            most_common_sentences = gr.Textbox(
            lines=8,
            label="Most Common Opinions",
            visible=False,
            value=[]
        )
            most_unique_sentences = gr.Textbox(
            lines=8,
            label="Most Divergent Opinions",
            visible=False,
            value=[]
        )
        
        # Add a new textbox for topic labels and colors
        topic_text_box = gr.HighlightedText(
            label="Topic Labels (Color-Coded)",
            visible=False,
            value=[],
            show_legend=True,
        )
        
        review1 = gr.HighlightedText(
            show_legend=False,
            label="Review 1",
            visible= number_of_displayed_reviews >= 1,
            key="initial_review1",
            # color_map={"Positive": "#d4fcd6", "Negative": "#fcd6d6"}
        )
        review2 = gr.HighlightedText(
            show_legend=False,
            label="Review 2",
            visible= number_of_displayed_reviews >= 2,
            key="initial_review2"
            # color_map={"Positive": "#d4fcd6", "Negative": "#fcd6d6"}
        )
        review3 = gr.HighlightedText(
            show_legend=False,
            label="Review 3",
            visible= number_of_displayed_reviews >= 3,
            key="initial_review3"
            # color_map={"Positive": "#d4fcd6", "Negative": "#fcd6d6"}
        )
        review4 = gr.HighlightedText(
            show_legend=False,
            label="Review 4",
            visible= number_of_displayed_reviews >= 4,
            key="initial_review4"
            # color_map={"Positive": "#d4fcd6", "Negative": "#fcd6d6"}
        )
        review5 = gr.HighlightedText(
            show_legend=False,
            label="Review 5",
            visible= number_of_displayed_reviews >= 5,
            key="initial_review5"
            # color_map={"Positive": "#d4fcd6", "Negative": "#fcd6d6"}
        )
        review6 = gr.HighlightedText(
            show_legend=False,
            label="Review 6",
            visible= number_of_displayed_reviews >= 6,
            key="initial_review6"
            # color_map={"Positive": "#d4fcd6", "Negative": "#fcd6d6"}
        )
        review7 = gr.HighlightedText(
            show_legend=False,
            label="Review 7",
            visible= number_of_displayed_reviews >= 7,
            key="initial_review7"
            # color_map={"Positive": "#d4fcd6", "Negative": "#fcd6d6"}
        )
        review8 = gr.HighlightedText(
            show_legend=False,
            label="Review 8",
            visible= number_of_displayed_reviews >= 8,
            key="initial_review8"
            # color_map={"Positive": "#d4fcd6", "Negative": "#fcd6d6"}
        )

        # Callback functions that update state.
        def year_change(year, state, score_type):
            state["year_choice"] = year
            state["scored_reviews_for_year"] = get_preprocessed_scores(year)
            state["review_ids"] = list(state["scored_reviews_for_year"].keys())
            state["current_review_index"] = 0
            state["current_review"] = state["scored_reviews_for_year"][state["review_ids"][0]]
            return update_review_display(state, score_type)

        def next_review(state, score_type):
            state["current_review_index"] = (state["current_review_index"] + 1) % len(state["review_ids"])
            state["current_review"] = state["scored_reviews_for_year"][state["review_ids"][state["current_review_index"]]]
            return update_review_display(state, score_type)

        def previous_review(state, score_type):
            state["current_review_index"] = (state["current_review_index"] - 1) % len(state["review_ids"])
            state["current_review"] = state["scored_reviews_for_year"][state["review_ids"][state["current_review_index"]]]
            return update_review_display(state, score_type)

        # Hook up the callbacks with the session state.
        year.change(
            fn=year_change,
            inputs=[year, state, score_type],
            outputs=[review_id, review1, review2, review3, review4, review5, review6, review7, review8, most_common_sentences, most_unique_sentences, topic_text_box, state]
        )
        score_type.change(
            fn=update_review_display,
            inputs=[state, score_type],
            outputs=[review_id, review1, review2, review3, review4, review5, review6, review7, review8, most_common_sentences, most_unique_sentences, topic_text_box, state]
        )
        next_button.click(
            fn=next_review,
            inputs=[state, score_type],
            outputs=[review_id, review1, review2, review3, review4, review5, review6, review7, review8, most_common_sentences, most_unique_sentences, topic_text_box, state]
        )
        previous_button.click(
            fn=previous_review,
            inputs=[state, score_type],
            outputs=[review_id, review1, review2, review3, review4, review5, review6, review7, review8, most_common_sentences, most_unique_sentences, topic_text_box, state]
        )   
        
        
        
        
    # -----------------------------------
    # Interactive Tab
    # -----------------------------------
    with gr.Tab("Interactive", interactive=True):            
        with gr.Row():
            with gr.Column():
                
                gr.Markdown("## Input Reviews")
                
                # review_count = gr.Slider(minimum=1, maximum=3, step=1, value=3, label="Number of Reviews", interactive=True)

                review1_textbox = gr.Textbox(lines=5, value=EXAMPLES[0], label="Review 1", interactive=True)
                review2_textbox = gr.Textbox(lines=5, value=EXAMPLES[1], label="Review 2", interactive=True)
                review3_textbox = gr.Textbox(lines=5, value=EXAMPLES[2], label="Review 3", interactive=True)
                
                with gr.Row():
                    submit_button = gr.Button("Process", variant="primary", interactive=True)
                    clear_button = gr.Button("Clear", variant="secondary", interactive=True)
                gr.Markdown("**Note**: *Once your inputs are processed, you can see the different result by <ins>**only changing the parameters**</ins>, and without the need to re-process.*", container=True)
                
                
                
            with gr.Column():
                
                gr.Markdown("## Results")
                
                mode_radio = gr.Radio(
                    choices=[("In-line Highlighting", "highlight"), ("Generate Summaries", "summary")],
                    value="highlight",
                    label="Output Mode:",
                    interactive=False,
                    visible=False  # Initially hidden, will be shown based on mode selection
                )
                focus_radio = gr.Radio(
                    choices=[("Agreement", "unique"), "Polarity", "Topic",],
                    value="unique",
                    label="Focus on:",
                    interactive=True
                )                
                generation_method_radio = gr.Radio(
                    choices=[("Extractive", "extractive")], #TODO: add ("Abstractive", "abstractive") and abstractive generation
                    value="extractive",
                    label="Generation Method:",
                    interactive=True,
                    visible=False
                )
                
                # Fixed rationality (3.0) and iterations (2) to be consistent with the compute_rsa.py script
                #iterations_slider = gr.Slider(minimum=1, maximum=10, step=1, value=2, label="Iterations", interactive=False, visible=False)
                # rationality_slider = gr.Slider(minimum=0.0, maximum=10.0, step=0.1, value=2.0, label="Rationality", interactive=False, visible=False)
                    
                with gr.Row():
                    unique_sentences = gr.Textbox(
                        lines=6, label="Most Divergent Opinions", visible=True, value=None, container=True
                    )
                    common_sentences = gr.Textbox(
                        lines=6, label="Most Common Opinions", visible=True, value=None, container=True
                    )
                
                uniqueness_score_text1 = gr.HighlightedText(
                    show_legend=True, label="Agreement in Review 1", visible=True, value=None,
                )
                uniqueness_score_text2 = gr.HighlightedText(
                    show_legend=True, label="Agreement in Review 2", visible=True, value=None,
                )
                uniqueness_score_text3 = gr.HighlightedText(
                    show_legend=True, label="Agreement in Review 3", visible=True, value=None,
                )
                
                
                polarity_score_text1 = gr.HighlightedText(
                    show_legend=True, label="Polarity in Review 1", visible=False, value=None,
                    color_map={"➕": "#d4fcd6", "➖": "#fcd6d6" }
                )
                polarity_score_text2 = gr.HighlightedText(
                    show_legend=True, label="Polarity in Review 2", visible=False, value=None,
                    color_map={"➕": "#d4fcd6", "➖": "#fcd6d6" }
                )
                polarity_score_text3 = gr.HighlightedText(
                    show_legend=True, label="Polarity in Review 3", visible=False, value=None,
                    color_map={"➕": "#d4fcd6", "➖": "#fcd6d6" }
                )
                
                aspect_score_text1 = gr.HighlightedText(
                    show_legend=False, label="Topic in Review 1", visible=False, value=None,
                    color_map = topic_color_map
                )
                aspect_score_text2 = gr.HighlightedText(
                    show_legend=False, label="Topic in Review 2", visible=False, value=None,
                    color_map = topic_color_map
                )
                aspect_score_text3 = gr.HighlightedText(
                    show_legend=False, label="Topic in Review 3", visible=False, value=None,
                    color_map = topic_color_map
                )
                
                

            
            # Connect summarize function to submit button
            submit_button.click(
                fn=summarize,
                inputs=[
                    review1_textbox, review2_textbox, review3_textbox,
                    focus_radio, mode_radio
                ],
                outputs=[
                    uniqueness_score_text1, uniqueness_score_text2, uniqueness_score_text3,
                    common_sentences, unique_sentences,
                    polarity_score_text1, polarity_score_text2, polarity_score_text3,
                    aspect_score_text1, aspect_score_text2, aspect_score_text3 
                    
                ]
            )
            
            # Define clear button behavior
            clear_button.click(
                fn=lambda: (None, None, None, None, None, None, None, None, None, None, None), # clear all fields
                inputs=[],
                outputs=[
                    review1_textbox, review2_textbox, review3_textbox,
                    uniqueness_score_text1, uniqueness_score_text2, uniqueness_score_text3,
                    common_sentences, unique_sentences
                ]
            )
            
            # Update visibility of generation_method_radio based on mode_radio value
            # def toggle_generation_method(mode):
            #     if mode == "summary":
            #         return gr.update(visible=True), gr.update(visible=False) # show generation method radio, hide focus radio
            #     else:
            #         return gr.update(visible=False), gr.update(visible=True) # show focus radio, hide generation method radio
            
            # mode_radio.change(
            #     fn=toggle_generation_method,
            #     inputs=mode_radio,
            #     outputs=[generation_method_radio, focus_radio]
            # )
            
            # Update visibility of output textboxes based on mode_radio and focus_radio values
            def toggle_output_textboxes(mode, focus):
                if mode == "highlight" and focus == "unique":
                    return (
                        gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), # in-line uniqueness highlights
                        gr.update(visible=True), gr.update(visible=True), # summary highlights
                        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), # polarity highlights
                        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False) # aspect highlights
                    )

                elif focus == "Polarity":
                    return (
                        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), # in-line uniqueness highlights
                        gr.update(visible=False), gr.update(visible=False), # summary highlights
                        gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), # polarity highlights
                        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False) # aspect highlights
                    )
                
                elif focus == "Topic":
                    return (
                        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), # in-line uniqueness highlights
                        gr.update(visible=False), gr.update(visible=False), # summary highlights
                        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), # polarity highlights
                        gr.update(visible=True), gr.update(visible=True), gr.update(visible=True) # aspect highlights
                    )
            
            focus_radio.change(
                fn=toggle_output_textboxes,
                inputs=[mode_radio, focus_radio],
                outputs=[
                    uniqueness_score_text1, uniqueness_score_text2, uniqueness_score_text3,
                    common_sentences, unique_sentences,
                    polarity_score_text1, polarity_score_text2, polarity_score_text3,
                    aspect_score_text1, aspect_score_text2, aspect_score_text3
                ]
            )
            # mode_radio.change(
            #     fn=toggle_output_textboxes,
            #     inputs=[mode_radio, focus_radio],
            #     outputs=[
            #         uniqueness_score_text1, uniqueness_score_text2, uniqueness_score_text3,
            #         consensuality_score_text1, consensuality_score_text2, consensuality_score_text3,
            #         most_consensual_sentences, most_unique_sentences
            #     ]
            # )
           
        # TODO: Configure the slider for the number of review boxes 
        
        # def toggle_reviews(number_of_displayed_reviews):
        #     number_of_displayed_reviews = int(number_of_displayed_reviews)
        #     updates = []
        #     # for review(i), set visible True if its index is <= n, otherwise False.
        #     for i in range(1, 4): updates.append(gr.update(visible=(i <= number_of_displayed_reviews)))
        #     return tuple(updates)

        # review_count.change(
        #     fn=toggle_reviews,
        #     inputs=[review_count],
        #     outputs=[review1_textbox, review2_textbox, review3_textbox]
        # )
        
    demo.load(
        fn=update_review_display,
        inputs=[state, score_type],
            outputs=[review_id, review1, review2, review3, review4, review5, review6, review7, review8, most_common_sentences, most_unique_sentences, topic_text_box, state]
    )          

demo.launch(share=False)
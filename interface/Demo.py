import math
import numpy as np
import seaborn as sns

import sys, os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from glimpse.rsasumm.rsa_reranker import RSAReranking
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from scored_reviews_builder import load_scored_reviews
from glimpse.glimpse.data_loading.Glimpse_tokenizer import glimpse_tokenizer

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

MODEL = "facebook/bart-large-cnn"

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
tokenizer = AutoTokenizer.from_pretrained(MODEL)


EXAMPLES = [
    "The paper gives really interesting insights on the topic of transfer learning. It is well presented and the experiment are extensive. I believe the authors missed Jane and al 2021. In addition, I think, there is a mistake in the math.",
    "The paper gives really interesting insights on the topic of transfer learning. It is well presented and the experiment are extensive. Some parts remain really unclear and I would like to see a more detailed explanation of the proposed method.",
    "The paper gives really interesting insights on the topic of transfer learning. It is not well presented and lack experiments. Some parts remain really unclear and I would like to see a more detailed explanation of the proposed method.",
]



def summarize(text1, text2, text3, focus, mode, rationality, iterations=2):
    
    print(focus, mode, rationality, iterations)
    
    # get sentences for each text
    text2_sentences = glimpse_tokenizer(text2)
    text1_sentences = glimpse_tokenizer(text1)
    text3_sentences = glimpse_tokenizer(text3)


    # remove empty sentences
    text1_sentences = [sentence for sentence in text1_sentences if sentence != ""]
    text2_sentences = [sentence for sentence in text2_sentences if sentence != ""]
    text3_sentences = [sentence for sentence in text3_sentences if sentence != ""]

    sentences = list(set(text1_sentences + text2_sentences + text3_sentences))

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
    consensuality_scores_01 = (consensuality_scores - consensuality_scores.min()) / (consensuality_scores.max() - consensuality_scores.min())

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

    text_1_consensuality_ = consensuality_scores_01.loc[text1_sentences]
    text_2_consensuality_ = consensuality_scores_01.loc[text2_sentences]
    text_3_consensuality_ = consensuality_scores_01.loc[text3_sentences]

    text_1_consensuality_ = [(sentence, text_1_consensuality_[sentence]) for sentence in text1_sentences]
    text_2_consensuality_ = [(sentence, text_2_consensuality_[sentence]) for sentence in text2_sentences]
    text_3_consensuality_ = [(sentence, text_3_consensuality_[sentence]) for sentence in text3_sentences]


    #for text in text_1_summaries: print(text)
    #for text in text_1_consensuality: print(text)
    # print("Most consensual sentences: ", most_consensual)
    # print(text_1_consensuality)
    
    print(type(text_1_consensuality))
    return text_1_summaries, text_2_summaries, text_3_summaries, text_1_consensuality, text_2_consensuality, text_3_consensuality, most_consensual, least_consensual


# GLIMPSE Home/Description Page
glimpse_description = """
# GLIMPSE: Pragmatically Informative Multi-Document Summarization of Scholarly Reviews

GLIMPSE is a summarization tool designed to assist **area chairs** and **researchers** in efficiently analyzing and synthesizing scholarly peer reviews. Utilizing the **Rational Speech Act (RSA)** framework, GLIMPSE identifies both **common themes** and **unique perspectives** across multiple reviews, ensuring a comprehensive overview of the evaluation landscape.

Unlike traditional summarization methods that focus on consensus opinions, GLIMPSE emphasizes **both alignment and divergence** in reviewer feedback. This approach provides a **balanced and transparent representation** of the review content, supporting informed decision-making processes.

---

## **Key Features**
- **Discriminative Summarization:** Highlights both shared insights and unique arguments across reviews.  
- **RSA-Based Scoring:** Prioritizes key statements based on informativeness and uniqueness metrics.  
- **Balanced Summaries:** Ensures clarity and coverage of diverse reviewer perspectives.  
- **Traceability and Transparency:** Maintains clear attribution of summarized points to their original sources.  

GLIMPSE is designed to **streamline the review synthesis process**, offering an effective and reliable method for extracting meaningful insights from complex review datasets.

---

For more information and to begin using GLIMPSE, please proceed with the interface.  
You can choose between the **Interactive** mode for real-time summarization or the **Pre-processed** mode for batch processing of review data.
"""


with gr.Blocks(title="GLIMPSE") as demo:
    gr.Markdown("# GLIMPSE Method Interface")
    
    with gr.Tab("Home"):
        gr.Markdown(glimpse_description)
    
    with gr.Tab("Interactive", interactive=True):    
        gr.Markdown("""
            This is an interactive demo of the GLIMPSE Method.\n
            After pressing the 'Process' button, the model will generate the requested outputs based on the selected parameters.  
            The 'Output Mode' parameter allows you to choose between in-line highlighting and summary generation.  
            - In case of **In-Line Highlighting** mode, the 'Focus on' parameter allows you to choose between uniqueness and commonality.  
            - For **Summary Generation**, you can choose between having extractive or abstractive (TBA) summaries.
        """)
        
        with gr.Row():
            with gr.Column():
                
                gr.Markdown("## Input Reviews")
                
                # review_count = gr.Slider(minimum=1, maximum=3, step=1, value=3, label="Number of Reviews", interactive=True)

                review1_textbox = gr.Textbox(lines=8, value=EXAMPLES[0], label="Review 1", interactive=True)
                review2_textbox = gr.Textbox(lines=8, value=EXAMPLES[1], label="Review 2", interactive=True)
                review3_textbox = gr.Textbox(lines=8, value=EXAMPLES[2], label="Review 3", interactive=True)
                
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
                    interactive=True
                )
                focus_radio = gr.Radio(
                    choices=[("Uniqueness", "unique"), ("Commonality", "common")],
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
                rationality_slider = gr.Slider(minimum=0.0, maximum=10.0, step=0.1, value=2.0, label="Rationality", interactive=False, visible=False)
                    
                    
                uniqueness_score_text1 = gr.HighlightedText(
                    show_legend=True, label="Uniqueness scores for each sentence in Review 1", visible=True, value=None,
                )
                uniqueness_score_text2 = gr.HighlightedText(
                    show_legend=True, label="Uniqueness scores for each sentence in Review 2", visible=True, value=None,
                )
                uniqueness_score_text3 = gr.HighlightedText(
                    show_legend=True, label="Uniqueness scores for each sentence in Review 3", visible=True, value=None,
                )
                consensuality_score_text1 = gr.HighlightedText(
                    show_legend=True, label="Commonality scores for each sentence in Review 1", visible=False, value=None,
                )
                consensuality_score_text2 = gr.HighlightedText(
                    show_legend=True, label="Commonality scores for each sentence in Review 2", visible=False, value=None,
                )
                consensuality_score_text3 = gr.HighlightedText(
                    show_legend=True, label="Commonality score for each sentence in Review 3", visible=False, value=None,
                )
                with gr.Column():
                    most_unique_sentences = gr.Textbox(
                        lines=8, label="Most unique sentences", visible=False, show_copy_button=True, container=True
                    )
                with gr.Column():
                    most_consensual_sentences = gr.Textbox(
                        lines=8, label="Most common sentences", visible=False, show_copy_button=True, container=True
                    )

            
            # Connect summarize function to submit button
            submit_button.click(
                fn=summarize,
                inputs=[
                    review1_textbox, review2_textbox, review3_textbox,
                    focus_radio, mode_radio, rationality_slider
                ],
                outputs=[
                    uniqueness_score_text1, uniqueness_score_text2, uniqueness_score_text3,
                    consensuality_score_text1, consensuality_score_text2, consensuality_score_text3,
                    most_consensual_sentences, most_unique_sentences, 
                ]
            )
            
            # Define clear button behavior
            clear_button.click(
                fn=lambda: (None, None, None, None, None, None, None, None, None, None, None), # clear all fields
                inputs=[],
                outputs=[
                    review1_textbox, review2_textbox, review3_textbox,
                    uniqueness_score_text1, uniqueness_score_text2, uniqueness_score_text3,
                    consensuality_score_text1, consensuality_score_text2, consensuality_score_text3,
                    most_consensual_sentences, most_unique_sentences
                ]
            )
            
            # Update visibility of generation_method_radio based on mode_radio value
            def toggle_generation_method(mode):
                if mode == "summary":
                    return gr.update(visible=True), gr.update(visible=False) # show generation method radio, hide focus radio
                else:
                    return gr.update(visible=False), gr.update(visible=True) # show focus radio, hide generation method radio
            
            mode_radio.change(
                fn=toggle_generation_method,
                inputs=mode_radio,
                outputs=[generation_method_radio, focus_radio]
            )
            
            # Update visibility of output textboxes based on mode_radio and focus_radio values
            def toggle_output_textboxes(mode, focus):
                if mode == "highlight" and focus == "unique":
                    return (
                        gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), # in-line uniqueness highlights
                        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), # in-line commonality highlights
                        gr.update(visible=False), gr.update(visible=False) # summary highlights
                    )
                elif mode == "highlight" and focus == "common":
                    return (
                        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), # in-line uniqueness highlights
                        gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), # in-line commonality highlights
                        gr.update(visible=False), gr.update(visible=False) # summary highlights
                    )
                elif mode == "summary":
                    return (
                        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), # in-line uniqueness highlights
                        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), # in-line commonality highlights
                        gr.update(visible=True), gr.update(visible=True) # summary highlights
                    )
            
            focus_radio.change(
                fn=toggle_output_textboxes,
                inputs=[mode_radio, focus_radio],
                outputs=[
                    uniqueness_score_text1, uniqueness_score_text2, uniqueness_score_text3,
                    consensuality_score_text1, consensuality_score_text2, consensuality_score_text3,
                    most_consensual_sentences, most_unique_sentences
                ]
            )
            mode_radio.change(
                fn=toggle_output_textboxes,
                inputs=[mode_radio, focus_radio],
                outputs=[
                    uniqueness_score_text1, uniqueness_score_text2, uniqueness_score_text3,
                    consensuality_score_text1, consensuality_score_text2, consensuality_score_text3,
                    most_consensual_sentences, most_unique_sentences
                ]
            )
           
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
            
            
            
    with gr.Tab("Pre-processed"):
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
            
            # Check the given state for the score type to display
            show_polarity = score_type == "Polarity"
            show_consensuality = score_type == "Consensuality"
            show_topic = score_type == "Topic"
            
            new_review_id = (
                f"### Submission Link:\n\n{review_ids[current_index]}<br>"
                f"(Showing {current_index + 1} of {len(state['review_ids'])} reviews)"
            )
            
            number_of_displayed_reviews = len(current_review)
            review_updates = []

            for i in range(8):
                if i < number_of_displayed_reviews:
                    review_item = list(current_review[i].items())

                    if show_polarity:
                        # Binary color based on polarity
                        highlighted = [
                            (sentence, "✅" if metadata.get("polarity") == 1 else "❌")
                            for sentence, metadata in review_item
                        ]
                    elif show_consensuality:
                        # Gradient color based on consensuality
                        highlighted = [
                            (sentence, metadata["consensuality"])
                            for sentence, metadata in review_item
                        ]
                    elif show_topic:
                        # Topic color based on topic
                        highlighted = [
                            (sentence, metadata["topic"])
                            for sentence, metadata in review_item
                        ]
                    else:
                        # No highlighting
                        highlighted = [
                            (sentence, None)
                            for sentence, metadata in review_item
                        ]

                    review_updates.append(gr.update(visible=True, value=highlighted))
                else:
                    review_updates.append(gr.update(visible=False, value=""))

            return (new_review_id, *review_updates, state)


        # Precompute the initial outputs so something is shown on load.
        init_display = update_review_display(initial_state, score_type="Consensuality")
        # init_display returns: (review_id, review1, review2, review3, review4, review5, review6, review7, review8, state)

        # Input controls.
        year = gr.Dropdown(choices=years, label="Select Year", interactive=True, value=initial_year)
        score_type = gr.Radio(
            choices=["None", "Consensuality", "Polarity", "Topic"],
            label="Score Type to Display",
            value="None",
            interactive=True
        )
        
        
        with gr.Row():
            next_button = gr.Button("Next", variant="primary", interactive=True)
            previous_button = gr.Button("Previous", variant="secondary", interactive=True)
        review_id = gr.Markdown(value=init_display[0], container=True)

        # Output display.
        review1 = gr.HighlightedText(
            show_legend=True,
            label="Glimpse scores for each sentence in Review 1",
            visible= number_of_displayed_reviews >= 1,
            color_map={"✅": "#d4fcd6", "❌": "#fcd6d6"}
        )
        review2 = gr.HighlightedText(
            show_legend=True,
            label="Glimpse scores for each sentence in Review 2",
            visible= number_of_displayed_reviews >= 2,
            color_map={"✅": "#d4fcd6", "❌": "#fcd6d6"}
        )
        review3 = gr.HighlightedText(
            show_legend=True,
            label="Glimpse scores for each sentence in Review 3",
            visible= number_of_displayed_reviews >= 3,
            color_map={"✅": "#d4fcd6", "❌": "#fcd6d6"}
        )
        review4 = gr.HighlightedText(
            show_legend=True,
            label="Glimpse scores for each sentence in Review 4",
            visible= number_of_displayed_reviews >= 4,
            color_map={"✅": "#d4fcd6", "❌": "#fcd6d6"}
        )
        review5 = gr.HighlightedText(
            show_legend=True,
            label="Glimpse scores for each sentence in Review 5",
            visible= number_of_displayed_reviews >= 5,
            color_map={"✅": "#d4fcd6", "❌": "#fcd6d6"}
        )
        review6 = gr.HighlightedText(
            show_legend=True,
            label="Glimpse scores for each sentence in Review 6",
            visible= number_of_displayed_reviews >= 6,
            color_map={"✅": "#d4fcd6", "❌": "#fcd6d6"}
        )
        review7 = gr.HighlightedText(
            show_legend=True,
            label="Glimpse scores for each sentence in Review 7",
            visible= number_of_displayed_reviews >= 7,
            color_map={"✅": "#d4fcd6", "❌": "#fcd6d6"}
        )
        review8 = gr.HighlightedText(
            show_legend=True,
            label="Glimpse scores for each sentence in Review 8",
            visible= number_of_displayed_reviews >= 8,
            color_map={"✅": "#d4fcd6", "❌": "#fcd6d6"}
        )

        # Callback functions that update state.
        def year_change(year, state, show_polarity):
            state["year_choice"] = year
            state["scored_reviews_for_year"] = get_preprocessed_scores(year)
            state["review_ids"] = list(state["scored_reviews_for_year"].keys())
            state["current_review_index"] = 0
            state["current_review"] = state["scored_reviews_for_year"][state["review_ids"][0]]
            return update_review_display(state, show_polarity)

        def next_review(state, show_polarity):
            state["current_review_index"] = (state["current_review_index"] + 1) % len(state["review_ids"])
            state["current_review"] = state["scored_reviews_for_year"][state["review_ids"][state["current_review_index"]]]
            return update_review_display(state, show_polarity)

        def previous_review(state, show_polarity):
            state["current_review_index"] = (state["current_review_index"] - 1) % len(state["review_ids"])
            state["current_review"] = state["scored_reviews_for_year"][state["review_ids"][state["current_review_index"]]]
            return update_review_display(state, show_polarity)

        # Hook up the callbacks with the session state.
        year.change(
            fn=year_change,
            inputs=[year, state, score_type],
            outputs=[review_id, review1, review2, review3, review4, review5, review6, review7, review8, state]
        )
        score_type.change(
            fn=update_review_display,
            inputs=[state, score_type],
            outputs=[review_id, review1, review2, review3, review4, review5, review6, review7, review8, state]
        )
        next_button.click(
            fn=next_review,
            inputs=[state, score_type],
            outputs=[review_id, review1, review2, review3, review4, review5, review6, review7, review8, state]
        )
        previous_button.click(
            fn=previous_review,
            inputs=[state, score_type],
            outputs=[review_id, review1, review2, review3, review4, review5, review6, review7, review8, state]
        )
        
    demo.load(
        fn=update_review_display,
        inputs=[state, score_type],
        outputs=[review_id, review1, review2, review3, review4, review5, review6, review7, review8, state]
    )
        
                    

demo.launch(share=False)

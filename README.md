This is the repository of ReView: A Tool for Visualizing and Analyzing Scientific Reviews. [Code](https://github.com/sina1138/glimpse-ui) | [Hugging Face Spaces](https://huggingface.co/spaces/Sina1138/ReView)
<!-- [Paper]() | -->

<!-- ## Cloning this repository

This project uses Git submodules. To clone including all submodules, run:
```bash
git clone --recurse-submodules https://github.com/Sina1138/glimpse-ui.git
```
If you already cloned without submodules, run:
```git submodule update --init --recursive``` -->

## Installation

- Since this project was built with Python 3.10, first create the following virtual environment:
``` bash
module load miniconda/3
conda create -n ReView python=3.10
```
- Second, activate the environment and install pytorch:
``` bash
conda activate ReView
```

- In the next step, make sure you have git-lfs package and install git lfs, since it is used in this repo for the preprocessed dataset:
``` bash
conda install git-lfs
git lfs install
```

- Finally, all remaining required packages could be installed with the requirements file:

``` bash
pip install -r requirements.txt
```

- (Optional) To enable fetching reviews directly from OpenReview links in the Interactive tab:
``` bash
pip install openreview-py
```

## Running Interface Locally

To run this interface locally, first, make sure `gradio` and all the requirements are installed in your environment. Then, you can run the following for a local instance of the interface:
```bash
python interface/Demo.py
```

> **Note:** Do not use `gradio ./interface/Demo.py` (hot-reload mode), as it is currenlty incompatible with the interface's dynamic UI.

Additionally, you can edit the last line of code for a shareable link of your local instance as desired (change `demo.launch(share=False)` to `demo.launch(share=True)`)

## Introduction and Instructions

For an up-to-date, concise, and brief introduction of the interface, you can check out the "Introduction" tab of the ReView app [here](https://huggingface.co/spaces/Sina1138/ReView)

## Data Processing Pipeline

All data processing scripts live in the `pipeline/` directory:

| File | Purpose |
|------|---------|
| `pipeline/run_scoring.py` | Unified end-to-end pipeline orchestrator |
| `pipeline/process_new_data.sh` | Shell entry point (forwards to `run_scoring.py`) |
| `pipeline/fetch_iclr_data.py` | Fetch reviews from OpenReview API |
| `pipeline/preprocess_data.py` | Text cleaning and preprocessing |
| `pipeline/run_glimpse_scoring.py` | GLIMPSE consensuality/agreement scoring |
| `pipeline/run_polarity_scoring.py` | Polarity/sentiment scoring |
| `pipeline/run_topic_scoring.py` | Topic/aspect scoring |
| `pipeline/scored_reviews_builder.py` | Integrate all scores into final dataset |
| `pipeline/config.py` | Centralized configuration |

The pipeline auto-detects available years from the `data/` directory. To process data for any year:

```bash
# Fetch data for a new year
python pipeline/fetch_iclr_data.py --year 2026

# Run the full scoring pipeline (auto-detects all available years)
./pipeline/process_new_data.sh

# Or run for a specific year
./pipeline/process_new_data.sh --year 2026
```

## Performance

Since this project was built for deployment on Hugging Face Spaces, it is optimized to run on CPU. However, if better performance is needed, you can run this interface on a CUDA-enabled device and profit from the improved performance of the models in the interactive page. The code is set up to automatically use CUDA if available.

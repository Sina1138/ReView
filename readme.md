
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
pip install -r requirements
```

## Running Interface Locally

To run this interface locally, first, make sure `gradio` and all the requirements are installed in your environment. Then, you can run the following for a local instance of the interface:
```bash
gradio ./interface/Demo.py
```


Additionally, you can edit the last line of code for a shareable link of your local instance as desired (change `demo.launch(share=False)` to `demo.launch(share=True)`)

## Introduction and Instructions

For an up-to-date, concise, and brief introduction of the interface, you can check out the "Introduction" tab of the ReView app [here](https://huggingface.co/spaces/Sina1138/ReView)

## Performance

Since this project was built for deployment on Hugging Face Spaces, it is optimized to run on CPU. However, if better performance is needed, you can run this interface on a CUDA-enabled device and profit from the improved performance of the models in the interactive page. The code is set up to automatically use CUDA if available.
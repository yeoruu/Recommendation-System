# DMMD4SR: Diffusion Model-based Multi-level Multimodal Denoising for Sequential Recommendation

This repository contains the code for our **anonymous** submission to **ACMMM 2025**.

## Model Architecture

The overall framework of our proposed DMMD4SR model is illustrated below:

![Model Framework](model_framework.png)

## Environment Setup

The code requires the following main dependencies:

*   Python == 3.9
*   PyTorch == 2.1.1

You can set up the environment using conda or pip. For example:

```bash
# Using conda (recommended)
conda create -n dmmd4sr python=3.9
conda activate dmmd4sr

# Install PyTorch (check https://pytorch.org/get-started/previous-versions/ for your specific CUDA version)
# Example for CUDA 11.8:
# conda install pytorch==2.1.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# Example for CPU-only:
# conda install pytorch==2.1.1 torchvision torchaudio cpuonly -c pytorch

# Install other common dependencies (adjust if you have a requirements.txt)
pip install numpy pandas tqdm scikit-learn # Add other common packages if needed
```

*Note: Other required packages are common libraries and can be installed using pip as needed.*

## Datasets

We evaluate our model on the following 5-core subsets of the Amazon Review Data:

*   Home & Kitchen (`Home`)
*   Beauty
*   Tools & Home Improvement (`Tools`)
*   Toys & Games (`Toys`)

The datasets can be found at: [UCSD Amazon Review Data](https://jmcauley.ucsd.edu/data/amazon/)

Please download the relevant datasets and place/preprocess them according to the instructions within the codebase or preprocessing scripts (if provided).

## How to Run

To run the experiment on the Beauty dataset, execute the following script:

```bash
bash run_beauty.sh
```

*(You may need to adapt the script or create similar ones for other datasets.)*

## Code Framework Acknowledgement

Our implementation is based on the codebase of [STOSA](https://github.com/zfan20/STOSA?tab=readme-ov-file). We thank the authors for releasing their code.
```


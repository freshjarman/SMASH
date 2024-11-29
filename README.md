# SMASH: Score Matching-based Pseudolikelihood Estimation of Neural Marked Spatio-Temporal Point Process

This code is for SMASH main experiment on three marked STPP datasets: Earthquake, Crime and Football. Data files are included. The code is tested under a Linux desktop with torch 2.0 and Python 3.11.4.


## Environment
- Tested OS: Linux
- Python >= 3.7
- PyTorch == 2.0.1

## Data

Refer to the repo [DSTPP](https://github.com/tsinghua-fib-lab/Spatio-temporal-Diffusion-Point-Processes).

## Model Training

Use the following command to train and test SMASH on `Earthquake` dataset: 

``
cd ./scripts
bash train_earthquake.sh
``


## Note

The implemention is based on one of the baselines: DSTPP.


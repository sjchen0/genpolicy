# Trainable Policy for Fast Sampling

This repo is adapted from https://github.com/ML-GSAI/RADD.

## Code Organization

1. `run_train_policy.py`: Codes for training the policy
2. `run_sample.py`: Codes for sampling text and evaluating generative perplexity
3. `model/`: The model architecture
4. `losses.py`: Loss functions

## Dependency

Create a conda environment containing the required dependencies:
```
conda env create -f environment.yml
conda activate radd
```
## Pretrained Diffusion Language Model

We make use of RADD, a pretrained DLM trained with OpenWebText for 400k steps. It has multiple versions trained with different loss functions, available on Hugging Face:
|Model|Loss function|Total model size|
|:---:|:---:|:---:|
|[radd-lambda-dce](https://huggingface.co/JingyangOu/radd-lambda-dce)|$\lambda$-DCE|162M|
|[radd-t-dce](https://huggingface.co/JingyangOu/radd-t-dce)|$t$-DCE|162M|
|[radd-lambda-dce-medium](https://huggingface.co/JingyangOu/radd-lambda-dce-medium)|$\lambda$-DCE|405M|

For example, to load the `radd-t-dce` model and noise schedule, use the following code:
```python
from load_model import load_model
model, noise = load_model('JingyangOu/radd-t-dce', device='cuda') 
```

Our policy model will serve as an add-on to the pretrained RADD model to boost its sampling performance.

## Sampling

Currently support three methods: direct (random unmasking), confidence (difference between top-2 prob, given in [Kim et al.'25]), policy (ours).

Example script to sample and evaluate:

```bash
python run_sample.py --strategy direct --batch_size 1 --steps 16
python run_sample.py --strategy confidence --batch_size 1 --steps 16
python run_sample.py --strategy policy --batch_size 1 --steps 16
```

Policy sampling needs to load a trained policy model. One checkpoint is provided in `example/014609`.

## Policy Training

An example training script:

```bash
WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 torchrun --nproc-per-node 1 train_policy.py training.accum=1 training.batch_size=8 eval.batch_size=8 sampling.discrete_steps=16 model.dtype=float32
```

A new directory `output/DATE/TIME` with the following structure will be created for each run:
```
├── output
│   ├── .hydra
│   │   ├── config.yaml
│   │   ├── ...
│   ├── checkpoints
│   │   ├── checkpoint_*.pth
│   ├── checkpoints-meta
│   │   ├── checkpoint.pth
│   ├── samples
│   │   ├── iter_*
│   │   │   ├── sample_*.txt
│   ├── wandb
│   │   ├── ...
│   ├── logs
```
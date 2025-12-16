import os
import torch
from model import RADD, PolicyNet
import utils
from model.ema import ExponentialMovingAverage
import noise_lib
from hydra import initialize, compose
from omegaconf import OmegaConf
from pathlib import Path
import json
from omegaconf import OmegaConf


def load_model_hf(dir, device):
    score_model = RADD.from_pretrained(dir, local_files_only=True).to(device)
    noise = noise_lib.get_noise(score_model.config).to(device)
    return score_model, noise

def load_model_local(root_dir, device):
    cfg = utils.load_hydra_config_from_run(root_dir)
    noise = noise_lib.get_noise(cfg).to(device)
    score_model = RADD(cfg).to(device)

    ema = ExponentialMovingAverage(score_model.parameters(), decay=cfg.training.ema)

    ckpt_dir = os.path.join(root_dir, "checkpoints-meta", "checkpoint.pth")
    loaded_state = torch.load(ckpt_dir, map_location=device)

    score_model.load_state_dict(loaded_state['model'])
    ema.load_state_dict(loaded_state['ema'])

    ema.store(score_model.parameters())
    ema.copy_to(score_model.parameters())
    return score_model,  noise


def load_model(root_dir, device):
    return load_model_hf(root_dir, device)
    try:
        return load_model_hf(root_dir, device)
    except:
        raise Exception("Loading from HuggingFace failed. Please make sure the model path is correct and you have internet connection.")
        # return load_model_local(root_dir, device)

def load_policy(policy_ckpt, device):
    # import ipdb; ipdb.set_trace()
    policy_weights = torch.load(policy_ckpt, weights_only=False, map_location=device)['policy_model']
    config_path = Path(policy_ckpt).parent.parent/".hydra"
    with initialize(config_path=str(config_path), version_base=None):
        cfg = compose(config_name="config")
    policy_model = PolicyNet(cfg).to(device)
    policy_model.load_state_dict(policy_weights)
    return policy_model
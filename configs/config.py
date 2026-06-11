# configs/config.py

import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA = {
    "echonet_dynamic":  os.path.join(ROOT_DIR, "data", "echonet_dynamic"),
    "echonet_pediatric": os.path.join(ROOT_DIR, "data", "echonet_pediatric"),
    "camus": os.path.join(ROOT_DIR, "data", "camus"),
}

EXPERIMENTS_DIR = os.path.join(ROOT_DIR, "experiments")

CLIP = {
    "num_frames": 16,
    "img_size": 224,
    "channels": 3,
    "target_fps": 25,
}

BACKBONE = {
    "name": "uniformerv2_b16",
    "feature_dim": 768,
    "pretrained": True,
    "lora_rank": 8,
    "lora_alpha": 16,
    "adapter_dim": 64,
    "fallback_feature_dim": 512,
}

OAA = {
    "num_heads": 8,
    "dropout": 0.1,
    "roi_alpha": 0.15,
}

TAH = {
    "proj_dim": 256,
    "num_heads": 4,
    "num_layers": 2,
    "ffn_dim": 1024,
    "dropout": 0.1,
    "bias_init": -2.0,
}

SMOOTH = {
    "kernel_size": 5,
}

LOSS = {
    "lambda_reg": 0.2,
    "lambda_smooth": 0.1,
}

SSL = {
    "epochs": 100,
    "lr": 1e-4,
    "weight_decay": 0.01,
    "warmup_epochs": 5,
    "mask_ratio": 0.30,
    "beta": 1.0,
    "batch_size": 8,
    "num_workers": 8,
    "save_every": 10,
}

FINETUNE = {
    "epochs": 80,
    "lr": 5e-5,
    "weight_decay": 0.01,
    "warmup_epochs": 5,
    "lr_scheduler": "cosine",
    "early_stop_patience": 10,
    "batch_size": 8,
    "num_workers": 4,
    "mixed_precision":True,
    "grad_clip": 0.5,
    "seed": 42,
    "save_every": 5,
}

# MTR pre-training settings (used by training/tah_pretrain.py for M3)
MTR = {
    "epochs": 30,
    "lr": 1e-4,
    "weight_decay": 0.01,
    "batch_size": 8,
    "num_workers": 4,
    "mask_ratio": 0.25,
    "save_every": 5,
}

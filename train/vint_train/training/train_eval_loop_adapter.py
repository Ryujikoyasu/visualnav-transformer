import wandb
import os
import numpy as np
from typing import Dict
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms

from .train_eval_loop import train_eval_loop_nomad
from ..models.nomad.nomad_adapter import NoMaDAdapter

def train_eval_loop_nomad_adapter(
    train_model: bool,
    base_model: torch.nn.Module,
    adapter_bottleneck_dim: int,
    **kwargs
):
    """
    Adapter層を使用したNOMADモデルの学習ループ
    """
    # AdapterモデルでNOMADをラップ
    model = NoMaDAdapter(base_model, adapter_bottleneck_dim)
    
    # Adapterのパラメータのみで最適化
    optimizer = Adam(model.get_adapter_parameters(), lr=kwargs.get('lr', 1e-4))
    
    # 通常の学習ループを使用
    return train_eval_loop_nomad(
        train_model=train_model,
        model=model,
        optimizer=optimizer,
        **kwargs
    ) 
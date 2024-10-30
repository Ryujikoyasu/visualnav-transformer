import torch
import torch.nn as nn
from .nomad import NoMaD
from .adapter_modules import AdapterTransformerBlock

class NoMaDAdapter(nn.Module):
    """
    Adapter層を組み込んだNOMADモデル
    """
    def __init__(self, base_model: NoMaD, adapter_bottleneck_dim: int = 64):
        super().__init__()
        self.base_model = base_model
        
        # Transformerブロックを探してAdapterを追加
        self._add_adapters(adapter_bottleneck_dim)
        
        # ベースモデルのパラメータをフリーズ
        self._freeze_base_parameters()
        
    def _add_adapters(self, adapter_bottleneck_dim):
        """Transformer層にAdapter層を追加"""
        for name, module in self.base_model.named_modules():
            if "TransformerBlock" in module.__class__.__name__:
                parent_name = ".".join(name.split(".")[:-1])
                block_name = name.split(".")[-1]
                parent = self.base_model
                for part in parent_name.split("."):
                    parent = getattr(parent, part)
                setattr(parent, block_name, 
                       AdapterTransformerBlock(module, adapter_bottleneck_dim))
    
    def _freeze_base_parameters(self):
        """Adapter以外のパラメータをフリーズ"""
        for name, param in self.base_model.named_parameters():
            if "adapter" not in name:
                param.requires_grad = False
    
    def forward(self, func_name, **kwargs):
        return self.base_model(func_name, **kwargs)

    def get_adapter_parameters(self):
        """Adapter層のパラメータのみを返す"""
        return [p for n, p in self.named_parameters() if "adapter" in n] 
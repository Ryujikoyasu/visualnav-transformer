import torch
import torch.nn as nn
import torch.nn.functional as F

class AdapterLayer(nn.Module):
    """
    Transformer層に追加するAdapter層
    """
    def __init__(self, input_dim: int, bottleneck_dim: int):
        super().__init__()
        self.down_project = nn.Linear(input_dim, bottleneck_dim)
        self.up_project = nn.Linear(bottleneck_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        
    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.down_project(x)
        x = F.relu(x)
        x = self.up_project(x)
        return x + residual

class AdapterTransformerBlock(nn.Module):
    """
    Adapterを組み込んだTransformerBlock
    """
    def __init__(self, base_transformer_block, adapter_bottleneck_dim):
        super().__init__()
        self.base_block = base_transformer_block
        self.adapter1 = AdapterLayer(
            input_dim=base_transformer_block.norm1.normalized_shape[0],
            bottleneck_dim=adapter_bottleneck_dim
        )
        self.adapter2 = AdapterLayer(
            input_dim=base_transformer_block.norm2.normalized_shape[0],
            bottleneck_dim=adapter_bottleneck_dim
        )
        
    def forward(self, x):
        # Self-attention + Adapter1
        residual = x
        x = self.base_block.norm1(x)
        x = self.base_block.attn(x)
        x = self.base_block.drop_path(x)
        x = self.adapter1(x)
        x = x + residual
        
        # FFN + Adapter2
        residual = x
        x = self.base_block.norm2(x)
        x = self.base_block.mlp(x)
        x = self.base_block.drop_path(x)
        x = self.adapter2(x)
        x = x + residual
        return x 
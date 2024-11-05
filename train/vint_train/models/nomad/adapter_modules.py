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

def add_adapter_to_transformer_layer(transformer_layer: nn.TransformerEncoderLayer, adapter_bottleneck_dim: int):
    """
    既存のTransformerEncoderLayerにAdapter層を追加する
    """
    input_dim = transformer_layer.linear2.out_features
    
    # Self-attention後のAdapter
    attn_adapter = AdapterLayer(input_dim, adapter_bottleneck_dim)
    
    # FFN後のAdapter
    ffn_adapter = AdapterLayer(input_dim, adapter_bottleneck_dim)
    
    # 元のforward関数を保存
    original_forward = transformer_layer.forward
    
    def new_forward(src, src_mask=None, src_key_padding_mask=None):
        # 元のforward関数を呼び出し
        x = original_forward(src, src_mask, src_key_padding_mask)
        # Adapter層を通す
        x = attn_adapter(x)
        x = ffn_adapter(x)
        return x
    
    # 新しいforward関数を設定
    transformer_layer.forward = new_forward.__get__(transformer_layer)
    
    return transformer_layer
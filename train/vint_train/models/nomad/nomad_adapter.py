import torch
import torch.nn as nn
from .nomad import NoMaD
from .adapter_modules import AdapterLayer

class NoMaDAdapter(nn.Module):
    """
    Adapter層を組み込んだNOMADモデル
    """
    def __init__(self, base_model: NoMaD, adapter_bottleneck_dim: int):
        super().__init__()
        self.base_model = base_model
        
        # Adapterレイヤーの初期化
        self.vision_adapter = AdapterLayer(
            input_dim=self.base_model.vision_encoder.obs_encoding_size,
            bottleneck_dim=adapter_bottleneck_dim
        )
        
        # ベースモデルのパラメータを凍結
        for param in self.base_model.parameters():
            param.requires_grad = False
            
    def forward(self, obs_img, goal_img, goal_mask=None):
        # ベースモデルのビジョンエンコーダを通す
        obs_encoding = self.base_model.vision_encoder(obs_img, goal_img, goal_mask)
        
        # Adapterを通す
        adapted_encoding = self.vision_adapter(obs_encoding)
        
        # 残りの処理はベースモデルと同じ
        return self.base_model.forward_with_encoding(adapted_encoding)
    
    def get_adapter_parameters(self):
        """Adapterのパラメータのみを返す"""
        return self.vision_adapter.parameters()
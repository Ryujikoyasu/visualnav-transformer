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
            
    def forward(self, obs_img, noisy_actions, timesteps):
        # Twistデータセットでは目標画像を使用しないので、ダミーの目標画像を作成
        batch_size = obs_img.size(0)
        device = obs_img.device
        
        # 最後の観測画像を取得（context_size分の画像から）
        last_obs = obs_img[:, -3:, :, :]  # 最後の3チャンネル
        
        # ダミーの目標画像として最後の観測画像を使用
        goal_img = last_obs  # 同じ画像を使用
        
        # 常に目標をマスク
        goal_mask = torch.ones(batch_size, 1, device=device)
        
        # ベースモデルのビジョンエンコーダを通す
        obs_encoding = self.base_model.vision_encoder(obs_img, goal_img, goal_mask)
        
        # Adapterを通す
        adapted_encoding = self.vision_adapter(obs_encoding)
        
        # ノイズ予測ネットワークを通す
        noise_pred = self.base_model.noise_pred_net(noisy_actions, adapted_encoding, timesteps)
        
        return noise_pred
    
    def get_adapter_parameters(self):
        """Adapterのパラメータのみを返す"""
        return self.vision_adapter.parameters()
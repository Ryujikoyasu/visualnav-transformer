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
            
    def forward(self, obs_img, goal_image, noisy_actions, timesteps):
        batch_size = obs_img.size(0)
        device = obs_img.device
        
        # 1. 現在の観測画像の取得（context_size枚目の画像）
        current_obs = obs_img[:, -3:, :, :]  # 最後の3チャンネル
        
        # 2. obsgoal_imgの作成（現在の観測画像とゴール画像を結合）
        obsgoal_img = torch.cat([current_obs, goal_image], dim=1)  # (B, 6, H, W)
        
        # 3. vision_encoderを通す
        obs_encoding = self.base_model.forward(
            func_name="vision_encoder",
            obs_img=obs_img,      # context_size枚の画像全体（obs_encoderで処理）
            goal_img=obsgoal_img, # 6チャンネル画像（goal_encoderで処理）
            input_goal_mask=torch.ones(batch_size, 1, device=device)
        )
        
        # 4. Adapterを通す
        adapted_encoding = self.vision_adapter(obs_encoding)
        
        # 5. noise_pred_netを通す
        noise_pred = self.base_model.forward(
            func_name="noise_pred_net",
            sample=noisy_actions,
            timestep=timesteps,
            global_cond=adapted_encoding
        )
        
        return noise_pred
    
    def get_adapter_parameters(self):
        """Adapterのパラメータのみを返す"""
        return self.vision_adapter.parameters()
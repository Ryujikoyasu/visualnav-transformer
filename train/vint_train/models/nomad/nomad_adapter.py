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

        # 6チャンネルの目標画像を作成
        obsgoal_img = torch.cat([obs_img[:, -3:, :, :], goal_image], dim=1)  # (B, 6, H, W)

        # 目標マスクを作成（常にマスク）
        goal_mask = torch.ones(batch_size, 1, device=device)

        # vision_encoderを通す
        obs_encoding = self.base_model.forward(
            func_name="vision_encoder",
            obs_img=obs_img,     # 全ての観測画像
            goal_img=obsgoal_img,     # 6チャンネルの目標画像
            input_goal_mask=goal_mask
        )

        # Adapterを通す
        adapted_encoding = self.vision_adapter(obs_encoding)

        # noise_pred_netを通す
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
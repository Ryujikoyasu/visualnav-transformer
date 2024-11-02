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
        batch_size = obs_img.size(0)
        device = obs_img.device

        # 観測画像を3チャンネルずつ分割
        obs_img_split = torch.split(obs_img, 3, dim=1)

        # 1. goal_encoder用の入力を準備
        last_obs = obs_img_split[-1]  # 最後の3チャンネル

        # goal_maskをゼロに設定（ゴールがないことを示す）
        goal_mask = torch.zeros(batch_size, 1, device=device)  # (B, 1)

        # goal_imgをゼロテンソルに設定（チャンネル数を6にする）
        obsgoal_img = torch.zeros(batch_size, 6, last_obs.size(2), last_obs.size(3), device=device)  # (B, 6, H, W)

        # 2. obs_encoder用の入力を準備
        context_imgs = obs_img  # 元の形状のまま使用

        # vision_encoderを通す
        obs_encoding = self.base_model.forward(
            func_name="vision_encoder",
            obs_img=context_imgs,     # 全ての観測画像
            goal_img=obsgoal_img,     # 6チャンネルのゼロテンソル
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
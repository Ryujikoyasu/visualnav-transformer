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
        
        # 観測画像のチャンネル数を確認
        print(f"Input obs_img shape: {obs_img.shape}")
        
        # 観測画像を3チャンネルずつ分割
        num_channels = obs_img.size(1)
        num_splits = num_channels // 3
        obs_img_split = torch.split(obs_img, 3, dim=1)
        
        # 分割結果を確認
        print(f"Number of splits: {len(obs_img_split)}")
        for i, split in enumerate(obs_img_split):
            print(f"Split {i} shape: {split.shape}")
        
        # コンテキスト画像とゴール画像を分離
        context_imgs = obs_img[:, :-3]  # 最後の3チャンネルを除く
        goal_img = obs_img[:, -3:]  # 最後の3チャンネル
        
        # 目標マスクを作成（常にマスク）
        goal_mask = torch.ones(batch_size, 1, device=device)
        
        # vision_encoderを通す
        obs_encoding = self.base_model.forward(
            func_name="vision_encoder",
            obs_img=context_imgs,
            goal_img=goal_img,
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
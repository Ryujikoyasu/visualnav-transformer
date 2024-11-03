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
        print("\n=== NoMaDAdapter Forward Pass Debug ===")
        print(f"Input shapes:")
        print(f"obs_img: {obs_img.shape}")
        print(f"goal_image: {goal_image.shape}")
        print(f"noisy_actions: {noisy_actions.shape}")
        
        batch_size = obs_img.size(0)
        device = obs_img.device
        
        # 1. 観測画像の準備
        context_size = obs_img.size(1) // 3
        obs_img = obs_img.view(batch_size, context_size, 3, obs_img.size(2), obs_img.size(3))
        current_obs = obs_img[:, -1]
        
        # 2. vision_encoderを通す
        obs_encoding = self.base_model.forward(
            func_name="vision_encoder",
            obs_img=obs_img.view(batch_size, -1, obs_img.size(3), obs_img.size(4)),
            goal_img=goal_image,
            input_goal_mask=torch.ones(batch_size, 1, device=device)
        )
        print(f"obs_encoding shape: {obs_encoding.shape}")
        
        # 3. Adapterを通す
        adapted_encoding = self.vision_adapter(obs_encoding)
        print(f"adapted_encoding shape: {adapted_encoding.shape}")
        
        # 4. noise_pred_netを通す
        noise_pred = self.base_model.forward(
            func_name="noise_pred_net",
            sample=noisy_actions,
            timestep=timesteps,
            global_cond=adapted_encoding
        )
        print(f"noise_pred shape: {noise_pred.shape}")
        print("===================================\n")
        
        return noise_pred
    
    def get_adapter_parameters(self):
        """Adapterのパラメータのみを返す"""
        return self.vision_adapter.parameters()
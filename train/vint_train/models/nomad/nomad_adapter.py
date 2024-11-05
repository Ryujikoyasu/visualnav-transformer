import torch
import torch.nn as nn
from .nomad import NoMaD
from .adapter_modules import add_adapter_to_transformer_layer

class NoMaDAdapter(nn.Module):
    """
    Adapter層を組み込んだNOMADモデル
    """
    def __init__(self, base_model: NoMaD, adapter_bottleneck_dim: int):
        super().__init__()
        self.base_model = base_model
        
        # アダプター層をtransformer層に組み込む
        if hasattr(self.base_model.vision_encoder, 'sa_encoder'):
            for layer in self.base_model.vision_encoder.sa_encoder.layers:
                add_adapter_to_transformer_layer(layer, adapter_bottleneck_dim)
        
        # ベースモデルのパラメータを凍結（アダプター層以外）
        for name, param in self.base_model.named_parameters():
            if 'adapter' not in name:
                param.requires_grad = False

    def get_adapter_parameters(self):
        """Adapterのパラメータのみを返す"""
        return (param for name, param in self.named_parameters() if 'adapter' in name)
                
    def load_adapter(self, adapter_path):
        """特定のアダプターをロード"""
        adapter_state_dict = torch.load(adapter_path)
        
        # 現在のアダプターのパラメータのみを更新
        current_state = self.state_dict()
        for k, v in adapter_state_dict.items():
            if 'adapter' in k:
                current_state[k] = v
        self.load_state_dict(current_state)

    def forward(self, obs_img, goal_image, noisy_actions, timesteps):
        print("\n=== NoMaDAdapter Forward Pass Debug ===")
        print(f"Input shapes:")
        print(f"obs_img: {obs_img.shape}")
        print(f"goal_image: {goal_image.shape}")
        print(f"noisy_actions: {noisy_actions.shape}")
        
        batch_size = obs_img.size(0)
        device = obs_img.device
        
        # 観測画像とゴール画像を連結
        obsgoal_img = torch.cat([obs_img, goal_image], dim=1)
        
        # goal_maskを正しく設定
        goal_mask = torch.ones(batch_size, 1, device=device).long().view(-1)  # 1次元のベクトルに変換
        
        # ベースモデルのvision_encoderを呼び出す
        obs_encoding = self.base_model.forward(
            func_name="vision_encoder",
            obs_img=obsgoal_img,  # 6チャネルに連結
            goal_img=goal_image,
            input_goal_mask=goal_mask  # 修正: goal_maskを渡す
        )
        print(f"obs_encoding shape: {obs_encoding.shape}")
        
        # Adapterを通す
        adapted_encoding = obs_encoding
        if hasattr(self.base_model.vision_encoder, 'sa_encoder'):
            for block in self.base_model.vision_encoder.sa_encoder.layers:
                adapted_encoding = block(adapted_encoding)
        print(f"adapted_encoding shape: {adapted_encoding.shape}")
        
        # noise_pred_netを通す
        noise_pred = self.base_model.forward(
            func_name="noise_pred_net",
            sample=noisy_actions,
            timestep=timesteps,
            global_cond=adapted_encoding
        )
        print(f"noise_pred shape: {noise_pred.shape}")
        
        print("===================================\n")
        
        return noise_pred
    
import torch
import torch.nn as nn
from .nomad import NoMaD
from .adapter_modules import AdapterTransformerBlock

class NoMaDAdapter(nn.Module):
    """
    Adapter層を組み込んだNOMADモデル
    """
    def __init__(self, base_model: NoMaD, adapter_bottleneck_dim: int):
        super().__init__()
        self.base_model = base_model
        
        # アダプター層を組み込む
        # ここでは、transformer_blocksが存在しない場合の処理を追加
        if hasattr(self.base_model, 'transformer_blocks'):
            for name, block in self.base_model.transformer_blocks.named_children():
                adapted_block = AdapterTransformerBlock(block, adapter_bottleneck_dim)
                setattr(self.base_model.transformer_blocks, name, adapted_block)
        else:
            # 代替の方法でアダプター層を組み込む
            # 例えば、self.base_modelの他の属性を使用してアダプターを組み込む
            # ここでは仮にself.base_model.blocksを使用する例を示します
            for name, block in self.base_model.blocks.named_children():
                adapted_block = AdapterTransformerBlock(block, adapter_bottleneck_dim)
                setattr(self.base_model.blocks, name, adapted_block)
        
        # ベースモデルのパラメータを凍結（アダプター層以外）
        for name, param in self.base_model.named_parameters():
            if 'adapter' not in name:
                param.requires_grad = False
                
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
        
        # ベースモデルのvision_encoderを呼び出す
        obs_encoding = self.base_model.forward(
            func_name="vision_encoder",
            obs_img=obsgoal_img,  # 6チャネルに連結
            goal_img=goal_image,
            input_goal_mask=torch.ones(batch_size, 1, device=device).long()
        )
        print(f"obs_encoding shape: {obs_encoding.shape}")
        
        # Adapterを通す
        adapted_encoding = obs_encoding
        if hasattr(self.base_model, 'transformer_blocks'):
            for block in self.base_model.transformer_blocks:
                adapted_encoding = block.adapter1(adapted_encoding)
                adapted_encoding = block.adapter2(adapted_encoding)
        else:
            for block in self.base_model.blocks:
                adapted_encoding = block.adapter1(adapted_encoding)
                adapted_encoding = block.adapter2(adapted_encoding)
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
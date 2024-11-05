import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal位置エンコーディング
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

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

class DiffusionAdapterLayer(nn.Module):
    """
    Diffusion Policy用のAdapter層
    """
    def __init__(self, channels: int, bottleneck_channels: int):
        super().__init__()
        self.down_conv = nn.Conv1d(channels, bottleneck_channels, 1)  # 1x1 conv
        self.up_conv = nn.Conv1d(bottleneck_channels, channels, 1)    # 1x1 conv
        self.norm = nn.GroupNorm(8, channels)  # UNetと同じGroupNormを使用
        
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.down_conv(x)
        x = F.mish(x)  # UNetと同じMish活性化関数
        x = self.up_conv(x)
        return x + residual

class DiffusionAdapter(nn.Module):
    """
    Diffusion PolicyのUNetにAdapter層を追加するためのクラス
    各解像度レベルの最初のブロックにのみAdapterを配置
    """
    def __init__(self, base_unet, adapter_bottleneck_dim, down_dims):
        super().__init__()
        self.base_unet = base_unet
        
        # タイムステップエンコーディングの次元を512に
        self.time_embed_dim = 512
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(256),
            nn.Linear(256, 1024),
            nn.Mish(),
            nn.Linear(1024, self.time_embed_dim)
        )
        
        # Global conditioningの次元を合わせる
        self.cond_proj = nn.Linear(256, self.time_embed_dim)
        
        # 各解像度のAdapterの次元を確認
        self.down_adapters = nn.ModuleList([
            DiffusionAdapterLayer(dim, adapter_bottleneck_dim)
            for dim in down_dims  # [64, 128, 256]
        ])
        
        self.mid_adapter = DiffusionAdapterLayer(down_dims[-1], adapter_bottleneck_dim)
        
        self.up_adapters = nn.ModuleList([
            DiffusionAdapterLayer(dim, adapter_bottleneck_dim)
            for dim in reversed(down_dims)  # [256, 128, 64]
        ])
        
        # デバッグ用の次元出力を追加
        print(f"Time embedding dim: {self.time_embed_dim}")
        print(f"Down dims: {down_dims}")
        print(f"Adapter bottleneck dim: {adapter_bottleneck_dim}")
    
    def forward(self, x, timesteps, global_cond):
        # 入力の次元を表示
        print(f"Input x shape: {x.shape}")
        print(f"Global cond shape: {global_cond.shape}")
        
        # Get dimensions
        B, seq_len, C = x.shape
        device = x.device
        
        # Global conditioningの次元を調整
        if global_cond is not None:
            global_cond = self.cond_proj(global_cond)
        
        # 形状の変更を明示的に行う
        x = x.permute(0, 2, 1).contiguous()
        
        # Timestep embedding
        t_emb = self.base_unet.diffusion_step_encoder(timesteps)
        
        # Global conditioning
        if global_cond is not None:
            t_emb = t_emb + global_cond
        
        # Downsampling
        h = []
        for i, down_block in enumerate(self.base_unet.down_modules):
            # 最初のレイヤーとAdapterを適用
            x = down_block[0](x, t_emb)  # 最初のConditionalResidualBlock1D
            x = self.down_adapters[i](x)  # Adapter
            
            # 残りのレイヤーを適用（Adapterなし）
            for layer in down_block[1:-1]:
                x = layer(x, t_emb)
            
            h.append(x)
            # Downsample
            if down_block[-1] is not None:
                x = down_block[-1](x)
        
        # Middle
        x = self.base_unet.mid_modules[0](x, t_emb)
        x = self.mid_adapter(x)
        x = self.base_unet.mid_modules[1](x, t_emb)
        
        # Upsampling
        for i, up_block in enumerate(self.base_unet.up_modules):
            x = torch.cat([x, h.pop()], dim=1)
            
            # 最初のレイヤーとAdapterを適用
            x = up_block[0](x, t_emb)
            x = self.up_adapters[i](x)
            
            # 残りのレイヤーを適用（Adapterなし）
            for layer in up_block[1:-1]:
                x = layer(x, t_emb)
            
            # Upsample
            if up_block[-1] is not None:
                x = up_block[-1](x)
        
        # Final convolution
        x = self.base_unet.final_conv(x)
        
        # Reshape output back: [B, channels, seq_len] -> [B, seq_len, channels]
        x = x.transpose(1, 2)
        
        return x
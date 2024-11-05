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
        self.norm = nn.GroupNorm(8, channels)  # channelsを正しく設定
        
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.down_conv(x)
        x = F.mish(x)
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
        
        # タイムステップエンコーディングは独自に生成せず、base_unetのものを使用
        self.time_embed_dim = 512  # デバッグ用に保持
        
        # Global conditioningの次元を合わせる
        self.cond_proj = nn.Linear(256, 512)  # base_unetの次元に合わせる
        
        # 各解像度のAdapterの次元を確認
        self.down_adapters = nn.ModuleList([
            DiffusionAdapterLayer(dim, adapter_bottleneck_dim)
            for dim in down_dims
        ])
        
        self.mid_adapter = DiffusionAdapterLayer(down_dims[-1], adapter_bottleneck_dim)
        
        self.up_adapters = nn.ModuleList([
            DiffusionAdapterLayer(dim, adapter_bottleneck_dim)
            for dim in reversed(down_dims)
        ])
        
        # デバッグ用の次元出力
        print(f"Time embedding dim: {self.time_embed_dim}")
        print(f"Down dims: {down_dims}")
        print(f"Adapter bottleneck dim: {adapter_bottleneck_dim}")
    
    def forward(self, x, timesteps, global_cond):
        # 入力の次元を表示
        print(f"Input x shape: {x.shape}")
        print(f"Global cond shape: {global_cond.shape}")
        
        # タイムステップエンコーディング
        t_emb = self.base_unet.diffusion_step_encoder(timesteps)
        print(f"t_emb shape: {t_emb.shape}")  # デバッグ出力
        
        # Global conditioningの次元を調整（Noneチェックを明示的に）
        if global_cond is not None:
            global_cond = self.cond_proj(global_cond)
            print(f"projected global_cond shape: {global_cond.shape}")  # デバッグ出力
            
            # t_embの次元をglobal_condに合わせる
            t_emb = torch.cat([t_emb, torch.zeros_like(global_cond[:, :256])], dim=1)
            print(f"expanded t_emb shape: {t_emb.shape}")  # デバッグ出力
            
            t_emb = t_emb + global_cond
        
        # 形状の変更を明示的に行う
        x = x.permute(0, 2, 1).contiguous()
        
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
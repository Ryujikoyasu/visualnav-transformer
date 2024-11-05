import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        # ダウンサンプリング層のAdapters（各解像度の最初のブロックのみ）
        self.down_adapters = nn.ModuleList([
            DiffusionAdapterLayer(dim, adapter_bottleneck_dim)
            for dim in down_dims
        ])
        
        # 中間層のAdapter（最初のブロックのみ）
        self.mid_adapter = DiffusionAdapterLayer(down_dims[-1], adapter_bottleneck_dim)
        
        # アップサンプリング層のAdapters（各解像度の最初のブロックのみ）
        self.up_adapters = nn.ModuleList([
            DiffusionAdapterLayer(dim, adapter_bottleneck_dim)
            for dim in reversed(down_dims)
        ])
    
    def forward(self, x, timesteps, global_cond):
        # Get dimensions
        B, seq_len, C = x.shape
        device = x.device
        
        # 形状の変更を明示的に行う
        x = x.permute(0, 2, 1).contiguous()  # [B, seq_len, C] -> [B, C, seq_len]
        print(f"Debug - x shape after permute: {x.shape}")  # デバッグ出力を追加
        
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
        
        # 最後に形状を戻す
        x = x.permute(0, 2, 1).contiguous()  # [B, C, seq_len] -> [B, seq_len, C]
        
        return x
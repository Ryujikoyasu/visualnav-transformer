import torch
import torch.nn as nn
import torch.nn.functional as F

class AdapterLayer(nn.Module):
    """
    Adapter層の基本実装
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

class DiffusionAdapter(nn.Module):
    """
    Diffusion PolicyのUNetにAdapter層を追加するためのクラス
    """
    def __init__(self, base_unet, adapter_bottleneck_dim):
        super().__init__()
        self.base_unet = base_unet
        
        # ダウンサンプリング層のAdapters
        self.down_adapters = nn.ModuleList([
            AdapterLayer(dim, adapter_bottleneck_dim)
            for dim in self.base_unet.down_dims
        ])
        
        # 中間層のAdapter
        self.mid_adapter = AdapterLayer(
            self.base_unet.down_dims[-1],
            adapter_bottleneck_dim
        )
        
        # アップサンプリング層のAdapters
        self.up_adapters = nn.ModuleList([
            AdapterLayer(dim, adapter_bottleneck_dim)
            for dim in reversed(self.base_unet.down_dims)
        ])
        
        # ベースモデルのパラメータを凍結
        for param in self.base_unet.parameters():
            param.requires_grad = False
    
    def forward(self, x, timesteps, global_cond):
        """
        Forward pass with adapters
        Args:
            x: input tensor
            timesteps: diffusion timesteps
            global_cond: global conditioning
        """
        # Get dimensions
        B, device = x.shape[0], x.device
        
        # Timestep embedding
        t_emb = self.base_unet.timestep_embedding(timesteps, self.base_unet.time_embed_dim)
        t_emb = self.base_unet.time_embed(t_emb)
        
        # Global conditioning
        if global_cond is not None:
            g_emb = self.base_unet.global_proj(global_cond)
            t_emb = t_emb + g_emb
        
        # Initial projection
        x = self.base_unet.input_proj(x)
        
        # Downsampling
        h = [x]
        for i, (resnet, downsample) in enumerate(self.base_unet.downs):
            # Resnet blocks
            x = resnet(x, t_emb)
            # Apply adapter after each down block
            x = self.down_adapters[i](x)
            # Store for skip connection
            h.append(x)
            # Downsample
            if downsample is not None:
                x = downsample(x)
        
        # Middle
        x = self.base_unet.mid(x, t_emb)
        x = self.mid_adapter(x)
        
        # Upsampling
        for i, (resnet, upsample) in enumerate(self.base_unet.ups):
            # Get skip connection
            x = torch.cat([x, h.pop()], dim=-1)
            # Resnet blocks
            x = resnet(x, t_emb)
            # Apply adapter
            x = self.up_adapters[i](x)
            # Upsample
            if upsample is not None:
                x = upsample(x)
        
        # Final projection
        x = torch.cat([x, h.pop()], dim=-1)
        x = self.base_unet.output_proj(x)
        
        return x
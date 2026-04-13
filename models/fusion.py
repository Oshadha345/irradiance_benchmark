import torch
import torch.nn as nn
import torch.nn.functional as F

class LadderFusion(nn.Module):
    """Temporal-conditioned channel gating for spatial visual features."""
    def __init__(self, visual_channels, temporal_channels):
        super().__init__()
        self.project = nn.Linear(temporal_channels, visual_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, visual, temporal):
        temp_proj = self.project(temporal) # (B, C_v)
        temp_proj = temp_proj.view(temp_proj.shape[0], temp_proj.shape[1], 1, 1)
        gate = self.sigmoid(temp_proj)
        return visual * gate + visual

class LearnableFusionMatrix(nn.Module):
    """Physics-Aware Mixing Matrix discovering cross-level relationships."""
    def __init__(self, num_levels=4, visual_dims=[256, 512, 1024, 1024], temporal_dim=128, fused_dim=128):
        super().__init__()
        self.num_levels = num_levels
        self.mixing_weights = nn.Parameter(torch.eye(num_levels))
        
        self.visual_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, fused_dim, kernel_size=1),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            ) for dim in visual_dims
        ])
        self.temporal_projs = nn.ModuleList([nn.Linear(temporal_dim, fused_dim) for _ in range(num_levels)])
        self.norm = nn.LayerNorm(fused_dim)

    def forward(self, visual_feats, temporal_feats):
        v_vecs = [self.visual_projs[i](feat) for i, feat in enumerate(visual_feats)]
        V_stack = torch.stack(v_vecs, dim=1) # (B, 4, Fused_Dim)
        W = F.softmax(self.mixing_weights, dim=1)
        V_mixed = torch.einsum('tv, bvd -> btd', W, V_stack)
        
        fused_outputs = []
        for i in range(self.num_levels):
            out = self.norm(self.temporal_projs[i](temporal_feats[i]) + V_mixed[:, i, :])
            fused_outputs.append(out)
        return fused_outputs, W
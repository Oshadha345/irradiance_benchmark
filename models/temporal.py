import torch
import torch.nn as nn

class PyramidTCN(nn.Module):
    """Multi-scale temporal convolutions extracting hierarchical physics context."""
    def __init__(self, input_channels=7, embedding_dim=128):
        super().__init__()
        self.embedding = nn.Linear(input_channels, embedding_dim)
        
        def make_branch(k, p):
            return nn.Sequential(
                nn.Conv1d(embedding_dim, embedding_dim, kernel_size=k, padding=p),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )
            
        self.branch1 = make_branch(3, 1)
        self.branch2 = make_branch(5, 2)
        self.branch3 = make_branch(7, 3)
        self.branch4 = make_branch(9, 4)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1) # (B, Emb, T)
        return [
            self.branch1(x).squeeze(-1),
            self.branch2(x).squeeze(-1),
            self.branch3(x).squeeze(-1),
            self.branch4(x).squeeze(-1)
        ]
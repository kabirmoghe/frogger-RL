import torch.nn as nn

class PolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_actions=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, x):
        # x: (batch, input_dim)
        return self.net(x)  # logits
        
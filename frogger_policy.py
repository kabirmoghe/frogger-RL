import torch.nn as nn

class PolicyNet(nn.Module):
    """
    Simple 2-layer MLP Policy network for frogger.
    - input_dim: dimension of the input
    - hidden_dim: dimension of the hidden layer
    - n_actions: number of actions

    Found 2 layers with 128 hidden units each is plenty for this level of complexity.
    """
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
        
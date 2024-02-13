import torch 
from torch import nn

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # 十分简单的网络结构
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1*28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
            )


    def forward(self, x):
        # 线性层，激活，线性层
        return self.model(x)
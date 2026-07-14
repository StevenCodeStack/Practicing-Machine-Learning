import torch
from torch import nn 


class CircleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.double_layers = nn.Sequential(
            nn.Linear(in_features=2,out_features=10,device="cuda"),
            nn.ReLU(),
            nn.Linear(in_features=10,out_features=10,device="cuda"),
            nn.ReLU(),
            nn.Linear(in_features=10,out_features=1,device="cuda")
        )
    def forward(self,x):
        return self.double_layers(x)
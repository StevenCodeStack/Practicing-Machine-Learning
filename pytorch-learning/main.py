import torch
import matplotlib.pyplot as plt
from torch import nn

weight = 0.7
bias = 0.3

start = 1
end = 11
step = 1

X_train = torch.arange(start,end,step).unsqueeze(dim=1)
y_train = X_train * weight + bias

def show_data(X_train=X_train,
              y_train=y_train,
              predictions=None):
    plt.figure(figsize=(10,7))
    plt.scatter(X_train,y_train, c="b", s=4, label="train data")
    plt.scatter(X_train,predictions, c="r", s=4, label="prediction data")
    plt.legend()
    plt.show();

torch.manual_seed(42)

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1,requires_grad=True))

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

model_0 = LinearRegressionModel();

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model_0.parameters(), lr=0.01)
epochs = 10

for epoch in range(epochs):
    model_0.train()
    y_pred = model_0(X_train)
    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"\nEpoch {epoch+1}/{epochs}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Weight: {model_0.weights.item():.4f}")
    print(f"Bias: {model_0.bias.item():.4f}")
import torch
import matplotlib.pyplot as plt
from torch import nn
from pathlib import Path

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

X_train = torch.arange(start,end,step).unsqueeze(dim=1)
y_train = X_train * weight + bias

def show_data(x_values=X_train,
              y_values=y_train,
              predictions=None):
    plt.figure(figsize=(10,7))
    plt.plot(x_values,y_values, c="b", label="train data")
    if predictions is not None:
        plt.plot(x_values,predictions, c="r", label="prediction data")
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
optimizer = torch.optim.SGD(model_0.parameters(), lr=0.001,momentum=0.9)
epochs = 200

epochs_count = []
loss_values = []

for epoch in range(epochs):
    model_0.train()
    y_pred = model_0(X_train)
    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.inference_mode():
        if epoch % 10 == 0:
            epochs_count.append(epoch)
            loss_values.append(loss.item())

with torch.inference_mode():
    trained_data = model_0(X_train)

show_data(x_values=epochs_count, y_values=loss_values)

# Directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# Save Path
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Saving
torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)


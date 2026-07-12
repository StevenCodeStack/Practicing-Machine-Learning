import torch
from torch import nn
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

weight = 0.5
bias = 0.4

start = 0
end = 1
step = 0.02

X_data = torch.arange(start, end, step).unsqueeze(dim=1).to(device)
y_data = X_data * weight + bias

split = int(0.8 * len(X_data))
X_train, y_train, X_test, y_test = X_data[:split], y_data[:split], X_data[split:], y_data[split:]


def show_data(x_values=X_train,
              y_values=y_train,
              predictions=None):
    plt.figure(figsize=(10,7))
    plt.plot(x_values,y_values, c="b", label="train data")
    if predictions is not None:
        plt.plot(x_values,predictions, c="r", label="prediction data")
    plt.legend()
    plt.show();

class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.linear_layer(x)
    
torch.manual_seed(42)    
model_1 = LinearRegressionModelV2()
model_1.to(device)

epochs = 2000
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_1.parameters(),lr=0.001)


with torch.inference_mode():
    y_test_prediction_before_training = model_1(X_test)
    show_data(x_values=X_test.to("cpu"), y_values=y_test.to("cpu"),predictions=y_test_prediction_before_training.to("cpu"))

for epoch in range(epochs):
    model_1.train()
    y_predictions = model_1(X_train)
    loss = loss_fn(y_predictions, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"epoch : {epoch} | Loss : {loss}")

with torch.inference_mode():
    y_test_prediction = model_1(X_test)
    show_data(x_values=X_test.to("cpu"), y_values=y_test.to("cpu"),predictions=y_test_prediction.to("cpu"))
    print(list(model_1.parameters()))

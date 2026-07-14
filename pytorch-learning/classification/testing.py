import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles 
import torch
from torch import nn
import matplotlib.pyplot as plt
from model import CircleModel
from helper_functions import plot_decision_boundary, plot_predictions

X,y = make_circles(1000, noise=0.05,random_state=42)

X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)

X_train, X_test, y_train, y_test = train_test_split(X,y,
													test_size=0.2,
													random_state=42)

X_train = X_train.to("cuda")
X_test = X_test.to("cuda")
y_train = y_train.to("cuda")
y_test = y_test.to("cuda")

model_0 = CircleModel().to("cuda")

# Train
epochs = 2000
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(),lr=0.1)

for epoch in range(epochs):
	model_0.train()

	y_logits = model_0(X_train).squeeze()
	loss = loss_fn(y_logits, y_train)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	if epoch % 100 == 0:
		print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model_0,X_train,y_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model_0,X_test,y_test)
plt.show()
import torch
from linear_regression_model import LinearRegressionModel

LINEAR_REGRESSION_MODEL_PATH = "models/01_pytorch_workflow_model_0.pth"

model = LinearRegressionModel()
model.load_state_dict(torch.load(LINEAR_REGRESSION_MODEL_PATH, weights_only=True))
model.eval()

print(model.state_dict())
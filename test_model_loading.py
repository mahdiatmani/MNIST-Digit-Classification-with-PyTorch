import torch

from model_load import MNISTModel


model = MNISTModel()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()
print(model)
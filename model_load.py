import torch
import torch.nn as nn

# Define the model architecture
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)

# Load the trained model
def load_model(model_path="mnist_model.pth"):
    model = MNISTModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Define a function to predict the digit
def predict_digit(model, image):
    image = torch.tensor(image, dtype=torch.float32).view(1, -1)  # Flatten the image
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

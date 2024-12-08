import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Define the model architecture (as used during training)
def load_model(model_path="mnist_model.pth"):
    model = nn.Sequential(
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((28, 28)).convert("L")  # Resize to 28x28 and convert to grayscale
    image = np.array(image, dtype=np.float32)
    image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0, 1]
    image = torch.tensor(image).view(1, -1)  # Flatten and add batch dimension
    return image

# Streamlit app
st.title("MNIST Digit Prediction")
st.write("This app predicts the digit in a given image using a pre-trained neural network model.")

# Model loading section
if "model" not in st.session_state:
    st.session_state.model = None

model_path = st.text_input("Enter model path:", "mnist_model.pth")

if st.button("Load Model"):
    try:
        st.session_state.model = load_model(model_path)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")

# File uploader for image
uploaded_file = st.file_uploader("Upload an image of a digit", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    if st.button("Predict"):
        if st.session_state.model is None:
            st.error("Please load the model first!")
        else:
            processed_image = preprocess_image(image)
            with torch.no_grad():
                output = st.session_state.model(processed_image)
                _, predicted = torch.max(output, 1)
                st.write(f"Predicted Digit: {predicted.item()}")

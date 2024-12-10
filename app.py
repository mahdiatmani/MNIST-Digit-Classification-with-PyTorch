import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

# Function to get test set images
def get_test_set_images(test_set_dir):
    try:
        # Get all image files in the directory
        image_files = [f for f in os.listdir(test_set_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Limit to 5 images
        image_files = image_files[:5]

        # Create a dictionary to store image paths and their display information
        test_images = {}
        for idx, filename in enumerate(image_files, 1):
            full_path = os.path.join(test_set_dir, filename)
            # Open and read image
            img = Image.open(full_path)
            test_images[f"Image {idx}"] = {
                "path": full_path,
                "image": img,
                "filename": filename
            }

        return test_images
    except Exception as e:
        st.error(f"Error reading test set images: {e}")
        return {}

# Define the model architecture (as used during training)
def load_model(model_path="mnist_model.pth"):
    model = nn.Sequential(
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to preprocess the image
def preprocess_image(image):
    try:
        # Resize to 28x28 and convert to grayscale
        image = image.resize((28, 28)).convert("L")

        # Display the preprocessed image for verification
        st.image(image, caption="Preprocessed Image (28x28 Grayscale)", width=150)

        # Normalize the image
        image_array = np.array(image, dtype=np.float32)
        image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())

        # Add visualization of the normalized image
        plt.figure(figsize=(5, 5))
        plt.imshow(image_array, cmap='gray')
        plt.title("Normalized Image")
        st.pyplot(plt)
        plt.close()

        # Convert to tensor
        image_tensor = torch.tensor(image_array).view(1, -1)
        return image_tensor
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# Streamlit app
def main():
    st.title("MNIST Digit Prediction")
    st.write("Select an image from the test set for prediction")

    # Input for test set directory
    test_set_dir = st.text_input("Enter test set directory path:", "testSet")

    # Load model section
    model_path = st.text_input("Enter model path:", "mnist_model.pth")

    # Load model button
    if st.button("Load Model"):
        model = load_model(model_path)
        if model is not None:
            st.session_state.model = model
            st.success("Model loaded successfully!")

    # Get test set images
    if st.button("Load Test Set Images"):
        try:
            # Load and display test set images
            st.session_state.test_images = get_test_set_images(test_set_dir)

            # Display thumbnails of test images
            cols = st.columns(5)
            for idx, (label, img_info) in enumerate(st.session_state.test_images.items()):
                with cols[idx]:
                    st.image(img_info['image'], caption=img_info['filename'], width=150)
        except Exception as e:
            st.error(f"Error loading test set images: {e}")

    # Image selection and prediction
    if hasattr(st.session_state, 'test_images'):
        selected_image = st.selectbox(
            "Choose an image to predict",
            list(st.session_state.test_images.keys())
        )

        # Display selected image
        selected_img_info = st.session_state.test_images[selected_image]
        st.image(selected_img_info['image'], caption=f"Selected: {selected_img_info['filename']}", use_column_width=True)

        # Prediction section
        if st.button("Predict Digit"):
            # Check if model is loaded
            if 'model' not in st.session_state or st.session_state.model is None:
                st.error("Please load the model first!")
            else:
                # Preprocess image
                processed_image = preprocess_image(selected_img_info['image'])

                if processed_image is not None:
                    try:
                        # Perform prediction
                        with torch.no_grad():
                            output = st.session_state.model(processed_image)
                            probabilities = torch.softmax(output, dim=1)
                            _, predicted = torch.max(output, 1)

                            # Display prediction
                            st.write(f"🔢 Predicted Digit: {predicted.item()}")

                            # Show probability distribution
                            st.write("Probability Distribution:")
                            prob_dict = {i: float(probabilities[0][i]) for i in range(10)}
                            st.bar_chart(prob_dict)
                    except Exception as e:
                        st.error(f"Prediction error: {e}")

# Run the app
if __name__ == "__main__":
    main()

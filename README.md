# MNIST Digit Classification with PyTorch

## Project Overview
This project implements a neural network for digit classification using the MNIST dataset. The model is built with PyTorch and achieves approximately 97% accuracy in recognizing handwritten digits.

## Features
- Neural network with multiple layers
- Data normalization
- Train/validation/test split
- Cross-entropy loss
- Stochastic Gradient Descent (SGD) optimizer

## Requirements
- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- scikit-learn

## Installation
```bash
pip install torch numpy matplotlib scikit-learn
```

## Dataset
- MNIST-120k dataset
- 120,000 handwritten digit images
- 28x28 pixel grayscale images
- 10 classes (digits 0-9)

## Model Architecture
- Input Layer: 784 neurons (28x28 flattened image)
- Hidden Layer 1: 128 neurons with ReLU activation
- Hidden Layer 2: 64 neurons with ReLU activation
- Output Layer: 10 neurons (one per digit)

## Training Parameters
- Epochs: 15
- Batch Size: 32
- Learning Rate: 0.001
- Optimizer: SGD with momentum

## Performance
- Training Accuracy: ~97%
- Validation Accuracy: 97.15%
- Test Accuracy: 97.00%

## Usage
```python
# Load model and test an image
prediction = test_single_image(model, single_image)
```

## Future Improvements
- Experiment with deeper architectures
- Try different optimizers
- Implement data augmentation

## License
MIT License

## Acknowledgments
- MNIST Dataset
- PyTorch Community
```

Would you like me to modify anything in the README?

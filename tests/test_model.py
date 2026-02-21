import torch
from src.models.model import SimpleCNN

def test_model_forward():
    model = SimpleCNN()

    # Dummy input (batch_size=1, 3 channels, 224x224)
    x = torch.randn(1, 3, 224, 224)

    output = model(x)

    # Check output shape
    assert output.shape == (1, 2)
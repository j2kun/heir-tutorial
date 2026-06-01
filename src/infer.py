import os
import torch
from torchvision import datasets
from torchvision.transforms import v2
from src.model import NeuralNetwork

def main():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    model = NeuralNetwork().to(device)
    
    # Resolve model path
    workspace_dir = os.environ.get('BUILD_WORKSPACE_DIRECTORY')
    if workspace_dir:
        model_path = os.path.join(workspace_dir, 'model.pth')
        data_root = os.path.join(workspace_dir, 'data')
    else:
        model_path = 'model.pth'
        data_root = 'data'

    if not os.path.exists(model_path):
         print(f"Model file not found at {model_path}. Please run train first.")
         return

    model.load_state_dict(torch.load(model_path, weights_only=True))
    print(f"Loaded model from {model_path}")

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    # Download test data from open datasets
    test_data = datasets.FashionMNIST(
        root=data_root,
        train=False,
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    )

    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        x = x.to(device)
        # Add batch dimension
        x = x.unsqueeze(0)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')

if __name__ == "__main__":
    main()

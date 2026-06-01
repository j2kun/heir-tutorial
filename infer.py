import os
import random
import torch
from torchvision import datasets
from torchvision.transforms import v2
from model import NeuralNetwork


def main():
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    model = NeuralNetwork().to(device)

    # Resolve model path
    workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if workspace_dir:
        model_path = os.path.join(workspace_dir, "model.pth")
        data_root = os.path.join(workspace_dir, "data")
    else:
        model_path = "model.pth"
        data_root = "data"

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

    # Download training data from open datasets
    training_data = datasets.FashionMNIST(
        root=data_root,
        train=True,
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    )

    model.eval()
    
    num_samples = 100
    indices = random.sample(range(len(training_data)), num_samples)
    
    correct = 0
    with torch.no_grad():
        for i in indices:
            x, y = training_data[i][0], training_data[i][1]
            x = x.to(device)
            # Add batch dimension
            x = x.unsqueeze(0)
            pred = model(x)
            predicted, actual = classes[pred[0].argmax(0)], classes[y]
            if predicted == actual:
                correct += 1
            print(f'Predicted: "{predicted}", Actual: "{actual}"')
            
    print(f"\nAccuracy on 100 random training samples: {correct}/{num_samples} ({correct/num_samples*100:.1f}%)")


if __name__ == "__main__":
    main()

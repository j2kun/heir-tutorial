import os
import random
import torch
from torchvision import datasets
from torchvision.transforms import v2
from model import FashionMNIST


def main():
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    model = FashionMNIST().to(device)

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
    
    activations = {}
    def get_activation(name):
        def hook(module, input, output):
            if name not in activations:
                activations[name] = []
            activations[name].append(output.detach().cpu())
        return hook

    hooks = []
    for name, module in model.named_modules():
        if name != "" and name != "linear_relu_stack":
            hooks.append(module.register_forward_hook(get_activation(name)))
    
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

    for hook in hooks:
        hook.remove()

    print("\nQuantiles of intermediate values:")
    for name, acts in activations.items():
        all_acts = torch.cat(acts, dim=0)
        flat_acts = all_acts.view(-1).float()
        quantiles = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        q_vals = torch.quantile(flat_acts, quantiles)
        print(f"Node: {name}")
        print(f"  0% (min): {q_vals[0]:.4f}")
        print(f" 25%:       {q_vals[1]:.4f}")
        print(f" 50% (med): {q_vals[2]:.4f}")
        print(f" 75%:       {q_vals[3]:.4f}")
        print(f"100% (max): {q_vals[4]:.4f}")


if __name__ == "__main__":
    main()

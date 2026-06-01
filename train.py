import os
import struct
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from model import FashionMNIST


def load_fashion_mnist(root, train=True):
    if train:
        images_path = os.path.join(root, "train-images-idx3-ubyte")
        labels_path = os.path.join(root, "train-labels-idx1-ubyte")
    else:
        images_path = os.path.join(root, "t10k-images-idx3-ubyte")
        labels_path = os.path.join(root, "t10k-labels-idx1-ubyte")

    with open(images_path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, 1, rows, cols)
        images = torch.from_numpy(images).float() / 255.0

    with open(labels_path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
        labels = torch.from_numpy(labels).long()

    return TensorDataset(images, labels)


def main():
    workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if workspace_dir:
        data_root = os.path.join(workspace_dir, "data")
    else:
        data_root = "data"

    training_data = load_fashion_mnist(data_root, train=True)
    test_data = load_fashion_mnist(data_root, train=False)

    batch_size = 64

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    model = FashionMNIST().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = 25
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)
    print("Done!")

    # Save model
    workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if workspace_dir:
        save_path = os.path.join(workspace_dir, "model.pth")
    else:
        save_path = "model.pth"

    torch.save(model.state_dict(), save_path)
    print(f"Saved PyTorch Model State to {save_path}")


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


if __name__ == "__main__":
    main()

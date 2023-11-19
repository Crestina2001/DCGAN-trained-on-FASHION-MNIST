import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_fashion_mnist(batch_size=64, transform=None):
    # Define a transform if not specified
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    # Load Fashion MNIST training data
    train_data = datasets.FashionMNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )

    # Load Fashion MNIST test data
    test_data = datasets.FashionMNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

import os
import random
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim

# Import the model
from models.Xception import Xception


if __name__ == "__main__":
    # Configure data paths
    train_dir = 'Your train data'
    test_dir = 'Your test data'

    # Data preprocessing
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]),
        'test': transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    }

    # Load datasets
    train_dataset = datasets.ImageFolder(
        train_dir, transform=data_transforms['train'])
    test_dataset = datasets.ImageFolder(
        test_dir, transform=data_transforms['test'])

    # Split the training dataset into training and validation sets
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(0.1 * num_train)
    random.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]

    batch_size = 32

    # Create data samplers for train and validation sets
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    # Create data loaders for train and validation sets
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=val_sampler)

    # Initialize the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Xception(num_classes=1)
    model.to(device)

    # Configure loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(
            f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}')

    # Save the model weights
    os.makedirs("models/weights", exist_ok=True)
    torch.save(model.state_dict(), "models/weights/best_xception_model.pth")
    print("Model saved to models/weights/best_xception_model.pth")

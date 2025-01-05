import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.ResNet18 import RetinaDataset, build_resnet18_model

# Data paths
if __name__ == "__main__":
    train_image_dir = r'Your train data'  # Path to training data
    validation_image_dir = r'Your validation data'  # Path to validation data

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # Datasets and DataLoaders
    train_dataset = RetinaDataset(train_image_dir, transform=transform)
    validation_dataset = RetinaDataset(
        validation_image_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_loader = DataLoader(
        validation_dataset, batch_size=32, shuffle=False)

    # Load the model
    # Build the ResNet-18 model for binary classification
    model = build_resnet18_model(num_classes=2)

    # Choose device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    best_acc = 0.0
    best_model_path = 'models/weights/best_resnet18_model.pth'
    os.makedirs("models/weights", exist_ok=True)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {train_acc:.2f}%")

        # Validation loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total  # Calculate validation accuracy
        print(f"Validation Accuracy: {val_acc:.2f}%")

        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with accuracy: {best_acc:.2f}%")

    print("Training complete.")

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from models import ResUNet
from tqdm import tqdm
from PIL import Image
import numpy as np

# Dataset class for loading images and masks


class RetinaDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None):
        """
        Initialize the dataset by providing image and mask directories.
        Args:
            image_dir (str): Path to the directory containing images.
            mask_dir (str, optional): Path to the directory containing mask images. Defaults to None.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted(
            [f for f in os.listdir(image_dir) if f.endswith("_img.jpg")])
        if mask_dir:

            self.mask_files = sorted(
                [f for f in os.listdir(mask_dir) if f.endswith("_mask.png")])

    def __len__(self):
        """
        Returns the number of images in the dataset.
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Returns the image and corresponding mask (if provided) for a given index.
        Args:
            idx (int): Index of the image in the dataset.
        Returns:
            tuple: (image, mask) if mask_dir is provided, else just (image)
        """
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        image = np.array(image) / 255.0
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        if self.mask_dir:
            mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
            mask = Image.open(mask_path).convert("L")
            mask = np.array(mask) / 255.0
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            return image, mask

        return image

# Training function for the model


def train_model(model, dataloader, criterion, optimizer, device, num_epochs=25, model_path="./model", start_epoch=1):
    """
    Train the model on the provided dataset.
    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): Dataloader for the training dataset.
        criterion (nn.Module): Loss function to use.
        optimizer (Optimizer): Optimizer for training.
        device (torch.device): Device (GPU/CPU) for training.
        num_epochs (int): Number of epochs to train for. Defaults to 25.
        model_path (str): Directory to save model checkpoints. Defaults to "./model".
        start_epoch (int): Epoch to start from (for resuming training). Defaults to 1.
    """
    model.train()
    os.makedirs(model_path, exist_ok=True)

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_loss = 0

        for images, masks in tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.4f}")
        # Save the model weights at the end of each epoch
        torch.save(model.state_dict(), os.path.join(
            model_path, f"resunet_epoch{epoch}.pth"))


# Main execution block
if __name__ == "__main__":
    data_dir = "Your train data"
    dataset = RetinaDataset(image_dir=data_dir, mask_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResUNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train the model for 25 epochs and save the model weights after each epoch
    train_model(model, dataloader, criterion, optimizer, device,
                num_epochs=25, model_path="Your model weight path")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
import scipy.io

from torch.utils.data import Dataset
from models.UNet import TransUnet


# Custom dataset for segmentation tasks
class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        """
        Initialize the dataset with image and mask directories.
        Args:
            images_dir (str): Path to the directory containing images.
            masks_dir (str): Path to the directory containing masks.
            transform (callable, optional): Optional transformations to apply to the images and masks.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        # Load and sort image and mask file names
        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted([file for file in os.listdir(
            masks_dir) if file.endswith('.mat')])  # Only include .mat files

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Load and return a sample from the dataset at the given index.
        Args:
            idx (int): Index of the sample.
        Returns:
            tuple: (image, mask) - The image and corresponding mask.
        """
        img_name = self.images[idx]
        mask_name = self.masks[idx]

        # Ensure that the image and mask names correspond
        assert img_name.split('.')[0] == mask_name.split('.')[0]

        # Load the image and mask
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)
        image = Image.open(img_path).convert('RGB')
        mat = scipy.io.loadmat(mask_path)
        mask = mat['mask']  # Extract the 'mask' field from the .mat file
        mask = Image.fromarray(mask.astype('uint8'), mode='L')

        if self.transform:
            # Apply transformations to the image and mask
            image = self.transform(image)
            mask = transforms.Resize((128, 128))(mask)
            mask = transforms.ToTensor()(mask)
            mask = (mask > 0).float()  # Binarize the mask

        return image, mask


# Dice coefficient for segmentation evaluation
def dice_coeff(pred, target):
    """
    Compute the Dice coefficient for segmentation accuracy.
    Args:
        pred (torch.Tensor): Predicted segmentation mask.
        target (torch.Tensor): Ground truth mask.
    Returns:
        float: Dice coefficient value.
    """
    smooth = 1.0
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


# Define transformations for data augmentation and preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.RandomResizedCrop(size=128, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.1),
    transforms.ToTensor(),
])

# Load the dataset and split into training and validation sets
full_dataset = SegmentationDataset(images_dir='your image data',
                                   masks_dir='your mask data',
                                   transform=transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create data loaders for batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=8,
                          shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8,
                        shuffle=False, num_workers=4)

# Initialize model, loss function, optimizer, and learning rate scheduler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransUnet(img_dim=128,
                  in_channels=3,
                  out_channels=128,
                  head_num=4,
                  mlp_dim=512,
                  block_num=8,
                  patch_dim=16,
                  class_num=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Training loop
num_epochs = 200
best_val_loss = float('inf')
early_stopping_patience = 5
no_improve_epochs = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    # Training step
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    train_loss /= len(train_loader.dataset)  # Average training loss
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}')

    # Validation step
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)

    val_loss /= len(val_loader.dataset)  # Average validation loss
    print(f'Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}')

    # Step the learning rate scheduler
    scheduler.step(val_loss)

    # Save the best model based on validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(),
                   'models/weights/best_transunet_model.pth')
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1

    # Early stopping
    if no_improve_epochs >= early_stopping_patience:
        print("Validation loss did not improve for {} epochs. Early stopping.".format(
            early_stopping_patience))
        break

print("Training completed.")

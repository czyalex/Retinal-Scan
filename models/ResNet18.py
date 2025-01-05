import os
from torch.utils.data import Dataset
from torchvision import models
from PIL import Image
import torch.nn as nn

# Dataset definition


class RetinaDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Initializes the dataset by specifying the image directory and any transformations.

        Args:
            image_dir (str): The directory containing images for the dataset.
            transform (callable, optional): A function to apply transformations to images.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for class_name in ['Negative', 'Positive']:
            class_dir = os.path.join(image_dir, class_name)
            label = 0 if class_name == 'Negative' else 1

            for filename in os.listdir(class_dir):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    image_path = os.path.join(class_dir, filename)
                    self.image_paths.append(image_path)
                    self.labels.append(label)

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves an image and its corresponding label by index.

        Args:
            idx (int): Index of the image to fetch.

        Returns:
            image (PIL.Image): The image at the specified index.
            label (int): The label of the image (0 for Negative, 1 for Positive).
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

# ResNet18 model construction


def build_resnet18_model(num_classes=2):
    """
    Builds a modified ResNet18 model for classification with a custom number of output classes.

    Args:
        num_classes (int): Number of classes in the classification task (default is 2: Negative, Positive).

    Returns:
        model (torch.nn.Module): The ResNet18 model with a modified fully connected layer.
    """
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

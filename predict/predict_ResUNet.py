import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from models.ResUNet import ResUNet


# Initialize and load the model
model = ResUNet()

# Image transformation
transform = transforms.Compose([transforms.ToTensor()])

# Set the model to evaluation mode and move to the correct device
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.load_state_dict(torch.load(
    "models/weights/best_resunet_model.pth", map_location=torch.device('cpu')))


def predict_image(image, threshold=0.5):

    # Load the input image
    image = image.convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Predict the output
    with torch.no_grad():
        output = model(input_tensor)
        output = output.squeeze(0).cpu().numpy()

    # Apply the threshold to binarize the output
    binary_mask = (output > threshold).astype(np.uint8) * 255

    # Convert the binary mask to an image
    result_image = Image.fromarray(binary_mask.squeeze(), mode="L")

    return result_image


if __name__ == "__main__":
    image = "Your input image"  # Path to the input image

    # Make prediction and save the output
    predict_image(image)

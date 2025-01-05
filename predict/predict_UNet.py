import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import logging


from models.UNet import TransUnet

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model (ensure TransUnet is correctly imported or defined)
model = TransUnet(img_dim=128, in_channels=3, out_channels=128, head_num=4,
                  mlp_dim=512, block_num=8, patch_dim=16, class_num=1).to(device)

# Load model weights
model.load_state_dict(torch.load(
    'models/weights/best_transunet_model.pth', map_location=device))
model.eval()

transform = transforms.Compose(
    [transforms.Resize((128, 128)), transforms.ToTensor()])


def process_image(image):
    """
    Input image path and return the cropped image. If there is no mask, print an error message, 
    and visualize the original image, mask, and cropped image.
    """
    try:
        # Load and convert the image
        image = image.convert('RGB')
        input_image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_image)
            # Convert the predicted output to binary mask and scale to 0-255
            pred_mask = (torch.sigmoid(output) >
                         0.5).squeeze().cpu().to(torch.uint8) * 255

        # Convert the mask to PIL image
        pred_mask_image = Image.fromarray(pred_mask.numpy())

        # Resize the mask to the original image size
        resized_pred_mask = pred_mask_image.resize(
            image.size, resample=Image.NEAREST)
        mask_array = np.array(resized_pred_mask)

        # Find the center of the mask
        indices = np.argwhere(mask_array > 0)
        if indices.size == 0:
            print(f"Input image has no mask, the input image is incorrect.")
            return False
        y_coords, x_coords = indices[:, 0], indices[:, 1]
        center_y, center_x = int(y_coords.mean()), int(x_coords.mean())

        # Define the square crop size
        crop_size = 648  # Adjust the crop size as needed

        # Calculate the cropping region coordinates
        half_size = crop_size // 2
        left = max(center_x - half_size, 0)
        upper = max(center_y - half_size, 0)
        right = min(center_x + half_size, image.width)
        lower = min(center_y + half_size, image.height)

        # Adjust cropping region if it exceeds image boundaries
        if right - left < crop_size:
            if left == 0:
                right = min(crop_size, image.width)
            elif right == image.width:
                left = max(image.width - crop_size, 0)
        if lower - upper < crop_size:
            if upper == 0:
                lower = min(crop_size, image.height)
            elif lower == image.height:
                upper = max(image.height - crop_size, 0)

        # Crop the original image
        cropped_image = image.crop((left, upper, right, lower))

        return cropped_image

    except Exception as e:
        logging.error(f"Error during processing: {e}")


# Example usage:
if __name__ == "__main__":
    # Replace with your image path
    image = 'your path to image'
    a = process_image(image)

    print(a)

import torch
from torchvision import transforms
from PIL import Image
from models.Xception import Xception

# Model loading
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Xception(num_classes=1)
model.load_state_dict(torch.load(
    'models/weights/best_xception_model.pth', map_location=torch.device('cpu')))
model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Inference function
def predict_image(image):
    image = image.convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        pred = torch.sigmoid(output).item()
        return 'Positive' if pred > 0.5 else 'Negative'


# Example inference
if __name__ == "__main__":
    image = Image.open(r'path_to_test_image.jpg')
    result = predict_image(image)
    print(f'Prediction result: {result}')

import torch
from torchvision import transforms
from PIL import Image
from models.ResNet18 import build_resnet18_model

# Load the model
model_path = 'models/weights/best_resnet18_model.pth'
model = build_resnet18_model(num_classes=2)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Inference function
def predict_image(image):
    image = image.convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return "Positive" if predicted.item() == 1 else "Negative"


# Example inference
if __name__ == "__main__":
    image = Image.open(r'path_to_test_image.jpg')
    result = predict_image(image)
    print(f"Prediction: {result}")

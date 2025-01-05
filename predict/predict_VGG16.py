import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the model
model_path = "models/weights/vgg16_glaucoma_model.h5"
model = load_model(model_path)
print(f"Model loaded from {model_path}")

# Image preprocessing
IMG_SIZE = (224, 224)


def preprocess_image(image):
    # image = load_img(image_path, target_size=IMG_SIZE)
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = img_to_array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Inference function
def predict_image(image):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)[0][0]
    return "Positive" if prediction > 0.5 else "Negative"


# Example inference
if __name__ == "__main__":
    image_path = "path_to_test_image.jpg"
    image = load_img(image_path, target_size=IMG_SIZE)
    result = predict_image(image)
    print(f"Prediction: {result}")

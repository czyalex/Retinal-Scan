from predict.predict_VGG16 import predict_image as VGG16_pred
from predict.predict_Xception import predict_image as Xception_pred
from predict.predict_ResNet50 import predict_image as Res50_pred
from predict.predict_ResNet18 import predict_image as Res18_pred
from predict.predict_ResUNet import predict_image as vessel_segement
from predict.predict_UNet import process_image as segement
from PIL import Image
from io import BytesIO
import base64
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify
import webbrowser
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("Current Working Directory:", os.getcwd())


app = Flask(__name__, template_folder='GUI', static_folder='GUI/css')

# Determine the final prediction based on results from multiple models


def determine_final_prediction(Res18_result, Res50_result, Xception_result, VGG16_result):
    # Count how many models predicted "Glaucoma"
    glaucoma_count = sum([
        Res18_result == "Positive",
        Res50_result == "Positive",
        Xception_result == "Positive",
        VGG16_result == "Positive"
    ])

    # Determine the final prediction based on the count of "Glaucoma" predictions
    if glaucoma_count == 4:
        final_prediction = "Glaucoma"
    elif glaucoma_count == 3:
        final_prediction = "Large Probability of Glaucoma"
    elif glaucoma_count == 2:
        final_prediction = "Possibly Glaucoma"
    elif glaucoma_count == 1:
        final_prediction = "Little Probability of Glaucoma"
    else:
        final_prediction = "Healthy"

    return final_prediction

# Simulated image processing and prediction function


def predict_raw_image(image):
    # Simulate image processing and prediction
    # Assume segmentation and prediction operations below
    original_image = image  # Original image object
    segmented_disc_cup = segement(image)  # Simulated segmentation result

    if not segmented_disc_cup:
        raise Exception("Segmentation failed.")

    segmented_blood_vessels = vessel_segement(
        image)  # Simulated vessel segmentation

    # Simulate predictions from different models
    Res18_result = Res18_pred(segmented_disc_cup)
    Res50_result = Res50_pred(segmented_disc_cup)
    Xception_result = Xception_pred(segmented_disc_cup)
    VGG16_result = VGG16_pred(segmented_disc_cup)

    # Determine the final prediction
    final_prediction = determine_final_prediction(
        Res18_result, Res50_result, Xception_result, VGG16_result)

    return {
        "originalImage": original_image,  # Original image
        "segmentedDiscCup": segmented_disc_cup,
        "segmentedBloodVessels": segmented_blood_vessels,
        "modelA": Res18_result,
        "modelB": Res50_result,
        "modelC": Xception_result,
        "modelD": VGG16_result,
        "finalPrediction": final_prediction
    }

# Direct prediction without segmentation


def predict_image(image):
    Res18_result = Res18_pred(image)
    Res50_result = Res50_pred(image)
    Xception_result = Xception_pred(image)
    VGG16_result = VGG16_pred(image)

    # Determine the final prediction
    final_prediction = determine_final_prediction(
        Res18_result, Res50_result, Xception_result, VGG16_result)

    return {
        "originalImage": image,
        "modelA": Res18_result,
        "modelB": Res50_result,
        "modelC": Xception_result,
        "modelD": VGG16_result,
        "finalPrediction": final_prediction
    }

# Convert an image to a base64 string


def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")  # Save as PNG format
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_base64


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    prediction_type = request.form.get(
        'prediction_type', 'segmented')  # Default is 'segmented'

    if file:
        img = Image.open(file.stream)

        # Check prediction type and process accordingly
        if prediction_type == 'segmented':
            result = predict_image(img)
        else:
            result = predict_raw_image(img)

        if not result:
            return jsonify({"error": "Unable to process the image. It might be an unsupported format or corrupted."}), 400
        else:
            result['originalImage'] = image_to_base64(
                img)  # Convert image to base64
            # Convert segmented images to base64 if segmentation was used
            if 'segmentedDiscCup' in result:
                result['segmentedDiscCup'] = image_to_base64(
                    result['segmentedDiscCup'])
            if 'segmentedBloodVessels' in result:
                result['segmentedBloodVessels'] = image_to_base64(
                    result['segmentedBloodVessels'])

            return jsonify(result)


if __name__ == '__main__':
    # Automatically open the browser before starting Flask
    webbrowser.open("http://127.0.0.1:5000")

    # Start the Flask application
    app.run(debug=True)

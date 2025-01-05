# Software Development Documentation - Glaucoma Prediction Software Based on Retinal Scan Images

## 1. Introduction
Glaucoma is a progressive eye disease that has been a leading cause of irreversible vision loss in clinics for over a decade. Early diagnosis and treatment of glaucoma are crucial to prevent further visual impairment. By establishing an easy-to-use auxiliary diagnostic platform with high accuracy and efficiency, our software aims to help medical professionals predict the risk of developing glaucoma by analyzing patients' retinal scan images. The operating guide, software design flow, model training process, and future application prospects for software developers, testers, biomedical researchers, and end-users are elaborated in this document.

## 2. Instructions for Use
### 2.1 Environment Setup

* **Python Version:**  Python >= 3.10 is required.

* **Dependencies:**  Install dependencies from `requirements.txt` using:

   ```bash
   pip install -r requirements.txt
   ```

* **Unzip models:**  Decompress the files in "releases" into the "models\weights" folder.

### 2.2 Interactive Interface

![image](https://github.com/BIA4-course/2024-25-Group-04/blob/main/images/user_interface.png)

* **Image Upload**: Users can upload retinal scan images via a simple table.
* **Prediction Type Selection**: Users can choose the prediction type based on the input image. If the uploaded image is a raw, unprocessed retinal scan, select the 'Raw Image' type. If the uploaded image is a segmented retinal scan, select the 'Segmented Image' type.
* **Prediction Conducting**: The software will display the original image, the segmented image (if applicable), and the prediction results from various models.

### 2.3 Results Interpretation

#### 2.3.1 Raw Image Type
![image](https://github.com/BIA4-course/2024-25-Group-04/blob/main/images/raw_image_type.png)

* **Image Preprocessing:** The system receives the uploaded raw retinal images and performs necessary preprocessing steps, such as resizing and contrast enhancement, to standardize the images.

* **Optic Disc and Vessel Segmentation:** A neural network is used to segment the optic disc, optic cup, and blood vessels. The results are displayed in the "Segmented Disc and Cup" and "Segmented Blood Vessels" sections.

* **Classification according to Two Features:** A trained model (e.g., ResNet 18, ResNet 50, Xception, VGG16) is used to predict the health status of the optic disc and blood vessels. The prediction results from each model are displayed individually.

* **Result Display:** Based on the predictions from various models, three models predict "Negative," while one model predicts "Positive." The final result is "Little Probability of Glaucoma."

#### 2.3.2 Segmented Image Type
![image](https://github.com/BIA4-course/2024-25-Group-04/blob/main/images/segmented_image_type.png)

* **Image Preprocessing:** The system receives the uploaded pre-cropped retinal images and performs necessary preprocessing steps, such as resizing and contrast enhancement, to standardize the images.

* **Direct Classification according to Segmented Images**: A trained model (e.g., ResNet 18, ResNet 50, Xception, VGG16) is used to predict the positive and negative status for glaucoma. The prediction results from each model are displayed individually.

* **Result Display**: Based on the predictions from various models, all four models predict "Negative," and the final result is "Healthy."

#### 2.3.3 Final Prediction Principle
| **Number of Models Predicting Positive** | **Number of Models Predicting Negative** | **Final Prediction** |
|:-------:|:-------:|:-------:|
| 4 | 0 | Glaucoma |
| 3 | 1 | Large Probability of Glaucoma |
| 2 | 2 | Possibly Glaucoma |
| 1 | 3 | Little Probability of Glaucoma |
| 0 | 4 | Healthy |


## 3. System Overview

### 3.1 Flow Chart

The detailed workflow of this software are described in the graph below:
![workflow](https://github.com/BIA4-course/2024-25-Group-04/blob/main/images/workflow.jpg)

### 3.2 Model Training and Validation

This project utilizes four deep learning models: ResNet18, ResNet50, VGG16, and Xception for glaucoma detection. Below is the key logic for training each model.

#### 3.2.1 Environment Setup
- **Dependencies**: Install the necessary libraries: `torch`, `torchvision`, `tensorflow`, `matplotlib`.
- **Data Paths**: Set up data directories and ensure that training, validation, and test datasets are organized by class.

#### 3.2.2 Data Preparation & Preprocessing
- **Preprocessing**: Perform standard preprocessing: resize images to a fixed size, normalize images, and apply data augmentation.
- **Data Loading**: 
Use `ImageDataGenerator` in TensorFlow for VGG16.
Use `DataLoader` in PyTorch with `torchvision.transforms` for ResNet18, ResNet50, and Xception.

#### 3.2.3 Model Training Process
- **ResNet18 & ResNet50 (PyTorch)**: Use pre-trained models, `CrossEntropyLoss`, Adam optimizer, and save the best model.
- **VGG16 (TensorFlow/Keras)**: Use `ImageDataGenerator` for data augmentation, `binary_crossentropy`, Adam optimizer, and early stopping callback.
- **Xception (PyTorch)**: 
Use Xception architecture implemented in PyTorch.
Use `BCEWithLogitsLoss` for binary classification.
Adam optimizer, and save the best model.

#### 3.2.4 Training Configuration
- **Learning Rate**: Typically set to 0.0001, can be adjusted based on the model’s convergence.
- **Batch Size**: Use 32, but adjust based on available GPU memory.
- **Epochs**: Train for 10-25 epochs depending on when the model converges.
- **Hardware**: Preferably use GPU for faster training.

#### 3.2.5 Model Evaluation & Saving
- **Evaluation**: After each training epoch, evaluate the model on the validation set to track performance.
- **Saving**: Save the model weights whenever validation accuracy improves.

#### 3.2.6 Two Models for Image Segmentation

- **TransUNet for Disc and Cup Segmentation**
Use Transformer-based U-Net architecture implemented in PyTorch.
Use BCEWithLogitsLoss for binary segmentation.
Adam optimizer, learning rate scheduler, early stopping, and save the best model.

- **ResUNet  for Blood Vessels Segmentation**
Use ResUNet architecture implemented in PyTorch for retinal image segmentation.
Use BCELoss for binary segmentation.
Adam optimizer, save the best model, and perform training with 25 epochs.


## 4. Application Prospects
This software serves as an auxiliary diagnostic tool, enabling doctors to identify glaucoma in its early stages. It enhances the accuracy of diagnostic processes and minimizes the risk of incorrect diagnoses. By enabling earlier treatment interventions, the software significantly contributes to improved patient prognoses. Additionally, it assists patients with retinal examinations to monitor disease progression.

Subsequent software development could focus on the following aspects:
1. Integrating more data types, such as optical coherence tomography, and physiological indicators like retinal nerve fiber layer thickness, to achieve higher prediction accuracy.
2. Adding a function to provide appropriate treatment plans based on the patient's disease stage.
3. Expanding the application range by developing prediction models for other eye diseases, such as age-related macular degeneration and diabetic retinopathy.


## 5. Conclusion
This document provides a detailed introduction to the development and usage of the Glaucoma Prediction Software from retinal scan images. By combining modern web technology and deep learning algorithms, the software is designed to offer medical professionals a powerful auxiliary diagnostic tool. With continuous technological iteration and model optimization, the software is expected to play a significant role in the early diagnosis and treatment of eye diseases.

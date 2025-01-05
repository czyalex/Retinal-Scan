from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout


def build_vgg16_model(input_shape=(224, 224, 3), num_classes=1):
    # Load the VGG16 model (excluding the top fully connected layers)
    base_model = VGG16(weights="imagenet", include_top=False,
                       input_shape=input_shape)

    # Freeze the convolutional layers
    base_model.trainable = False

    # Add custom fully connected layers
    x = Flatten()(base_model.output)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation="sigmoid")(x)

    # Build the model
    model = Model(inputs=base_model.input, outputs=output)
    return model

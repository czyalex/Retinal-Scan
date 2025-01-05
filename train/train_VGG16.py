import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from models.VGG16 import build_vgg16_model

# Data paths
if __name__ == "__main__":
    data_dir = "Path to data"  # data\Train(Validation)(Test)\image.jpg
    train_dir = os.path.join(data_dir, "Train")
    val_dir = os.path.join(data_dir, "Validation")
    test_dir = os.path.join(data_dir, "Test")

    # Image parameters
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32

    # Data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,  # Normalize pixel values to [0, 1]
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,  # Randomly flip images horizontally
    )
    # Only rescaling for validation and test data
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Data generators
    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary"
    )
    val_generator = val_test_datagen.flow_from_directory(
        val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary", shuffle=False
    )

    # Load VGG16 model
    model = build_vgg16_model(input_shape=(224, 224, 3), num_classes=1)

    # Compile the model
    learning_rate = 0.0001
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",  # Binary classification loss
        metrics=["accuracy"],
    )

    # Display model architecture
    model.summary()

    # Define early stopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True)

    # Train the model
    history = model.fit(
        train_generator,
        epochs=25,
        validation_data=val_generator,
        callbacks=[early_stopping],
    )

    # Save the model
    os.makedirs("models/weights", exist_ok=True)
    model.save("models/weights/vgg16_glaucoma_model.h5")
    print("Model saved to models/weights/vgg16_glaucoma_model.h5")

    # Visualize training results
    def plot_training(history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(len(acc))

        plt.figure(figsize=(12, 8))

        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label="Training Accuracy")
        plt.plot(epochs_range, val_acc, label="Validation Accuracy")
        plt.legend(loc="lower right")
        plt.title("Training and Validation Accuracy")

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label="Training Loss")
        plt.plot(epochs_range, val_loss, label="Validation Loss")
        plt.legend(loc="upper right")
        plt.title("Training and Validation Loss")
        plt.show()

    plot_training(history)

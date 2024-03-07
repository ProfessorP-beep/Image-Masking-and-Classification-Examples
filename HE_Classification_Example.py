import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
import os
from PIL import Image

# Step 1: Data Preprocessing
def load_data(data_dir, img_size=(150, 150)):
    X = []
    y = []
    labels = os.listdir(data_dir)
    for label in labels:
        label_dir = os.path.join(data_dir, label)
        for img_file in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_file)
            img = Image.open(img_path)
            img = img.resize(img_size)
            img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
            X.append(img)
            y.append(label)
    X = np.array(X)
    y = np.array(y)
    return X, y

# Step 2: Model Building
def build_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Step 3: Training
def train_model(model, X_train, y_train, X_val, y_val, epochs=10):
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val))

# Step 4: Evaluation
def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')

# Main function
if __name__ == "__main__":
    data_dir = "path_to_your_data_directory"
    X, y = load_data(data_dir)

    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Determine input shape and number of classes
    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(y))

    # Build and compile the model
    model = build_model(input_shape)

    # Train the model
    train_model(model, X_train, y_train, X_val, y_val)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

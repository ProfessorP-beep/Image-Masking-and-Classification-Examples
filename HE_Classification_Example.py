import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
import os
from PIL import Image
from tensorflow.keras.models import load_model
import pandas as pd

# https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images/data

def rename_directories(directory):
    for root, dirs, _ in os.walk(directory):
        for dir_name in dirs:
            if dir_name == "0":
                os.rename(os.path.join(root, dir_name), os.path.join(root, "benign"))
            elif dir_name == "1":
                os.rename(os.path.join(root, dir_name), os.path.join(root, "malignant"))

def load_images_from_directory(directory):
    images = []
    labels = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                image_path = os.path.join(root, file)
                parent_directory = os.path.basename(os.path.dirname(image_path))
                # Assign label based on parent directory name
                label = parent_directory
                images.append(image_path)
                labels.append(label)
    return images, labels

def preprocess_images(image_paths, img_size=(150, 150)):
    images = []
    for image_path in image_paths:
        img = Image.open(image_path)
        img = img.resize(img_size)
        img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
        images.append(img)
    return np.array(images)



# Rename directories from "0" to "benign" and from "1" to "malignant"
main_directory = "C:\\Users\\pwebs\\OneDrive\\Documents\\IDC_regular_ps50_idx5"
rename_directories(main_directory)

# Load all images recursively from the main directory
all_images, all_labels = load_images_from_directory(main_directory)
X = preprocess_images(all_images)
y = np.array(all_labels)

# Convert all images and labels to NumPy arrays
all_images = np.array(all_images)
all_labels = np.array(all_labels)

# Reserve a portion of the images for later use
# You can adjust the percentage of images to reserve as needed
reserve_percentage = 0.30
num_reserved_samples = int(len(all_images) * reserve_percentage)

# Need to maintain the same ratio of samples in reserved as in the training dataset
reserved_images = all_images[:num_reserved_samples]
reserved_labels = all_labels[:num_reserved_samples]

remaining_images = all_images[num_reserved_samples:]
remaining_labels = all_labels[num_reserved_samples:]

# Split the remaining images into training, validation, and test sets while preserving labels
X_train, X_temp, y_train, y_temp = train_test_split(
    remaining_images, remaining_labels, test_size=0.2, random_state=42, stratify=remaining_labels)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

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
        layers.Dense(1, activation='sigmoid')  # Binary classification, so output layer has 1 neuron and sigmoid activation
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',  # Binary crossentropy loss for binary classification
                  metrics=['accuracy'])
    return model

# Step 3: Training and History
def train_model(model, X_train, y_train, X_val, y_val):
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
    return history

# Step 4: Evaluation
def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')

# Main function
if __name__ == "__main__":
   # Convert data to NumPy arrays
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    X_test, y_test = np.array(X_test), np.array(y_test)

    # Determine input shape
    input_shape = X_train.shape[1:]

    # Build and compile the model
    model = build_model(input_shape)

    # Train the model
    history = train_model(model, X_train, y_train, X_val, y_val)

    # Plot training and validation loss
    plot_loss(history)

    # Evaluate the model on validation data
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print("Validation Loss:", val_loss)
    print("Validation Accuracy:", val_accuracy)

    # Evaluate the model on test data
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)

    # Plot validation and test results
    labels = ['Validation', 'Test']
    loss_values = [val_loss, test_loss]
    accuracy_values = [val_accuracy, test_accuracy]

    plt.figure(figsize=(10, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.bar(labels, loss_values, color=['blue', 'green'])
    plt.title('Loss')
    plt.ylabel('Loss')

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.bar(labels, accuracy_values, color=['blue', 'green'])
    plt.title('Accuracy')
    plt.ylabel('Accuracy')

    plt.show()

model.save("breast_cancer_classification.keras")

# Load the pre-trained model
model = load_model("breast_cancer_classification.keras")

# Function to preprocess subdirectory images
def preprocess_images_from_subdirectories(root_dir, img_size=(150, 150)):
    preprocessed_images = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                img_path = os.path.join(subdir, file)
                img = Image.open(img_path)
                img = img.resize(img_size)
                img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
                preprocessed_images.append(img)
    return np.array(preprocessed_images)

benign_dir = "C:\\Users\\pwebs\\OneDrive\\Documents\\10300\\benign_0"
malignant_dir = "C:\\Users\\pwebs\\OneDrive\\Documents\\10300\\malignant_1"

benign_images = preprocess_images_from_subdirectories(benign_dir)
malignant_images = preprocess_images_from_subdirectories(malignant_dir)

# Perform inference
benign_predictions = model.predict(benign_images)
malignant_predictions = model.predict(malignant_images)

# convert probabilities to class labels
benign_labels = [1 if pred >= 0.5 else 0 for pred in benign_predictions]
malignant_labels = [1 if pred >= 0.5 else 0 for pred in malignant_predictions]


# Classify images based on model predictions
def classify_images(predictions):
    classifications = []
    for prediction in predictions:
        if prediction >= 0.5:
            classifications.append("Malignant")
        else:
            classifications.append("Benign")
    return classifications

# Classify benign and malignant images
benign_classifications = classify_images(benign_predictions)
malignant_classifications = classify_images(malignant_predictions)
    
# Count benign and malignant classifications
benign_count = benign_classifications.count("Benign")
malignant_count = malignant_classifications.count("Malignant")

# Create data table for classification results
classification_results = pd.DataFrame({
    "Filename": os.listdir(benign_dir) + os.listdir(malignant_dir),
    "Classification": benign_classifications + malignant_classifications
})

# Plot visualization
plt.figure(figsize=(10, 5))
plt.bar(["Benign", "Malignant"], [benign_count, malignant_count], color=['blue', 'red'])
plt.title('Counts of Benign and Malignant Classifications')
plt.xlabel('Classification')
plt.ylabel('Count')
plt.show()

# Print benign and malignant counts
print("Benign count:", benign_count)
print("Malignant count:", malignant_count)

# Print classification results data table
print("\nClassification Results:")
print(classification_results)
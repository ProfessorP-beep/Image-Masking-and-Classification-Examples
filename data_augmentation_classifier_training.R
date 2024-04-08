# This script is based on a github project by Phillip Webster, PhD. Please Cite (https://github.com/ProfessorP-beep/Image-Masking-and-Classification-Examples/blob/master/Image_Masking_and_Classification.py)
# Contributors: Phillip Webster, PhD., Cera Fisher, PhD.
# Install required packages
install.packages("tidyverse", "imager", "keras", "reticulate")
install.packages("RSNNS")

# Load required libraries
library(imager)
library(tidyverse)
library(keras)
library(png)
library(reticulate)
library(dplyr)
library(plyr)
library(RSNNS)
library(caret)
library(readr)
library(tensorflow)
library(progress)
library(ggplot2)

#Setup tensorflow env.
install_tensorflow(envname = "r-tensorflow")
tf$constant("Hello TensorFlow!")
tensorflow <- import("tensorflow")

#Need to setup scipy
# Set the Python version explicitly
use_python("/path/to/image")
py_install('scipy')
scipy <- import("scipy")
# Access the version string
scipy_version <- scipy$`__version__`

# Print the SciPy version
print(scipy_version)

base_dir <- '/path/to/dir'
sapply(base_dir, function(dir){length(list.files(dir))})

# Function to preprocess images and extractclass labels
# Preprocess Images
preprocess_images <- function(directory, target_size) {
  images <- list.files(directory, pattern = "\\.png$", recursive = TRUE, full.names = TRUE)
  subfolders <- dirname(images)  # Get the subfolder names
  
  loaded_images <- lapply(images, function(image_path) {
    image <- readPNG(image_path)  # You'll need to use appropriate image reading function here
    
    # Check the number of dimensions of the image array
    if (length(dim(image)) == 2) {
      # Grayscale image: add a third dimension for compatibility with resize function
      image <- array(image, dim = c(dim(image), 1, 1))
    } else if (length(dim(image)) == 3) {
      # Color image: add a fourth dimension for compatibility with resize function
      image <- array(image, dim = c(dim(image), 1))
    }
    
    # Resize the image - come back here and add an if statement to adjust resize() arguments for whether the image is greyscale or RGB
    #RGB: Z-scale: 3 (representing the three color channels: red, green, and blue)
    #RGB: Number of vector channels: 1 (since each pixel contains only one of three values: red, green, and blue)
    #Greyscale: Z-scale: 1 cause there is only a single channel
    image <- resize(image, size_x = target_size[1], size_y = target_size[2], size_z = 3, size_c = 1)
    
    return(image)
  })
  
  return(list(images = loaded_images, subfolders = subfolders))
}

training_data <- preprocess_images(base_dir, c(256,256))
training_data_edit <- training_data

# Example usage to extract subfolder names from file paths
# training_data_edit$subfolders <- str_extract(training_data_edit$subfolders, "(?<=/)[^/]+$")
# 
# # Display the modified subfolder names
# print(training_data_edit$subfolders)
# 
# unique_labels <- unique(training_data_edit$subfolders)
# print(unique_labels)
# 
# # Create a mapping dictionary for labels
# label_mapping <- c("body" = 0, "head" = 1, "tail" = 2)
# 
# # Map labels to integer values
# integer_labels <- sapply(training_data_edit$subfolders, function(label) label_mapping[label])
# 
# # Convert integer labels to categorical format
# categorical_labels <- to_categorical(integer_labels)
# Example usage to extract subfolder names from file paths
training_data_edit$subfolders <- str_extract(training_data_edit$subfolders, "(?<=/)[^/]+$")

# Display the modified subfolder names
print(training_data_edit$subfolders)

# # Create a mapping dictionary for labels
# label_mapping <- c("body" = 0, "head" = 1, "tail" = 2)
# 
# # Map labels to integer values
# integer_labels <- sapply(training_data_edit$subfolders, function(label) label_mapping[label])
# 
# # Convert integer labels to categorical format
# categorical_labels <- to_categorical(integer_labels)
# Check unique labels
unique_labels <- unique(training_data_edit$subfolders)
print(unique_labels)

# Create a mapping dictionary for labels
label_mapping <- c("body" = 0, "head" = 1, "tail" = 2)

# Map labels to integer values
integer_labels <- sapply(training_data_edit$subfolders, function(label) label_mapping[label])

# Convert integer labels to categorical format
labels <- to_categorical(integer_labels, num_classes = length(unique_labels))

# Extract dimensions of the images
num_images <- length(training_data_edit$images)
width <- 256
height <- 256
channels <- 3

# Create an empty array to store the image data
images_array <- array(0, dim = c(num_images, width, height, channels))

# Iterate over each image and extract pixel values
for (i in 1:num_images) {
  # Extract the cimg object
  img <- as.cimg(training_data_edit$images[[i]])
  
  # Extract pixel data (assuming RGB images)
  pixels <- apply(img, 1:3, as.vector)  # Flatten the image
  
  # Reshape pixel data to match the desired dimensions
  reshaped_pixels <- array(pixels, dim = c(width, height, channels))
  
  # Store the reshaped pixel data in the images_array
  images_array[i,,,] <- reshaped_pixels
}

# Verify dimensions of the images_array
print(dim(images_array))

# Verify labels and shapes
print(labels)
print(dim(labels))

## Data Augmentation ##
batch_size <- 32
epochs <- 10
steps_per_epoch <- ceiling(length(training_data_edit$images) / batch_size)

# Generate augmented images
# Create an ImageDataGenerator for augmentation
datagen <- image_data_generator(
  rotation_range = 15,
  width_shift_range = 0.1,
  height_shift_range = 0.1,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)

# Generate augmented data generator
augmented_data <- flow_images_from_data(
  x = images_array,
  y = labels,  # Assuming you have labels corresponding to each image
  batch_size = batch_size,
  shuffle = TRUE,
  generator = datagen
)

# Define your neural network model
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", input_shape = c(256, 256, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 3, activation = "softmax")

# Compile the model
 model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_adam(),
  metrics = c("accuracy")
)

# Train the model using augmented data
# For some reason despite confirming the tensorflow-env, python-env, and scipy install
# Keras regular fit() won't work because R can't find scipy.
# I had chatgpt write a training function and just edited it
# Define the number of epochs and steps per epoch
epochs <- 10
steps_per_epoch <- 100

# Create a progress bar for epochs
pb_epochs <- progress_bar$new(total = epochs, format = "Epoch :current/:total")

# Train the model using a custom training loop
history <- list()
train_loss <- c()
train_accuracy <- c()


# Define custom training function
train_model_custom <- function(class_model, train_generator, steps_per_epoch, epochs) {
  pb_epochs <- progress_bar$new(total = epochs, format = "Epoch :current/:total")
  train_loss <- numeric()
  train_accuracy <- numeric()
  batch_losses <- list()
  batch_accuracies <- list()
  
  for (epoch in 1:epochs) {
    pb_epochs$tick()
    cat(sprintf("Epoch %d/%d\n", epoch, epochs))
    
    train_loss_epoch <- numeric()
    train_accuracy_epoch <- numeric()
    
    for (step in 1:steps_per_epoch) {
      batch <- next(train_generator)
      images_batch <- batch[[1]]
      labels_batch <- batch[[2]]
      
      metrics <- class_model %>% train_on_batch(images_batch, labels_batch)
      
      train_loss_epoch <- c(train_loss_epoch, metrics[[1]])
      train_accuracy_epoch <- c(train_accuracy_epoch, metrics[[2]])
    }
    
    avg_loss <- mean(train_loss_epoch)
    avg_accuracy <- mean(train_accuracy_epoch)
    
    train_loss <- c(train_loss, avg_loss)
    train_accuracy <- c(train_accuracy, avg_accuracy)
    
    cat(sprintf("Epoch %d/%d - Avg Loss: %.4f - Avg Accuracy: %.4f\n", 
                epoch, epochs, avg_loss, avg_accuracy))
    
    batch_losses[[epoch]] <- train_loss_epoch
    batch_accuracies[[epoch]] <- train_accuracy_epoch
  }
  
  history <- data.frame(
    epoch = 1:epochs,
    loss = train_loss,
    accuracy = train_accuracy
  )
  
  # Compute overall accuracy and loss
  overall_loss <- mean(unlist(batch_losses))
  overall_accuracy <- mean(unlist(batch_accuracies))
  
  cat(sprintf("Overall Loss: %.4f\n", overall_loss))
  cat(sprintf("Overall Accuracy: %.4f\n", overall_accuracy))
  
  return(history)
}




# Plotting Loss and Accuracy
# To plot the loss and accuracy metrics stored in the history list, you can use ggplot2 for creating line plots. Here's an example of how to generate these plots:

# Convert history to a data frame for plotting



# Smooth plot for loss and accuracy using ggplot2
smooth_plot <- ggplot(data = history_df, aes(x = epoch, y = value, color = metric_type)) +
  geom_line(stat = "smooth", method = "loess", size = 1.5, aes(group = metric_type)) +
  labs(x = "Epoch", y = "Value", color = "Metric Type", title = "Training History") +
  scale_color_manual(values = c("blue", "green")) +
  theme_minimal() +
  theme(legend.position = "top")

# Display the smoothed plot
print(smooth_plot)

# Model Summary
summary(class_model)

#Going to output this as a table
library(knitr)

# Capture model summary as character vector
# Extract the summary data into a data frame
model_summary_df <- as.data.frame(model_summary)

# Export data frame to CSV file
write.csv(model_summary_df, "model_summary.csv", row.names = FALSE)

### Save Model ###
save_model_hdf5(class_model, 'augmented_classcan_classifier.hd5') 

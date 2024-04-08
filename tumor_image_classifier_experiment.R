# This script is part 2 and for using our saved model from data_augmentation_classifier_training.R script
# Load required libraries
library(imager)
library(tidyverse)
library(keras)
library(png)
library(reticulate)
library(dplyr)
library(plyr)
library(tensorflow)
library(RSNNS)
library(caret)


exp_dir <- '/path/to/dir'
img_size <- c(256, 256)



#preprocess_images defined in image_classifier_bk.R
exp_images <- preprocess_images(exp_dir, img_size)
exp_imgs_edit <- exp_images

exp_imgs_edit$subfolders <- str_extract(exp_imgs_edit$subfolders, "(?<=/)[^/]+$")

# Display the modified subfolder names
print(exp_imgs_edit$subfolders)

exp_labels <- unique(exp_imgs_edit$subfolders)
print(exp_labels)

# Create a mapping dictionary for labels
exp_label_mapping <- c("body" = 0, "head" = 1, "tail" = 2)

# Map labels to integer values
# Assuming exp_imgs_edit$subfolders is a list of 345 subfolder names
print(length(exp_imgs_edit$subfolders))  # Should be 345

# Map subfolder names to integer labels for experimental data
exp_integer_labels <- sapply(exp_imgs_edit$subfolders, function(exp_labels) exp_label_mapping[exp_labels])

# Check the length of exp_integer_labels (should be 345)
print(length(exp_integer_labels))  # Should be 345

# Determine the number of unique labels (number of classes)
num_classes <- length(exp_label_mapping)

# Convert integer labels to categorical format (one-hot encoding)
exp_labels <- to_categorical(exp_integer_labels, num_classes = num_classes)

# Verify the length and structure of exp_labels
print(dim(exp_labels))  # Should be (345, num_classes)

# Extract dimensions of the images
num_images <- length(exp_imgs_edit$images)
width <- 256
height <- 256
channels <- 3

# Create an empty array to store the image data
exp_images_array <- array(0, dim = c(num_images, width, height, channels))

# Iterate over each image and extract pixel values
for (i in 1:num_images) {
  # Extract the cimg object
  img <- as.cimg(exp_imgs_edit$images[[i]])
  
  # Extract pixel data (assuming RGB images)
  pixels <- apply(img, 1:3, as.vector)  # Flatten the image
  
  # Reshape pixel data to match the desired dimensions
  reshaped_pixels <- array(pixels, dim = c(width, height, channels))
  
  # Store the reshaped pixel data in the exp_images_array
  exp_images_array[i,,,] <- reshaped_pixels
}

# Verify dimensions of the exp_images_array
print(dim(exp_images_array))

# Verify labels and shapes
print(exp_labels)
print(dim(exp_labels))

# Load Model

tumor_image_classifier <- load_model_hdf5('~/augmented_tumor_classifier.hd5') # After running the data_augmentation training script until its added to git.

exp_predictions <- predict(tumor_image_classifier, exp_images_array, batch_size = 25)

# Extract predicted class labels (assuming categorical output)
predicted_labels <- apply(exp_predictions, 1, which.max) - 1  # Convert to 0-based indexing

# Map predicted labels back to original class names
predicted_class_names <- names(exp_label_mapping)[predicted_labels + 1]

# Display predicted class names
print(predicted_class_names)

library(caret)

predicted_labels_factor <- factor(as.character(predicted_labels), levels = levels(factor(exp_integer_labels)))

str(predicted_labels_factor)
str(exp_integer_labels)
# Print levels of predicted_labels_factor and exp_integer_labels
print(levels(predicted_labels_factor))
print(levels(exp_integer_labels))

# Convert exp_integer_labels to factor (if not already)
exp_integer_labels <- factor(exp_integer_labels, levels = 0:2)

# Check lengths of predicted_labels_factor and exp_integer_labels
print(length(predicted_labels_factor))
print(length(exp_integer_labels))

# Print a sample of predicted_labels_factor and exp_integer_labels
print(head(predicted_labels_factor))
print(head(exp_integer_labels))

# Convert exp_integer_labels to factor with specified levels
exp_factor_labels <- factor(exp_integer_labels, levels = 0:2, labels = c("body", "head", "tail"))

# Print levels and structure of exp_factor_labels
print(levels(exp_factor_labels))
print(str(exp_factor_labels))

# Set levels of predicted_labels_factor to match exp_factor_labels
levels(predicted_labels_factor) <- levels(exp_factor_labels)

print(levels(predicted_labels_factor))
print(levels(exp_factor_labels))

# Create a confusion matrix
confusion_matrix <- confusionMatrix(predicted_labels, exp_integer_labels)

# Plot the confusion matrix
print(confusion_matrix)

# Extract confusion matrix data
conf_mat <- confusion_matrix$table

# Convert confusion matrix to data frame
confusion_df <- as.data.frame.matrix(conf_mat)

# Construct a data frame representing the confusion matrix
confusion_table <- data.frame(
  Prediction = c("body", "head", "tail"),
  body = c(273, 17, 24),
  head = c(14, 1, 1),
  tail = c(14, 0, 1)
)

# Display the confusion matrix table
print("Confusion Matrix:")
print(confusion_table)

# Construct a data frame representing the overall statistics
overall_stats_table <- data.frame(
  Metric = c("Accuracy", "95% CI", "No Information Rate", "P-Value [Acc > NIR]", "Kappa", "Mcnemar's Test P-Value"),
  Value = c(0.7971, "(0.7507, 0.8383)", 0.9101, 1.00, -0.0133, 0.27)
)

# Display the overall statistics table
print("Overall Statistics:")
print(overall_stats_table)

# Construct a data frame representing the statistics by class
class_stats_table <- data.frame(
  Class = c("body", "head", "tail"),
  Sensitivity = c(0.86943, 0.0625, 0.066667),
  Specificity = c(0.09677, 0.948328, 0.924242),
  `Pos Pred Value` = c(0.90698, 0.055556, 0.038462),
  `Neg Pred Value` = c(0.06818, 0.954128, 0.956113),
  Prevalence = c(0.91014, 0.046377, 0.043478),
  `Detection Rate` = c(0.7913, 0.002899, 0.002899),
  `Detection Prevalence` = c(0.87246, 0.052174, 0.075362),
  `Balanced Accuracy` = c(0.4831, 0.505414, 0.495455)
)

# Display the statistics by class table
print("Statistics by Class:")
print(class_stats_table)

write.csv(class_stats_table, 'class_stats_table.csv')


library(png)  # Load the required library for PNG image handling

# Function to plot images with predictions (in color)
plot_images_with_predictions <- function(images_list, predictions, true_labels, target_size = c(256, 256)) {
  par(mfcol = c(5, 5))  # Set up a 5x5 grid of plots
  par(mar = c(0, 0, 1.5, 0), xaxs = 'i', yaxs = 'i')
  
  for (i in 1:20) {
    img <- images_list[[i]]  # Get the image data for the current index
    
    # Resize the image to the target size using imager::resize()
    img_resized <- imager::resize(img, size_x = target_size[1], size_y = target_size[2], size_c = 3, size_z = 1)
    
    # Extract predicted and true labels
    predicted_label <- which.max(predictions[i, ]) - 1  # Convert to 0-based index
    true_label <- true_labels[i]
    
    # Determine plot title and color based on prediction correctness
    if (predicted_label == true_label) {
      color <- '#008800'  # Green color for correct predictions
      prediction_status <- "Correct"
    } else {
      color <- '#bb0000'  # Red color for incorrect predictions
      prediction_status <- "Incorrect"
    }
    
    # Plot the image with the corresponding title
    # Use plot() to display color images
    plot(0:1, 0:1, type = 'n', xlim = c(0, 1), ylim = c(0, 1), xaxt = 'n', yaxt = 'n', xlab = '', ylab = '')
    rasterImage(img_resized, 0, 0, 1, 1, interpolate = FALSE)
    
    # Add title to the plot
    title(main = paste("Predicted:", predicted_label, "(True:", true_label, ")", "Status:", prediction_status),
          col.main = color)
  }
}

# Example usage:
# Assuming you have exp_imgs_edit$images, exp_predictions, and exp_integer_labels defined

# Randomly shuffle the indices (if needed)
set.seed(123)  # Set seed for reproducibility
random_indices <- sample(length(exp_imgs_edit$images))

# Use the function to plot images with predictions
plot_images_with_predictions(exp_imgs_edit$images[random_indices], exp_predictions[random_indices, ], exp_integer_labels[random_indices])

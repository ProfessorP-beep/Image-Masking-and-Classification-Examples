# Example adapted from https://keras.io/examples/vision/masked_image_modeling/
# Written with help from ChatGPT and Gemini
# Data are a subset of 4999 NIH chest X-ray images https://www.kaggle.com/datasets/nih-chest-xrays/data/data
# image folder 
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import deeplake

## Dependencies to Prepare Dataset
from PIL import Image
from keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split

# This subset of chest xrays scan data
    #need to check if the png is a rgb or greyscale format
# Load a sample image
sample_image = cv2.imread("C:\\Users\\pwebs\\OneDrive\\Documents\\google_chest_subset_images\\00000001_000.png", cv2.IMREAD_UNCHANGED)

# Check the shape of the image
print("Image shape:", sample_image.shape) #images are greyscale otherwise they would have shape 1024, 1024, 3


# Function to load images from directory
def load_images(directory, img_size):
    images = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # Load image as grayscale
        image = cv2.resize(image, img_size)  # Resize image to desired dimensions
        images.append(image)
    return images

# Function to load and preprocess images
def preprocess_images(directory, img_size):
    images = load_images(directory, img_size)
    return images

def save_images(images, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    for i, image in enumerate(images):
        filename = f"image_{i}.png"
        output_path = os.path.join(output_directory, filename)
        # Save image as grayscale PNG without embedding color profile
        cv2.imwrite(output_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0], [cv2.IMWRITE_PNG_BILEVEL, 1])

directory = "C:\\Users\\pwebs\\OneDrive\\Documents\\google_chest_subset_images"
output_directory = "C:\\Users\\pwebs\\OneDrive\\Documents\\google_chest_subset_ct_greyscale"
img_size = (224, 224)

save_images(grayscale_images, output_directory)



#Define training
# Setting seeds for reproducibility.
SEED = 42 #random number generation for shuffling data, model parameters, etc. Common practice is to use 42 or 0 as long as it is consistent across runs.
keras.utils.set_random_seed(SEED)

## Hyper Parameter and Pretraining
# DATA
BUFFER_SIZE = 1024 #number of elements from dataset that shuffle buffer will contain
BATCH_SIZE = 256 #number of samples processed in one iteration
AUTO = tf.data.AUTOTUNE #keras function to automatically tune the number of elements that are prefetched
INPUT_SHAPE = (224, 224, 1) #the shape of the input data / images. h x w = 32 x 32 and the last input is the color channel. Since these are CT scans this is set to 1.
NUM_CLASSES = 1 #number of classes or categories for classification. cT scans are typically binary (background vs image) so there is only 1 class (is it the region of interest or not)

# OPTIMIZER
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 1e-4

# PRETRAINING
EPOCHS = 100

# AUGMENTATION
IMAGE_SIZE = 244  # We will resize input images to this size.
PATCH_SIZE = 6  # Size of the patches to be extracted from the input images.
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
MASK_PROPORTION = 0.75  # We have found 75% masking to give us the best results.

# Define inputs for autoencoder
    #tf.data.Dataset.from_tensor_slices This creates a dataset from the training data x_train as a numpy array
    #BUFFER_SIZE shuffled the training dataset to avoid biased learning from ordering of the data
    #BATCH_SIZE batches the shuffled dataset into smaller batch sizes to process simultaneously
    #.prefetch(AUTO) prefetches the batches of data to the GPU or CPU so the model can train on data concurrently with preprocessing and execution
# Load and preprocess images - There are 4999 Lung CT Scan png images in this folder
images = preprocess_images(output_directory, img_size)

# Split data into training and testing sets
x_train, x_test = train_test_split(images, test_size=0.2, random_state=42)

# Split the remaining data into training and validation sets
x_train, x_validate = train_test_split(x_train, test_size=0.25, random_state=42)

# Convert lists of images to NumPy arrays
x_train = np.array(x_train)
x_validate = np.array(x_validate)
x_test = np.array(x_test)

# Print shapes to verify
print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)
print("Validation data shape:", x_validate.shape)

train_ds = tf.data.Dataset.from_tensor_slices(x_train)
train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTO)

val_ds = tf.data.Dataset.from_tensor_slices(x_test)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTO)

test_ds = tf.data.Dataset.from_tensor_slices(x_validate)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTO)

# ENCODER and DECODER
LAYER_NORM_EPS = 1e-6
ENC_PROJECTION_DIM = 128
DEC_PROJECTION_DIM = 64
ENC_NUM_HEADS = 4
ENC_LAYERS = 6
DEC_NUM_HEADS = 4
DEC_LAYERS = (
    2  # The decoder is lightweight but should be reasonably deep for reconstruction.
)
ENC_TRANSFORMER_UNITS = [
    ENC_PROJECTION_DIM * 2,
    ENC_PROJECTION_DIM,
]  # Size of the transformer layers.
DEC_TRANSFORMER_UNITS = [
    DEC_PROJECTION_DIM * 2,
    DEC_PROJECTION_DIM,
]

#Data Augmentation
def get_train_augmentation_model():
    model = keras.Sequential(
        [
            layers.Rescaling(1 / 255.0),
            layers.Lambda(lambda x: tf.image.resize_with_pad(x, IMAGE_SIZE, IMAGE_SIZE)),
            layers.Reshape((IMAGE_SIZE, IMAGE_SIZE, 1)),  # Add channel dimension
            layers.RandomCrop(IMAGE_SIZE, IMAGE_SIZE),
            layers.RandomFlip("horizontal"),
        ],
        name="train_data_augmentation",
    )
    return model

def get_test_augmentation_model():
    model = keras.Sequential(
        [
            layers.Rescaling(1 / 255.0),
            layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
        ],
        name="test_data_augmentation",
    )
    return model

# A layer for extracting patches from images
# Adjust the target shape in the reshape operation to match the expected output dimensions
class Patches(layers.Layer):
    def __init__(self, patch_size=PATCH_SIZE, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        # Create patches from the input images
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )

        # Get the dimensions of the patches tensor
        batch_size = tf.shape(patches)[0]
        num_patches = tf.shape(patches)[1]
        patch_height = tf.shape(patches)[2]
        patch_width = tf.shape(patches)[3]

        # Print the shapes for debugging
        print("Shapes:")
        print("Patches:", patches.shape)
        print("Batch size:", batch_size)
        print("Num patches:", num_patches)
        print("Patch height:", patch_height)
        print("Patch width:", patch_width)

        # Reshape the patches to (batch_size, num_patches, patch_height, patch_width, 1)
        patches = tf.reshape(patches, (batch_size, num_patches, patch_height, patch_width, 1))

        return patches


    def show_patched_image(self, images, patches):
        # This is a utility function which accepts a batch of images and its
        # corresponding patches and helps visualize one image and its patches
        # side by side.
        idx = np.random.choice(patches.shape[0])
        print(f"Index selected: {idx}.")

        plt.figure(figsize=(4, 4))
        plt.imshow(keras.utils.array_to_img(images[idx]))
        plt.axis("off")
        plt.show()

        n = int(np.ceil(np.sqrt(patches.shape[1])))  # Round up the square root
        plt.figure(figsize=(4, 4))
        for i, patch in enumerate(patches[idx]):
            ax = plt.subplot(n, n, i + 1)
            plt.imshow(patch.numpy().astype(np.uint8))  # Convert TensorFlow tensor to NumPy array
            plt.axis("off")
        plt.show()

        # Return the index chosen to validate it outside the method.
        return idx
    
    def reconstruct_from_patch(self, patch):
        # Get the shape information from the patch tensor
        num_patches = patch.shape[0]
        patch_height = patch.shape[1]
        patch_width = patch.shape[2]
        channels = patch.shape[3]

        # Print the shapes for debugging
        print("Patch shape:", patch.shape)
        print("Num patches:", num_patches)
        print("Patch height:", patch_height)
        print("Patch width:", patch_width)
        print("Channels:", channels)

        # Calculate the number of patches per side based on the actual patch height
        num_patches_per_side = int(np.ceil(np.sqrt(num_patches)))

        # Ensure the calculated number of patches per side doesn't exceed the actual patch height and width
        if num_patches_per_side**2 > num_patches:
            num_patches_per_side -= 1

        # Calculate the dimensions of the reconstructed image
        reconstructed_height = num_patches_per_side * patch_height
        reconstructed_width = num_patches_per_side * patch_width

        # Create a zero array to hold the reconstructed image
        reconstructed = np.zeros((reconstructed_height, reconstructed_width, channels))

        # Combine patches into a single image
        for i in range(num_patches_per_side):
            for j in range(num_patches_per_side):
                idx = i * num_patches_per_side + j
                # Handle potential out-of-bounds indexing due to imperfect square calculation
                if idx < num_patches:
                    row_start = i * patch_height
                    row_end = (i + 1) * patch_height
                    col_start = j * patch_width
                    col_end = (j + 1) * patch_width
                    # Print intermediate steps
                print(f"Reconstructing patch {idx} at ({row_start}:{row_end}, {col_start}:{col_end})")
                # Update the reconstructed image
                reconstructed[row_start:row_end, col_start:col_end, :] = patch[idx]


        return reconstructed



# Visualize Patches 
# Get a batch of images.
image_batch = next(iter(train_ds))

# Expand dimensions to add a channel dimension
image_batch = tf.expand_dims(image_batch, axis=-1)

# Now, the shape should be (batch_size, height, width, channels)
print("Shape of image_batch:", image_batch.shape)

# Augment the images.
augmentation_model = get_train_augmentation_model()
augmented_images = augmentation_model(image_batch)

# Define the patch layer.
patch_layer = Patches()

# Get the patches from the batched images.
patches = patch_layer(images=augmented_images)

# Now pass the images and the corresponding patches
# to the `show_patched_image` method.
random_index = patch_layer.show_patched_image(images=augmented_images, patches=patches)

# Chose the same chose image and try reconstructing the patches
# into the original image.
image = patch_layer.reconstruct_from_patch(patches[random_index])
plt.imshow(image)
plt.axis("off")
plt.show()

#close previous figure
plt.close()

## Masking by diving an image into regular non-overlapping patches
    #This defines a custom Keras layer "PathEncoder"
class PatchEncoder(layers.Layer):
    def __init__( #initializes layer with following paramters
        self,
        patch_size=PATCH_SIZE,
        projection_dim=ENC_PROJECTION_DIM,
        mask_proportion=MASK_PROPORTION,
        downstream=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.mask_proportion = mask_proportion
        self.downstream = downstream

        # This is a trainable mask token initialized randomly from a normal
        # distribution.
        self.mask_token = tf.Variable(
            tf.random.normal([1, patch_size * patch_size * 3]), trainable=True
        )

    def build(self, input_shape): #determines shape of input and computes number of patches to be masked
        (_, self.num_patches, self.patch_area) = input_shape

        # Create the projection layer for the patches.
        self.projection = layers.Dense(units=self.projection_dim)

        # Create the positional embedding layer.
        self.position_embedding = layers.Embedding(
            input_dim=self.num_patches, output_dim=self.projection_dim
        )

        # Number of patches that will be masked.
        self.num_mask = int(self.mask_proportion * self.num_patches)

    def call(self, patches):
        # Get the positional embeddings.
        batch_size = tf.shape(patches)[0]
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        pos_embeddings = self.position_embedding(positions[tf.newaxis, ...])
        pos_embeddings = tf.tile(
            pos_embeddings, [batch_size, 1, 1]
        )  # (B, num_patches, projection_dim)

        # Embed the patches.
        patch_embeddings = (
            self.projection(patches) + pos_embeddings
        )  # (B, num_patches, projection_dim)

        if self.downstream:
            return patch_embeddings
        else:
            mask_indices, unmask_indices = self.get_random_indices(batch_size)
            # The encoder input is the unmasked patch embeddings. Here we gather
            # all the patches that should be unmasked.
            unmasked_embeddings = tf.gather(
                patch_embeddings, unmask_indices, axis=1, batch_dims=1
            )  # (B, unmask_numbers, projection_dim)

            # Get the unmasked and masked position embeddings. We will need them
            # for the decoder.
            unmasked_positions = tf.gather(
                pos_embeddings, unmask_indices, axis=1, batch_dims=1
            )  # (B, unmask_numbers, projection_dim)
            masked_positions = tf.gather(
                pos_embeddings, mask_indices, axis=1, batch_dims=1
            )  # (B, mask_numbers, projection_dim)

            # Repeat the mask token number of mask times.
            # Mask tokens replace the masks of the image.
            mask_tokens = tf.repeat(self.mask_token, repeats=self.num_mask, axis=0)
            mask_tokens = tf.repeat(
                mask_tokens[tf.newaxis, ...], repeats=batch_size, axis=0
            )

            # Get the masked embeddings for the tokens.
            masked_embeddings = self.projection(mask_tokens) + masked_positions
            return (
                unmasked_embeddings,  # Input to the encoder.
                masked_embeddings,  # First part of input to the decoder.
                unmasked_positions,  # Added to the encoder outputs.
                mask_indices,  # The indices that were masked.
                unmask_indices,  # The indices that were unmaksed.
            )

    def get_random_indices(self, batch_size):
        # Create random indices from a uniform distribution and then split
        # it into mask and unmask indices.
        rand_indices = tf.argsort(
            tf.random.uniform(shape=(batch_size, self.num_patches)), axis=-1
        )
        mask_indices = rand_indices[:, : self.num_mask]
        unmask_indices = rand_indices[:, self.num_mask :]
        return mask_indices, unmask_indices

    def generate_masked_image(self, patches, unmask_indices):
        # Choose a random patch and it corresponding unmask index.
        idx = np.random.choice(patches.shape[0])
        patch = patches[idx]
        unmask_index = unmask_indices[idx]

        # Build a numpy array of same shape as patch.
        new_patch = np.zeros_like(patch)

        # Iterate of the new_patch and plug the unmasked patches.
        count = 0
        for i in range(unmask_index.shape[0]):
            new_patch[unmask_index[i]] = patch[unmask_index[i]]
        return new_patch, idx

#Test to see the masking process
    # Create the patch encoder layer.
patch_encoder = PatchEncoder()

# Get the embeddings and positions.
(
    unmasked_embeddings,
    masked_embeddings,
    unmasked_positions,
    mask_indices,
    unmask_indices,
) = patch_encoder(patches=patches)


# Show a masked patch image.
new_patch, random_index = patch_encoder.generate_masked_image(patches, unmask_indices)

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
img = patch_layer.reconstruct_from_patch(new_patch)
plt.imshow(keras.utils.array_to_img(img))
plt.axis("off")
plt.title("Masked")
plt.subplot(1, 2, 2)
img = augmented_images[random_index]
plt.imshow(keras.utils.array_to_img(img))
plt.axis("off")
plt.title("Original")
plt.show()

#Multilayer Perceptron (MLP) 
    # a type of feedforward neural network consisting of multiple layers of nodes (neurons),
    #each layer fully connected to the next one.
    #This is a keras MLP from their example.
def mlp(x, dropout_rate, hidden_units):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

# Masking Autoencoder (MAE)
    #Finally we want to define an autoencoder function to reconstruct
    #the image inputs while remaining consistent to the specific type of masking applied.
def create_encoder(num_heads=ENC_NUM_HEADS, num_layers=ENC_LAYERS):
    inputs = layers.Input((None, ENC_PROJECTION_DIM))
    x = inputs

    for _ in range(num_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=ENC_PROJECTION_DIM, dropout=0.1
        )(x1, x1)

        # Skip connection 1.
        x2 = layers.Add()([attention_output, x])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=ENC_TRANSFORMER_UNITS, dropout_rate=0.1)

        # Skip connection 2.
        x = layers.Add()([x3, x2])

    outputs = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x)
    return keras.Model(inputs, outputs, name="mae_encoder")

# MAE decoder - decodes the encoded images from the function above
# def create_decoder(
#     num_layers=DEC_LAYERS, num_heads=DEC_NUM_HEADS, image_size=IMAGE_SIZE
# ):
#     inputs = layers.Input((NUM_PATCHES, ENC_PROJECTION_DIM))
#     x = layers.Dense(DEC_PROJECTION_DIM)(inputs)

#     for _ in range(num_layers):
#         # Layer normalization 1.
#         x1 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x)

#         # Create a multi-head attention layer.
#         attention_output = layers.MultiHeadAttention(
#             num_heads=num_heads, key_dim=DEC_PROJECTION_DIM, dropout=0.1
#         )(x1, x1)

#         # Skip connection 1.
#         x2 = layers.Add()([attention_output, x])

#         # Layer normalization 2.
#         x3 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x2)

#         # MLP.
#         x3 = mlp(x3, hidden_units=DEC_TRANSFORMER_UNITS, dropout_rate=0.1)

#         # Skip connection 2.
#         x = layers.Add()([x3, x2])

#     x = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x)
#     x = layers.Flatten()(x)
#     pre_final = layers.Dense(units=image_size * image_size * 3, activation="sigmoid")(x)
#     outputs = layers.Reshape((image_size, image_size, 3))(pre_final)

#     return keras.Model(inputs, outputs, name="mae_decoder")

def create_simple_decoder(num_layers=DEC_LAYERS, num_heads=DEC_NUM_HEADS, image_size=IMAGE_SIZE):
    inputs = layers.Input((NUM_PATCHES, ENC_PROJECTION_DIM))
    x = layers.Dense(DEC_PROJECTION_DIM)(inputs)

    for _ in range(num_layers):
        # Multi-Head Attention Layer
        x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=DEC_PROJECTION_DIM, dropout=0.1)(x, x)

        # MLP
        x = mlp(x, hidden_units=DEC_TRANSFORMER_UNITS, dropout_rate=0.1)

   # Calculate the output shape
    output_shape = (image_size * image_size * 1,)
    x = layers.Dense(units=output_shape[0], activation="sigmoid")(x)
    x = layers.Reshape((image_size, image_size, 1))(x)

    return keras.Model(inputs, x, name="simple_decoder")

# MAE trainer - trainer module that wraps around the encoder and decoder inside of a tf.keras.model subclass
    #allows customization of model.fit() loop

class MaskedAutoencoder(keras.Model):
    def __init__(
        self,
        train_augmentation_model,
        test_augmentation_model,
        patch_layer,
        patch_encoder,
        encoder,
        decoder,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.train_augmentation_model = train_augmentation_model
        self.test_augmentation_model = test_augmentation_model
        self.patch_layer = patch_layer
        self.patch_encoder = patch_encoder
        self.encoder = encoder
        self.decoder = decoder

    def calculate_loss(self, images, test=False):
        # Augment the input images.
        if test:
            augmented_images = self.test_augmentation_model(images)
        else:
            augmented_images = self.train_augmentation_model(images)

        # Patch the augmented images.
        patches = self.patch_layer(augmented_images)

        # Encode the patches.
        (
            unmasked_embeddings,
            masked_embeddings,
            unmasked_positions,
            mask_indices,
            unmask_indices,
        ) = self.patch_encoder(patches)

        # Pass the unmaksed patche to the encoder.
        encoder_outputs = self.encoder(unmasked_embeddings)

        # Create the decoder inputs.
        encoder_outputs = encoder_outputs + unmasked_positions
        decoder_inputs = tf.concat([encoder_outputs, masked_embeddings], axis=1)

        # Decode the inputs.
        decoder_outputs = self.decoder(decoder_inputs)
        decoder_patches = self.patch_layer(decoder_outputs)

        loss_patch = tf.gather(patches, mask_indices, axis=1, batch_dims=1)
        loss_output = tf.gather(decoder_patches, mask_indices, axis=1, batch_dims=1)

        # Compute the total loss.
        total_loss = self.compute_loss(y=loss_patch, y_pred=loss_output)

        return total_loss, loss_patch, loss_output

    def train_step(self, images):
        with tf.GradientTape() as tape:
            total_loss, loss_patch, loss_output = self.calculate_loss(images)

        # Apply gradients.
        train_vars = [
            self.train_augmentation_model.trainable_variables,
            self.patch_layer.trainable_variables,
            self.patch_encoder.trainable_variables,
            self.encoder.trainable_variables,
            self.decoder.trainable_variables,
        ]
        grads = tape.gradient(total_loss, train_vars)
        tv_list = []
        for grad, var in zip(grads, train_vars):
            for g, v in zip(grad, var):
                tv_list.append((g, v))
        self.optimizer.apply_gradients(tv_list)

        # Report progress.
        results = {}
        for metric in self.metrics:
            metric.update_state(loss_patch, loss_output)
            results[metric.name] = metric.result()
        return results

    def test_step(self, images):
        total_loss, loss_patch, loss_output = self.calculate_loss(images, test=True)

        # Update the trackers.
        results = {}
        for metric in self.metrics:
            metric.update_state(loss_patch, loss_output)
            results[metric.name] = metric.result()
        return results
    
# Masking Model Initialization
train_augmentation_model = get_train_augmentation_model()
test_augmentation_model = get_test_augmentation_model()
patch_layer = Patches()
patch_encoder = PatchEncoder()
encoder = create_encoder()
decoder = create_simple_decoder()

mae_model = MaskedAutoencoder(
    train_augmentation_model=train_augmentation_model,
    test_augmentation_model=test_augmentation_model,
    patch_layer=patch_layer,
    patch_encoder=patch_encoder,
    encoder=encoder,
    decoder=decoder,
)

#

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import backend as K
from scipy.signal import convolve2d
import cv2

# Load and preprocess an example image
img_path = 'img_code/img/img_03.jpg'  # Replace with the path to your image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)

# Calculate the mean values for each channel
mean_b = np.mean(img_array[:, :, 0])
mean_g = np.mean(img_array[:, :, 1])
mean_r = np.mean(img_array[:, :, 2])

# Create a new image by subtracting the mean values
new_b = img_array[:, :, 0] - 123.68
new_g = img_array[:, :, 1] - 116.779
new_r = img_array[:, :, 2] - 103.939

new_image = np.stack((new_b, new_g, new_r), axis=-1)

# Load the VGG16 model
vgg16 = VGG16(weights='imagenet', include_top=False)

# Specify the layer from which you want to extract the kernels
layer_name = 'block1_conv1'  # Example: first convolutional layer in VGG16

# Get the specified layer
layer = vgg16.get_layer(layer_name)

# Get the weights of all kernels (filters) in the layer
kernel_weights = layer.get_weights()[0]  # The first element contains the filter kernel weights

# Initialize a list to store the convolved channels
convolved_channels = []

# Perform convolution for all 64 channels using the extracted kernels

for i in range(64):
    # Perform convolution on each color channel
    convolved_channelB = convolve2d(new_image[:, :, 0], kernel_weights[:, :, 0, i], mode="same", boundary="fill", fillvalue=0)
    convolved_channelG = convolve2d(new_image[:, :, 1], kernel_weights[:, :, 1, i], mode="same", boundary="fill", fillvalue=0)
    convolved_channelR = convolve2d(new_image[:, :, 2], kernel_weights[:, :, 2, i], mode="same", boundary="fill", fillvalue=0)
    
    # Combine the color channels
    # convolved_channel = np.stack((convolved_channelB, convolved_channelG, convolved_channelR), axis=-1)
    # print(convolved_channel.shape)
    image_sum = convolved_channelB + convolved_channelG +convolved_channelR
    
    # Clip negative values to zero
    image_sum = np.maximum(0,image_sum)
    
    # Append the result to the list
    convolved_channels.append(image_sum )


# Stack the channels to create the final convolved image
convolved_image = np.stack(convolved_channels, axis=-1)



print(convolved_channels)

# Display each of the 64 channels in a grid
rows, cols = 8, 8  # Assuming you want an 8x8 grid
fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

for i in range(64):
    ax = axes[i // cols, i % cols]
    ax.imshow(convolved_channels[i])
    ax.axis('off')

plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()


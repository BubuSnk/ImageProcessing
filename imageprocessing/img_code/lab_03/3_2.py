import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import backend as K
import cv2

# Load and preprocess an example image
img_path = 'img_code/img/img_03.jpg'  # Replace with the path to your image
img = image.load_img(img_path, target_size=(224, 224))

# Convert the image to a numpy array
img_array = image.img_to_array(img)

# Expand dimensions to make it 4D (batch size of 1)
img_4D = np.expand_dims(img_array, axis=0)

# Calculate the mean values for each channel
mean_b = np.mean(img_array[:, :, 0])  # Blue channel
mean_g = np.mean(img_array[:, :, 1])  # Green channel
mean_r = np.mean(img_array[:, :, 2])  # Red channel

# Create a new image by subtracting the mean values
new_b = img_array[:, :, 0] - mean_b
new_g = img_array[:, :, 1] - mean_g
new_r = img_array[:, :, 2] - mean_r

# Stack the channels back together to create the new image
new_image = np.stack((new_b, new_g, new_r), axis=-1)

# Display the new image
cv2.imshow('New Image', new_image.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import backend as K

# Load the VGG16 model with pre-trained weights
base_model = VGG16(weights='imagenet', include_top=False)

# Specify the layer for which you want to visualize feature maps
layer_index = 1  # Index of the layer you want to visualize (0-based)

# Create a new model that outputs the activations of the desired layer
layer = base_model.layers[layer_index]
feature_map_model = Model(inputs=base_model.inputs, outputs=layer.output)

# Load and preprocess an example image
img_path = 'img_code/img/img_03.jpg' 
img = image.load_img(img_path, target_size=(224, 224))

# convert the image to an array
img_array = image.img_to_array(img)

# expand dimensions so that it represents a single 'sample
img_array = np.expand_dims(img_array, axis=0)

# prepare the image (e.g. scale pixel values for the vgg)
img_array = preprocess_input(img_array)

# Get the feature map for the input image
feature_maps = feature_map_model.predict(img_array)

# retrieve kernel weights from the 1st Convolutional laye
kernels, biases = base_model.layers[1].get_weights()

# print(img_array.shape)
print(kernels.shape)
for i in range(64):
    print("Kernel # ",i)
    print("*"*25)
    for j in range(3):
        print("-"*25)
        print("Channel # ",j)
        chanel_array = kernels[0,:,:,j]
        print(chanel_array)
        # print(chanel_array.min)
        print("Min Coefficients ",chanel_array.min())
    print("*"*25)

print(biases.shape)
print("Bias =")
print(biases)

print("Model Summary")
print("-="*50)
base_model.summary()
print("-="*50)


# Display the feature maps
num_features = feature_maps.shape[-1]
rows = num_features // 8
cols = 8

plt.figure(figsize=(16, 16))
for i in range(num_features):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(feature_maps[0, :, :, i])  # Choose a colormap
    plt.axis('off')

plt.show()
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

#g(x, y) = af(x, y)^ + b

# Load the image
image = cv2.imread('img_code\img\img_03.jpg')

# Define the linear equation parameters
a_value = 1
b_value = 0
y_value = 0.1

y_step = 0.1

img_num = 10

output_video = cv2.VideoWriter('output2_1_2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 1, image.shape[:2][::-1])


#0 < y < 1
for i in range(img_num):
    
    y = y_value + i * y_step
    new_img = (image.astype(float) ** y * a_value) + b_value

    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255

    new_img = new_img.astype(np.uint8)

     # Separate R, G, and B channels
    r_channel = new_img[:, :, 2]  # Red channel
    g_channel = new_img[:, :, 1]  # Green channel
    b_channel = new_img[:, :, 0]  # Blue channel

    # Normalize channels to 0-255 range
    r_channel_normalized = ((r_channel - np.min(r_channel)) / (np.max(r_channel) - np.min(r_channel))) * 255
    g_channel_normalized = ((g_channel - np.min(g_channel)) / (np.max(g_channel) - np.min(g_channel))) * 255
    b_channel_normalized = ((b_channel - np.min(b_channel)) / (np.max(b_channel) - np.min(b_channel))) * 255

    # Stack normalized channels
    new_img_normalized = np.stack([b_channel_normalized, g_channel_normalized, r_channel_normalized], axis=-1)   
    new_img_normalized = new_img_normalized.astype(np.uint8)

    text = f'y = {y:.2f}'
    cv2.putText(new_img_normalized, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    output_video.write(new_img_normalized)

#y > 1
for i in range(img_num):
    
    y = 0.9 + i * y_step
    new_img = (image.astype(float) ** y * a_value) + b_value 

    # new_img[new_img < 0] = 0
    # new_img[new_img > 255] = 255

    # new_img = new_img.astype(np.uint8)

     # Separate R, G, and B channels
    r_channel = new_img[:, :, 2]  # Red channel
    g_channel = new_img[:, :, 1]  # Green channel
    b_channel = new_img[:, :, 0]  # Blue channel

    # Normalize channels to 0-255 range
    r_channel_normalized = ((r_channel - np.min(r_channel)) / (np.max(r_channel) - np.min(r_channel))) * 255
    g_channel_normalized = ((g_channel - np.min(g_channel)) / (np.max(g_channel) - np.min(g_channel))) * 255
    b_channel_normalized = ((b_channel - np.min(b_channel)) / (np.max(b_channel) - np.min(b_channel))) * 255

    # Stack normalized channels
    new_img_normalized = np.stack([b_channel_normalized, g_channel_normalized, r_channel_normalized], axis=-1)  
    new_img_normalized = new_img_normalized.astype(np.uint8)

    text = f'y = {y:.2f}'
    cv2.putText(new_img_normalized, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    output_video.write(new_img_normalized)


    

output_video.release()




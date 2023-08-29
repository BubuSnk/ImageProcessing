import cv2
import numpy as np
from matplotlib import pyplot as plt

#g(x, y) = af(x, y) + b

# Load the image
image = cv2.imread('img_code\img\img_03.jpg')

# Define the linear equation parameters
a_value = 1
b_value = 0

a_step = 0.01
b_step = 5

img_num = 20

output_video = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 4, image.shape[:2][::-1])


#Fix a increase b
for i in range(img_num):
    b = b_value + i * b_step
    new_img = (image.astype(float) * a_value) + b

    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255

    new_img = new_img.astype(np.uint8)
    output_video.write(new_img)

#Fix a decrease b
for i in range(img_num):
    b = b_value - i * b_step
    new_img = (image.astype(float) * a_value) + b

    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255

    new_img = new_img.astype(np.uint8)
    output_video.write(new_img)

#Fix b increase a
for i in range(img_num):
    a = a_value + i * a_step
    new_img = (image.astype(float) * a ) + b_value

    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255

    new_img = new_img.astype(np.uint8)
    output_video.write(new_img)

#Fix b decrease a
for i in range(img_num):
    a = a_value - i * a_step
    new_img = (image.astype(float) * a ) + b_value

    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255

    new_img = new_img.astype(np.uint8)
    output_video.write(new_img)
    

output_video.release()




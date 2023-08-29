import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math

img_origin = cv2.imread("img/img_03.jpg")
img_gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)

qlevel = pow(2, 4)
Smax = 255

img_quantized = np.floor((img_gray / Smax) * qlevel)


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_gray, cmap='gray')
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(img_quantized, cmap='gray')
plt.title("Quantized Image")

plt.show()

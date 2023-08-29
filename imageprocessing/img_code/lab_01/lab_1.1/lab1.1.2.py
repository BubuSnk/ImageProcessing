import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

img_origin = cv2.imread("img\img_03.jpg")
img_rgb = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)
img_rgb[:, :, 1] = 0
img_rgb[:, :, 0] = 0

img_r = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)


img_r_transposed = np.transpose(img_r)
img_r_moveaxis = np.moveaxis(img_r_transposed, 0, 1)  
img_r_reshape = np.reshape(img_r, (img_r.shape[0], img_r.shape[1]))

plt.figure(figsize=(12, 5))

plt.subplot(1, 4, 1)
plt.imshow(img_r, cmap='gray')
plt.title("R_original")


plt.subplot(1, 4, 2)
plt.imshow(img_r_transposed, cmap='gray')
plt.title("R_Transpose")


plt.subplot(1, 4, 3)
plt.imshow(img_r_moveaxis, cmap='gray')
plt.title("R_move")

plt.subplot(1, 4, 4)
plt.imshow(img_r_reshape, cmap='gray')
plt.title("R_reshape")

plt.show()
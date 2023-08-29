import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

img_origin = cv2.imread("img/img_03.jpg")
img_gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
img_resize = cv2.resize(img_gray, (200, 200))

# Create X and Y coordinate grids for the 2D image
x, y = np.mgrid[0:200, 0:200]

# Define the Z values for the 3D surface plot (use img_resize directly)
z = img_resize

# Create a 3D plot
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(122, projection='3d')

# Use ax.plot_surface to plot the 3D surface
surface = ax.plot_surface(x, y, z, cmap='gray')

# Customize the 3D plot
ax.set_title('3D Surface Plot')
ax.view_init(elev=70, azim=30)

# Show the original and resized 2D images for comparison
plt.subplot(121)
plt.imshow(img_resize, cmap='gray')
plt.title("Original")

plt.show()

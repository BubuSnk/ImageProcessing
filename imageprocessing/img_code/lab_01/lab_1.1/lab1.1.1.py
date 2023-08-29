import cv2
import numpy as np
import math
import matplotlib.image as mpimg


from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

img_path = "img\img_02.jpg"
img_bgr = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)


img_bgr[:,:,1] = 0
img_bgr[:,:,0] = 0

# Hori = np.concatenate((img_bgr, img_rgb), axis=1)
# cv2.imshow("img_hori", Hori)

cv2.waitKey(0)
cv2.destroyAllWindows()

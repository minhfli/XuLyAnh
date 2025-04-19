from filter import *
from math import e
import cv2
import matplotlib
import matplotlib.axes
import numpy as np
import matplotlib.pyplot as plt

from filter import compute_local_mean_var
from plot_helper import plot_img, axs, set_subplot


image_path = "./assets/Lena_noisy.jpeg"
image_name = image_path.split("/")[-1]
output_path = "./assets/output/" + image_name


image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_gray = grayscale(image)

# set_subplot(1, 3)

plot_img(axs[0, 0], image_gray, "Filtered Image", cmap=plt.get_cmap("gray"))

m, v = compute_local_mean_var(image_gray)

plot_img(axs[0, 1], m, "Filtered Image", cmap=plt.get_cmap("gray"))
plot_img(axs[0, 2], v, "Filtered Image", cmap=plt.get_cmap("gray"))

image_gray = np.flip(image_gray, axis=0)
plot_img(axs[1, 0], image_gray, "Filtered Image", cmap=plt.get_cmap("gray"))

m, v = compute_local_mean_var(image_gray)

plot_img(axs[1, 1], m, "Filtered Image", cmap=plt.get_cmap("gray"))
plot_img(axs[1, 2], v, "Filtered Image", cmap=plt.get_cmap("gray"))


plt.tight_layout()
plt.show()

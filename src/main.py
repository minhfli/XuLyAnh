from math import e
import cv2
import matplotlib
import matplotlib.axes
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib.pyplot as plt

from filter import *
from plot_helper import *

"""
change here
"""
# image_path = ( # MRI image
#     "./assets/MRI/ADNI_002_S_1070_130146_ACPC/ADNI_002_S_1070_130146_ACPC_142.jpg"
# )
image_path = "./assets/cloud.jpg"
image_name = image_path.split("/")[-1]
output_path = "./output/" + image_name


image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_gray = grayscale(image)

image_height, image_width = image.shape[:2]

plot_img(axs[0, 0], image, "Original Image")
plot_img(axs[0, 1], image_gray, "Grayscale Image", cmap=plt.get_cmap("gray"))

alpha = 0.5
k = 2  #! k is 5 in the original paper
filtered_image = image_gray
for i in range(k):
    starting_image = get_starting_point_v2(filtered_image, alpha, kernel_size=7)
    # plot_img(axs[0, 2], starting_image, "Starting Image", cmap=plt.get_cmap("gray"))
    filtered_image = edge_preserve_filter_v2(starting_image, alpha, kernel_size=7)

plot_img(
    axs[1, 0],
    filtered_image,
    f"Step 1: Filtered Image, k={k}",
    cmap=plt.get_cmap("gray"),
)

Image_i = filtered_image
Image_d = image_gray - Image_i  # signal difference
plot_img(axs[1, 2], Image_d, "Step2: Signal Difference", cmap=plt.get_cmap("gray"))

M, V = compute_local_mean_var(Image_d)
plot_img(axs[1, 3], M, "Local Mean", cmap=plt.get_cmap("gray"))
plot_img(axs[1, 4], V, "Local Variance", cmap=plt.get_cmap("gray"))

threshold = V * 2.5
Image_m = np.where(np.abs(Image_d - M) < threshold, Image_i, image_gray)
plot_img(axs[2, 0], Image_m, "Step 3: Thresholded image", cmap=plt.get_cmap("gray"))

s = 0.5
Image_o = image_gray - s * Image_m
plot_img(axs[2, 1], Image_o, "Step 5", cmap=plt.get_cmap("gray"))

m = np.mean(Image_o)
v = np.var(Image_o, mean=m)
lower = m - 2.5 * v
upper = m + 2.5 * v
# Io_clipped = np.clip(Image_o, lower, upper)
Io_rescaled = (Image_o - lower) / (upper - lower)
plot_img(axs[2, 2], Io_rescaled, "Step 6: Rescaled", cmap=plt.get_cmap("gray"))
Io_clipped = np.clip(Io_rescaled, lower, upper)
plot_img(
    axs[2, 2], Io_rescaled, "Step 6: Rescaled and clipped", cmap=plt.get_cmap("gray")
)

plt.tight_layout()
plt.show()
pickle.dump(fig, open(output_path + ".pickle", "wb"))
fig.savefig(output_path + ".png")

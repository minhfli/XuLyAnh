from math import e
import cv2
import matplotlib
import matplotlib.axes
import numpy as np
import matplotlib.pyplot as plt
import pickle

from filter import *

# from plot_helper import *
from filter_rgb import *

"""
change here
"""
# image_path = (  # MRI image
#     "./assets/MRI/ADNI_002_S_1070_130146_ACPC/ADNI_002_S_1070_130146_ACPC_142.jpg"
# )
# image_path = (
#     "./assets/MRI/ADNI_002_S_1155_274154_ACPC/ADNI_002_S_1155_274154_ACPC_150.jpg"
# )
# image_path = "./assets/camera_man.jpeg"
# image_path = "./assets/MRI/Test.jpg"
image_path = "./assets/black_white.jpg"
# image_path = "./assets/Lena_noisy.jpeg"
# image_path = "./assets/DIP3E_Problem_Figures/CH10_Problem_Figures/FigP1007(b).tif"
# image_path = "./assets/salt_and_pepper.png"
image_name = image_path.split("/")[-1]
output_path = "./output/" + image_name
k = 3  #! k is 5 in the original paper

gausian_test = True

fig, axs = plt.subplots(2, 4)


def plot_img(ax: matplotlib.axes.Axes, img, title, cmap=None, vmin=0, vmax=255):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")


image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# image = (image / 255.0 - 0.5) * 2
image_gray = image
plot_img(axs[0, 0], image, "Original Image (grayscale but shown in RGB)")
plot_img(axs[1, 0], image_gray, "Grayscale Image", cmap=plt.get_cmap("gray"))

alphas = [0.5, 0.25, 0.125]

index = 0
for alpha in alphas:
    starting_image, beta = get_starting_point_v2(
        image_gray, alpha=alpha, kernel_size=11
    )

    cluster_filtered_image_v3 = cluster_filter_v3(
        image, starting_image, k=k, beta=beta, alpha=alpha, kernel_size=11
    )

    gausian_filtered_image = starting_image

    # plot_img(axs[1, 0], cluster_filtered_image, "Cluster fiter", cmap=plt.get_cmap("gray"))
    index += 1
    plot_img(
        axs[0, index],
        gausian_filtered_image,
        f"Gausian fiter a={alpha}",
        cmap=plt.get_cmap("gray"),
    )

    plot_img(
        axs[1, index],
        cluster_filtered_image_v3,
        f"Cluster fiter a={alpha}",
        cmap=plt.get_cmap("gray"),
    )

plt.tight_layout()
plt.show()

fig.savefig(output_path + "_filter_cmp.png")

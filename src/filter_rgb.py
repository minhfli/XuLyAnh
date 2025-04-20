# edge preseving filter
# icip.1994.413625
# IMAGE ENHANCEMENT BY EDGE-PRESERVING FILTERING
# Yiu-fai Wong

from math import exp
from alive_progress import alive_bar
from matplotlib import pyplot as plt
from matplotlib.pylab import f
import numpy as np

from plot_helper import plot_img, axs


def compute_weight_patch(x, y, min_x, max_x, min_y, max_y, alpha=0.5):
    distance = np.zeros((max_x - min_x, max_y - min_y))
    for i in range(min_x, max_x):
        for j in range(min_y, max_y):
            distance[i - min_x, j - min_y] = (x - i) ** 2 + (y - j) ** 2
    return np.exp(-alpha * distance)


def get_starting_point_v3(image_rgb, image_gray, alpha=0.5, kernel_size=11):
    height, width = image_gray.shape[:2]
    filtered_image = np.zeros((height, width))
    half_kernel = kernel_size // 2

    with alive_bar(height * width, title="Computing y0") as bar:
        for i in range(height):
            for j in range(width):
                min_x = max(0, i - half_kernel)
                max_x = min(height, i + half_kernel + 1)
                min_y = max(0, j - half_kernel)
                max_y = min(width, j + half_kernel + 1)
                patch = image_gray[min_x:max_x, min_y:max_y]
                weight = compute_weight_patch(
                    i, j, min_x, max_x, min_y, max_y, alpha=alpha
                )
                filtered_image[i, j] = np.sum(patch * weight) / np.sum(weight)
                bar()

    return filtered_image


def edge_preserve_filter_v3(image_rgb, image_gray, alpha=0.5, kernel_size=11):
    """
    edge_preserve_filter_v1 calculate weight for all pixel of the image,
    since the weight is close to 0 as the distance is far from the current pixel
    v2 compute weight for only the pixels close to the current pixel

    """

    height, width = image_gray.shape[:2]
    filtered_image = np.zeros((height, width))
    half_kernel = kernel_size // 2

    with alive_bar(height * width, title="Filtering") as bar:
        for i in range(height):
            for j in range(width):
                min_x = max(0, i - half_kernel)
                max_x = min(height, i + half_kernel + 1)
                min_y = max(0, j - half_kernel)
                max_y = min(width, j + half_kernel + 1)
                patch = image_gray[min_x:max_x, min_y:max_y]
                weight = compute_weight_patch(
                    i, j, min_x, max_x, min_y, max_y, alpha=alpha
                )
                mean = np.sum(patch * weight) / np.sum(weight)
                var2 = np.sum((patch - mean) ** 2 * weight) / np.sum(weight)
                if var2 < 0.00001:
                    beta = 1
                else:
                    beta = 1 / (2 * var2)
                    if beta > 1:
                        beta = 1

                tuso = np.sum(
                    patch * weight * np.exp(-beta * (patch - image_gray[i, j]) ** 2)
                )
                mauso = np.sum(weight * np.exp(-beta * (patch - image_gray[i, j]) ** 2))
                filtered_image[i, j] = tuso / mauso
                bar()
    return filtered_image


def compute_local_mean_var(image, kernel_size=40):
    height, width = image.shape[:2]

    local_mean = np.zeros((height, width))
    local_var = np.zeros((height, width))

    half_kernel = kernel_size // 2
    with alive_bar(height * width, title="Step3") as bar:
        for i in range(height):
            for j in range(width):
                x_start = max(0, i - half_kernel)
                x_end = min(height, i + half_kernel + 1)
                y_start = max(0, j - half_kernel)
                y_end = min(width, j + half_kernel + 1)

                patch = image[x_start:x_end, y_start:y_end]
                local_mean[i, j] = np.mean(patch)
                local_var[i, j] = np.var(patch)
                bar()
    return local_mean, local_var

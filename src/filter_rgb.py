# edge preseving filter
# icip.1994.413625
# IMAGE ENHANCEMENT BY EDGE-PRESERVING FILTERING
# Yiu-fai Wong

from math import exp
from alive_progress import alive_bar
from matplotlib import pyplot as plt
from matplotlib.pylab import f
import numpy as np

from .plot_helper import plot_img, axs
from filter import get_starting_point_v2, grayscale


def compute_weight_patch(x, y, min_x, max_x, min_y, max_y, alpha=0.5):
    distance = np.zeros((max_x - min_x, max_y - min_y))
    for i in range(min_x, max_x):
        for j in range(min_y, max_y):
            distance[i - min_x, j - min_y] = (x - i) ** 2 + (y - j) ** 2
    return np.exp(-alpha * distance)


def get_starting_point_v3(image_rgb, image_gray, alpha=0.5, kernel_size=11):
    filtered_image = np.zeros(image_rgb.shape)

    filtered_image[:, :, 0], _ = get_starting_point_v2(
        image_rgb[:, :, 0], alpha=alpha, kernel_size=kernel_size
    )
    filtered_image[:, :, 1], _ = get_starting_point_v2(
        image_rgb[:, :, 1], alpha=alpha, kernel_size=kernel_size
    )
    filtered_image[:, :, 2], _ = get_starting_point_v2(
        image_rgb[:, :, 2], alpha=alpha, kernel_size=kernel_size
    )
    filtered_image_gray, beta = get_starting_point_v2(
        image_gray, alpha=alpha, kernel_size=kernel_size
    )

    return filtered_image, filtered_image_gray, beta


def clustering_filter_rgb(image_rgb, image_gray, alpha=0.5, beta=None, kernel_size=11):
    height, width = image_gray.shape[:2]
    filtered_image = np.zeros(image_rgb.shape)
    half_kernel = kernel_size // 2

    with alive_bar(height * width, title="Filtering") as bar:
        for i in range(height):
            for j in range(width):
                min_x = max(0, i - half_kernel)
                max_x = min(height, i + half_kernel + 1)
                min_y = max(0, j - half_kernel)
                max_y = min(width, j + half_kernel + 1)
                patch = image_gray[min_x:max_x, min_y:max_y]
                patch_rgb = image_rgb[min_x:max_x, min_y:max_y]
                weight = compute_weight_patch(
                    i, j, min_x, max_x, min_y, max_y, alpha=alpha
                )

                tuso0 = np.sum(
                    patch_rgb[:, :, 0]
                    * weight
                    * np.exp(-beta[i, j] * (patch - image_gray[i, j]) ** 2),
                )
                tuso1 = np.sum(
                    patch_rgb[:, :, 1]
                    * weight
                    * np.exp(-beta[i, j] * (patch - image_gray[i, j]) ** 2),
                )
                tuso2 = np.sum(
                    patch_rgb[:, :, 2]
                    * weight
                    * np.exp(-beta[i, j] * (patch - image_gray[i, j]) ** 2),
                )
                mauso = np.sum(
                    weight * np.exp(-beta[i, j] * (patch - image_gray[i, j]) ** 2)
                )
                filtered_image[i, j] = [tuso0, tuso1, tuso2] / mauso
                bar()
    return filtered_image


def edge_preserve_filter_rgb(image_rgb, image_gray, k=5, alpha=0.5, kernel_size=11):

    filtered_rgb, filtered_gray = image_rgb, image_gray
    for i in range(k):
        starting_rgb, starting_gray, beta = get_starting_point_v3(
            filtered_rgb, filtered_gray, alpha, kernel_size=kernel_size
        )
        filtered_rgb = clustering_filter_rgb(
            starting_rgb, starting_gray, alpha, beta, kernel_size=kernel_size
        )
        filtered_gray = grayscale(filtered_rgb)
        if i == 0:
            plot_img(axs[0, 2], starting_rgb, "Starting Image", vmin=0.0, vmax=255.0)
            plot_img(
                axs[0, 3], starting_gray, "Starting Image", cmap=plt.get_cmap("gray")
            )

            plot_img(
                axs[1, 0],
                filtered_rgb,
                f"Step 1: Filtered Image, k=1",
                # cmap=plt.get_cmap("gray"),
                vmin=0.0,
                vmax=255.0,
            )

    plot_img(
        axs[1, 1],
        filtered_rgb,
        f"Step 1: Filtered Image, k={k}",
        # cmap=plt.get_cmap("gray"),
    )

    Image_i = filtered_rgb
    Image_d = image_rgb - Image_i  # signal difference
    plot_img(
        axs[1, 2],
        Image_d,
        "Step2: Signal Difference",
        # cmap=plt.get_cmap("gray"),
        vmin=None,
        vmax=None,
    )

    M, V = compute_local_mean_var_rgb(Image_d)
    plot_img(axs[1, 3], M, "Local Mean", vmin=None, vmax=None)
    plot_img(axs[1, 4], V, "Local Variance", vmin=None, vmax=None)

    threshold = V * 2.5
    Image_m = np.where(np.abs(Image_d - M) < threshold, Image_i, image_rgb)
    plot_img(
        axs[2, 0],
        Image_m,
        "Step 3: Thresholded image",
    )

    s = 0.5
    Image_o = image_rgb - s * Image_m
    plot_img(
        axs[2, 1],
        Image_o,
        "Step 5",
    )

    plot_img(
        axs[2, 3],
        Image_o,
        "Step 6: Auto rescaled",
        vmin=None,
        vmax=None,
    )
    # plot_img(
    #     axs2[1],
    #     Image_o,
    #     "Step 6: Auto rescaled",
    #     cmap=plt.get_cmap("gray"),
    #     vmin=None,
    #     vmax=None,
    # )
    return Image_o


def compute_local_mean_var_rgb(image, kernel_size=40):
    height, width = image.shape[:2]

    local_mean = np.zeros(image.shape)
    local_var = np.zeros(image.shape)

    half_kernel = kernel_size // 2
    with alive_bar(height * width, title="Step3") as bar:
        for i in range(height):
            for j in range(width):
                x_start = max(0, i - half_kernel)
                x_end = min(height, i + half_kernel + 1)
                y_start = max(0, j - half_kernel)
                y_end = min(width, j + half_kernel + 1)

                patch = image[x_start:x_end, y_start:y_end]
                local_mean[i, j] = np.mean(patch, axis=(0, 1))
                local_var[i, j] = np.var(patch, axis=(0, 1), mean=local_mean[i, j])
                bar()
    return local_mean, local_var

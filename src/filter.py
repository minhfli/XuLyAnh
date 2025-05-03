# edge preseving filter
# icip.1994.413625
# IMAGE ENHANCEMENT BY EDGE-PRESERVING FILTERING
# Yiu-fai Wong

from math import exp
from alive_progress import alive_bar
from matplotlib import pyplot as plt
from matplotlib.pylab import f
import numpy as np

from plot_helper import plot_img, axs, axs2


def grayscale(image):
    return np.dot(image[..., :3], [0.289, 0.587, 0.114])


def compute_distance(x_i, y_i, x, y):
    return ((x_i - x) ** 2 + (y_i - y) ** 2) ** 0.5


def compute_distance_2(x_i, y_i, x, y):
    return (x_i - x) ** 2 + (y_i - y) ** 2


def compute_distance2_array(height, width, i, j):
    distance = np.zeros((height, width))
    for k in range(height):
        for l in range(width):
            distance[k, l] = (k - i) ** 2 + (l - j) ** 2
    return distance


def compute_omega_array(
    distance_array,
    alpha=0.5,
):
    omega = np.exp(-alpha * distance_array)
    return omega


def get_starting_point(image, alpha=0.5):
    height = image.shape[0]
    width = image.shape[1]
    filtered_image = np.zeros((height, width))
    with alive_bar(height * width, title="Processing") as bar:
        for i in range(height):
            for j in range(width):
                _distance = compute_distance2_array(height, width, i, j)
                _omega = compute_omega_array(_distance, alpha)
                _y_omega = image * _omega
                y_avg = np.sum(_y_omega) / np.sum(_omega)
                filtered_image[i, j] = y_avg
                bar()

    return filtered_image


def compute_exp_beta(omega_array, y_omega_array, image, current_pixel):
    y_avg = np.sum(y_omega_array) / np.sum(omega_array)

    sigma = np.sum((image - y_avg) ** 2 * omega_array) / np.sum(omega_array)

    if sigma < 0.00001:  # at flat region, image ~ y_avg while omega ~ 0
        _exp = np.zeros(image.shape)
        _exp[current_pixel] = 1
    else:
        beta = 1 / (2 * sigma)
        _exp = np.exp(-beta * (image - image[current_pixel]) ** 2)
    return _exp


def cluster_filter_v1(image, alpha=0.5):

    height = image.shape[0]
    width = image.shape[1]
    filtered_image = np.zeros((height, width))
    with alive_bar(height * width, title="Processing") as bar:
        for i in range(height):
            for j in range(width):
                _distance = compute_distance2_array(height, width, i, j)
                _omega = compute_omega_array(_distance, alpha)
                _y_omega = image * _omega
                _exp_beta = compute_exp_beta(_omega, _y_omega, image, (i, j))

                top = np.sum(_y_omega * _exp_beta)
                bottom = np.sum(_omega * _exp_beta)
                filtered_image[i, j] = top / bottom
                bar()

    return filtered_image


def compute_weight_patch(x, y, min_x, max_x, min_y, max_y, alpha=0.5):
    distance = np.zeros((max_x - min_x, max_y - min_y))
    for i in range(min_x, max_x):
        for j in range(min_y, max_y):
            distance[i - min_x, j - min_y] = (x - i) ** 2 + (y - j) ** 2
    return np.exp(-alpha * distance)


def get_starting_point_v2(image, alpha=0.5, kernel_size=11):
    height, width = image.shape[:2]
    filtered_image = np.zeros((height, width))
    beta = np.zeros((height, width))
    half_kernel = kernel_size // 2

    with alive_bar(height * width, title="Computing y0") as bar:
        for i in range(height):
            for j in range(width):
                min_x = max(0, i - half_kernel)
                max_x = min(height, i + half_kernel + 1)
                min_y = max(0, j - half_kernel)
                max_y = min(width, j + half_kernel + 1)
                patch = image[min_x:max_x, min_y:max_y]
                weight = compute_weight_patch(
                    i, j, min_x, max_x, min_y, max_y, alpha=alpha
                )
                mean = np.sum(patch * weight) / np.sum(weight)
                var2 = np.sum((patch - mean) ** 2 * weight) / np.sum(weight)
                if var2 < 0.00001:
                    beta[i, j] = 10
                else:
                    beta[i, j] = 1 / (2 * var2)
                    if beta[i, j] > 10:
                        beta[i, j] = 10

                filtered_image[i, j] = np.sum(patch * weight) / np.sum(weight)
                bar()

    return filtered_image, beta


def cluster_filter_v2(image, alpha=0.5, beta=None, k=5, kernel_size=11):
    """
    edge_preserve_filter_v1 calculate weight for all pixel of the image,
    since the weight is close to 0 as the distance is far from the current pixel
    v2 compute weight for only the pixels close to the current pixel

    """

    height, width = image.shape[:2]
    filtered_image = np.zeros((height, width))
    half_kernel = kernel_size // 2

    with alive_bar(height * width * k, title="Filtering") as bar:
        for K in range(k):
            for i in range(height):
                for j in range(width):
                    min_x = max(0, i - half_kernel)
                    max_x = min(height, i + half_kernel + 1)
                    min_y = max(0, j - half_kernel)
                    max_y = min(width, j + half_kernel + 1)
                    patch = image[min_x:max_x, min_y:max_y]
                    weight = compute_weight_patch(
                        i, j, min_x, max_x, min_y, max_y, alpha=alpha
                    )

                    tuso = np.sum(
                        patch
                        * weight
                        * np.exp(-beta[i, j] * (patch - image[i, j]) ** 2)
                    )
                    mauso = np.sum(
                        weight * np.exp(-beta[i, j] * (patch - image[i, j]) ** 2)
                    )
                    filtered_image[i, j] = tuso / mauso
                    bar()
            image = filtered_image.copy()

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


def edge_preserve_filter(image, k=5, alpha=0.5, kernel_size=11):

    filtered_image = image
    for i in range(k):
        starting_image, beta = get_starting_point_v2(
            filtered_image, alpha, kernel_size=kernel_size
        )
        # plot_img(axs[0, 2], starting_image, "Starting Image", cmap=plt.get_cmap("gray"))
        filtered_image = cluster_filter_v2(
            starting_image, alpha, beta, kernel_size=kernel_size
        )
        if i == 0:
            plot_img(
                axs[1, 0],
                filtered_image,
                f"Step 1: Filtered Image, k=1",
                cmap=plt.get_cmap("gray"),
            )

    plot_img(
        axs[1, 1],
        filtered_image,
        f"Step 1: Filtered Image, k={k}",
        cmap=plt.get_cmap("gray"),
    )

    Image_i = filtered_image
    Image_d = image - Image_i  # signal difference
    plot_img(
        axs[1, 2],
        Image_d,
        "Step2: Signal Difference",
        cmap=plt.get_cmap("gray"),
        vmin=None,
        vmax=None,
    )

    M, V = compute_local_mean_var(Image_d)
    plot_img(
        axs[1, 3], M, "Local Mean", cmap=plt.get_cmap("gray"), vmin=None, vmax=None
    )
    plot_img(
        axs[1, 4], V, "Local Variance", cmap=plt.get_cmap("gray"), vmin=None, vmax=None
    )

    threshold = V * 2.5
    Image_m = np.where(np.abs(Image_d - M) < threshold, Image_i, image)
    plot_img(axs[2, 0], Image_m, "Step 3: Thresholded image", cmap=plt.get_cmap("gray"))

    s = 0.5
    Image_o = image - s * Image_m
    plot_img(axs[2, 1], Image_o, "Step 5", cmap=plt.get_cmap("gray"))

    m = np.mean(Image_o)
    v = np.var(Image_o, mean=m)
    lower = m - 2.5 * v
    upper = m + 2.5 * v
    Io_rescaled = (Image_o - lower) / (upper - lower)
    plot_img(
        axs[2, 2],
        Io_rescaled,
        "Step 6: Rescale from m-2.5v to m+2.5v",
        cmap=plt.get_cmap("gray"),
        vmin=0,
        vmax=1,
    )

    plot_img(
        axs[2, 3],
        Image_o,
        "Step 6: Auto rescaled",
        cmap=plt.get_cmap("gray"),
        vmin=None,
        vmax=None,
    )
    plot_img(
        axs2[1],
        Image_o,
        "Step 6: Auto rescaled",
        cmap=plt.get_cmap("gray"),
        vmin=None,
        vmax=None,
    )
    return Image_o

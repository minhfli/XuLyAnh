import matplotlib.pyplot as plt
import matplotlib.axes

fig, axs = plt.subplots(3, 5)


def set_subplot(rows, cols):
    global fig, axs
    fig, axs = plt.subplots(rows, cols)


def plot_img(ax: matplotlib.axes.Axes, img, title, cmap=None):
    ax.imshow(img, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")

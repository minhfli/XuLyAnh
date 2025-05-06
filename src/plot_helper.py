import matplotlib.pyplot as plt
import matplotlib.axes

fig, axs = plt.subplots(3, 5)
fig.tight_layout()

fig2, axs2 = plt.subplots(1, 2, figsize=(16, 9))
fig3, axs3 = plt.subplots(1, 2, figsize=(16, 9))


def plot_img(ax: matplotlib.axes.Axes, img, title, cmap=None, vmin=0, vmax=255):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")

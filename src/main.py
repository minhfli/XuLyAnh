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
image_path = "./assets/MRI/Test.jpg"
image_name = image_path.split("/")[-1]
output_path = "./output/" + image_name
alpha = 0.5
k = 1  #! k is 5 in the original paper

rgb_mode = False
if rgb_mode == False:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_gray = image
    plot_img(axs[0, 0], image, "Original Image (grayscale but shown in RGB)")
    plot_img(axs[0, 1], image_gray, "Grayscale Image", cmap=plt.get_cmap("gray"))
    plot_img(axs2[0], image_gray, "Grayscale Image", cmap=plt.get_cmap("gray"))

    filtered_image = edge_preserve_filter(image_gray, k=k, alpha=alpha, kernel_size=11)
else:  # this is not finished
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_gray = grayscale(image)
    plot_img(axs[0, 0], image, "Original Image")
    plot_img(axs[0, 1], image_gray, "Grayscale Image", cmap=plt.get_cmap("gray"))
    edge_preserve_filter_rgb(image, image_gray, k=k, alpha=alpha, kernel_size=11)

plt.tight_layout()
plt.show()
pickle.dump(fig, open(output_path + ".pickle", "wb"))
fig.savefig(output_path + ".png")
pickle.dump(fig2, open(output_path + "2.pickle", "wb"))
fig2.savefig(output_path + "2.png")

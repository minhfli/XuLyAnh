import cv2
import matplotlib
import matplotlib.axes
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib.pyplot as plt

from filter import *
from .plot_helper import *
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
# image_path = "./assets/DIP3E_Problem_Figures/CH04_Problem_Figures/FigP0438(a).tif"
# image_path = "./assets/DIP3E_Problem_Figures/CH05_Problem_Figures/FigP0520.tif"
# image_path = "./assets/salt_and_pepper.png"
# image_path = "./assets/dog.jpg"
# image_path = "./assets/overexposed-portrait.webp"
# image_path = "./assets/overexposed-baby.jpg"
image_name = image_path.split("/")[-1]
output_path = "./output/" + image_name
alpha = 0.5
k = 5  #! k is 5 in the original paper

gausian_test = False

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# image = (image / 255.0 - 0.5) * 2
image_gray = image
plot_img(axs[0, 0], image, "Original Image (grayscale but shown in RGB)")
plot_img(axs[0, 1], image_gray, "Grayscale Image", cmap=plt.get_cmap("gray"))
plot_img(axs2[0], image_gray, "Grayscale Image", cmap=plt.get_cmap("gray"))

cluster_enhanced_image = edge_preserve_filter(
    image_gray, k=k, alpha=alpha, kernel_size=11
)
gausian_enhanced_image = enhance_with_gausian_filter(
    image_gray, k=k, alpha=alpha, kernel_size=11
)

plot_img(
    axs3[0],
    cluster_enhanced_image,
    "Enhanced w cluster filter",
    cmap=plt.get_cmap("gray"),
    vmax=None,
    vmin=None,
)
plot_img(
    axs3[1],
    gausian_enhanced_image,
    "Enhanced w gausian filter",
    cmap=plt.get_cmap("gray"),
    vmax=None,
    vmin=None,
)

plt.tight_layout()
plt.show()
# if gausian_test == False:
#     pickle.dump(fig, open(output_path + ".pickle", "wb"))
#     fig.savefig(output_path + ".png")
#     pickle.dump(fig2, open(output_path + "2.pickle", "wb"))
#     fig2.savefig(output_path + "2.png")
# else:
#     pickle.dump(fig, open(output_path + "_gs.pickle", "wb"))
#     fig.savefig(output_path + "_gs.png")
#     pickle.dump(fig2, open(output_path + "_gs2.pickle", "wb"))
#     fig2.savefig(output_path + "_gs2.png")

fig3.savefig(output_path + "_enhance_cmp.png")

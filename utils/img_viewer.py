import matplotlib.pyplot as plt
from utils.conv_utils import as_convolved_img


def display_img(img):
    plt.imshow(img, cmap='gray')
    plt.show()

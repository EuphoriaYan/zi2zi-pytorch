import matplotlib.pyplot as plt
import numpy as np


def read_split_image(img):
    box1 = (0, 0, img.size[1], img.size[1])  # (left, upper, right, lower) - tuple
    box2 = (img.size[1], 0, img.size[0], img.size[1])
    img_A = img.crop(box1)  # target
    img_B = img.crop(box2)  # source
    return img_A, img_B


def plot_tensor(tensor):
    img = np.transpose(tensor.data, (1, 2, 0))
    plt.imshow(img)

from PIL import Image
import os
from torchvision import transforms
import torch
from group_conv import Rot90Group, bilinear_interpolation
import matplotlib.pyplot as plt
import random

IMGS_PATH = "imgs"


class RandomRotationFromSet(object):
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(img, angle)

def test_group():
    img = Image.open(os.path.join(IMGS_PATH, "dog.jpeg"))
    img_tensor = transforms.ToTensor()(img)

    img_grid = torch.stack(torch.meshgrid(
        torch.linspace(-1, 1, img_tensor.shape[-2]),
        torch.linspace(-1, 1, img_tensor.shape[-1]),
        indexing='ij'
    ))

    c4 = Rot90Group(order=4)
    e, g1, g2, g3 = c4.elements()
    g4 = c4.product(g2, c4.product(g1, g2)) # 180 + 270 = 90 degrees

    transformed_img_grid = c4.left_action_on_R2(c4.inverse(g4), img_grid)

    # Sample the image on the transformed grid points.
    transformed_img = bilinear_interpolation(img_tensor, transformed_img_grid)[0]

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img_tensor.transpose(0, 1).transpose(1, 2))
    ax[0].set_title("Original Image")
    ax[0].axis("off")
    ax[1].imshow(transformed_img.transpose(0, 1).transpose(1, 2))
    ax[1].set_title("Transformed Image")
    ax[1].axis("off")
    plt.show()
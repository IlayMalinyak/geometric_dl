from PIL import Image
import os
from torchvision import transforms
import torch
from group_conv import Rot90Group, bilinear_interpolation
import matplotlib.pyplot as plt
import random
import itertools
import numpy as np

IMGS_PATH = "imgs"

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class RandomRotationFromSet(object):
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(img, angle)

def plot_fit(
    fit_res: dict,
    fig=None,
    log_loss=False,
    legend=None,
    train_test_overlay: bool = False,
):
    """
    Plots a FitResult object.
    Creates four plots: train loss, test loss, train acc, test acc.
    :param fit_res: The fit result to plot.
    :param fig: A figure previously returned from this function. If not None,
        plots will the added to this figure.
    :param log_loss: Whether to plot the losses in log scale.
    :param legend: What to call this FitResult in the legend.
    :param train_test_overlay: Whether to overlay train/test plots on the same axis.
    :return: The figure.
    """
    if fig is None:
        nrows = 1 if train_test_overlay else 2
        ncols = 2
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(8 * ncols, 5 * nrows),
            sharex="col",
            sharey=False,
            squeeze=False,
        )
        axes = axes.reshape(-1)
    else:
        axes = fig.axes

    for ax in axes:
        for line in ax.lines:
            if line.get_label() == legend:
                line.remove()

    p = itertools.product(enumerate(["train", "val"]), enumerate(["loss", "acc"]))
    for (i, traintest), (j, lossacc) in p:

        ax = axes[j if train_test_overlay else i * 2 + j]

        attr = f"{traintest}_{lossacc}"
        data =fit_res[attr]
        label = traintest if train_test_overlay else legend
        h = ax.plot(np.arange(1, len(data) + 1), data, label=label)
        ax.set_title(attr)

        if lossacc == "loss":
            ax.set_xlabel("Iteration #")
            ax.set_ylabel("Loss")
            if log_loss:
                ax.set_yscale("log")
                ax.set_ylabel("Loss (log)")
        else:
            ax.set_xlabel("Epoch #")
            ax.set_ylabel("Accuracy (%)")

        if legend or train_test_overlay:
            ax.legend()
        ax.grid(True)

    return fig, axes

def show_feature_maps(model_out, rots, out_channel_idx, save_path, show_group=False):
    if not show_group:
        fig, ax = plt.subplots(1, rots.numel(), figsize=(10,3))
        for idx, rotation in enumerate(rots):
            ax[idx].imshow(
                model_out[idx, out_channel_idx, :, :].cpu().detach().numpy()
            )
            ax[idx].set_title(f"{int(rotation)} deg")     
        fig.text(0.5, 0.04, 'Rotations of input image', ha='center') 
    else:
        fig, ax = plt.subplots(model_out.shape[2], rots.numel(), figsize=(10,9))     
        for idx, rotation in enumerate(rots):
            for group_element_idx in range(model_out.shape[2]):
                ax[group_element_idx ,idx].imshow(
                    model_out[idx, out_channel_idx, group_element_idx, :, :].cpu().detach().numpy()
                )
            ax[0, idx].set_title(f"{int(rotation)} deg")
            
        
        fig.text(0.5, 0.04, 'Rotations of input image', ha='center')
        fig.text(0.04, 0.5, '$H$ dimension in feature map', va='center', rotation='vertical')
        
    plt.savefig(f'{save_path}/feature_maps.png')
    plt.close()


def test_rotation_group():
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
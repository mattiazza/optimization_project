import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL as Image


def plot_subplots(
    samples: list[dict],
    grid_size: int = 3,
    using_resnet: bool = False,
) -> None:
    """
    Plot a grid of images from samples.

    Args:
        • samples (list): List of dictionary containing images and the respective labels.
                        Each dictionary should have keys "image", "label", "label_name", "predicted_label" and "predicted_name".
        • grid_size (int): Size of the grid to plot the images.
    """
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))

    if grid_size == 1:
        raise ValueError("grid_size must be greater than 1 to plot multiple images.")

    axes = axes.flatten()

    pred_name = None
    for ax, item in zip(axes, samples):
        image = item["image"]

        true_label = item["label"]
        true_name = item["label_name"]
        if using_resnet:
            # Truncate the label names for plotting if they are too long
            true_name = true_name if len(true_name) <= 20 else true_name[:20] + "..."

            pred_label = item["pred_resnet"]
            pred_name = item["pred_resnet_name"]
            # Truncate the predicted names for plotting if they are too long
            pred_name = pred_name if len(pred_name) <= 20 else pred_name[:20] + "..."

        else:
            pred_label = item["pred_inception"]

        ax.imshow(image)
        ax.set_title(
            f"Pred: {pred_label} ({pred_name if pred_name is not None else ''})\nTrue: {true_label} ({true_name})"
        )
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def paired_plot(
    image: Image.Image,
    label: int,
    target: int,
    pred: int,
    image_fw: Image.Image,
    pred_fw: int,
    image_fw_as: Image.Image,
    pred_fw_as: int,
    image_fw_pair: Image.Image,
    pred_fw_pair: int,
    title: str,
    save_name: Optional[str] = None,
    savefig: bool = False,
):
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))

    axes = axes.flatten()

    fig.suptitle(title, fontsize=16)

    axes[0].imshow(image)
    axes[0].set_title("Original Image")  # \nPredicted: {prediction}\nTrue: {label}")
    axes[0].text(
        0.5,
        -0.01,
        f"Pred: {pred}\nTrue: {label}",
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        fontsize=14,
    )
    axes[0].axis("off")

    axes[1].imshow(image_fw)
    axes[1].set_title(
        "Frank-Wolfe Attack"
    )  # \nPredicted: {prediction_adv}\nTrue: {label}")
    axes[1].text(
        0.5,
        -0.01,
        f"Pred: {pred_fw}\nTarget: {target}",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=14,
    )
    axes[1].axis("off")

    axes[2].imshow(image_fw_as)
    axes[2].set_title(
        "FW Away-Steps Attack"
    )  # \nPredicted: {prediction_adv}\nTrue: {label}")
    axes[2].text(
        0.5,
        -0.01,
        f"Pred: {pred_fw_as}\nTarget: {target}",
        transform=axes[2].transAxes,
        ha="center",
        va="top",
        fontsize=14,
    )
    axes[2].axis("off")

    axes[3].imshow(image_fw_pair)
    axes[3].set_title(
        "FW Pairwise Attack"
    )  # \nPredicted: {prediction_adv}\nTrue: {label}")
    axes[3].text(
        0.5,
        -0.01,
        f"Pred: {pred_fw_pair}\nTarget: {target}",
        transform=axes[3].transAxes,
        ha="center",
        va="top",
        fontsize=14,
    )
    axes[3].axis("off")

    plt.tight_layout()

    if savefig:
        os.makedirs("plots", exist_ok=True)
        if save_name is None:
            raise ValueError("save_name must be provided if savefig is True.")
        plt.savefig(f"plots/{save_name}.png", bbox_inches="tight")

    plt.show()


def plot_different_epsilon_demo(
    df: pd.DataFrame,
    image_original_demo: pd.DataFrame,
    algorithms: list,
    dataset_name: str,
    save_name: Optional[str] = None,
    figsize: tuple = (10, 6),
    savefig: bool = False,
):
    # Create a subplot with one row and five columns
    plt.figure(figsize=figsize)

    i = 0
    for algo in algorithms:
        epsilons = df[f"epsilon_{algo}"]
        if i % (len(epsilons) + 1) == 0:
            # First subplot: Original image
            i += 1
            plt.subplot(
                len(algorithms), len(epsilons) + 1, i
            )  # 3 , len(epsilon_plot)+1, columns
            plt.imshow(image_original_demo[image_original_demo["algo"] == algo]["image"].values[0])

            # if i <= len(epsilons):  # Show title only for the first row
            plt.title("Original Image")
            plt.axis("off")

        for eps in epsilons:
            i += 1
            plt.subplot(len(algorithms), len(epsilons) + 1, i)
            plt.imshow(df[df[f"epsilon_{algo}"] == eps][f"image_{algo}"].values[0])

            # if i <= (len(epsilons) + 1):  # Show title only for the first row
            plt.title(f"ε = {eps}")
            plt.axis("off")

    plt.tight_layout()
    plt.suptitle(
        f"{dataset_name.upper()} Under Frank-Wolfe Attack with Different Epsilon Values",
        y=1.05,
    )
    if savefig:
        os.makedirs("plots", exist_ok=True)  # Create the directory if it doesn't exist
        plt.savefig(f"plots/{save_name}.png", bbox_inches="tight")  # Save the figure

    plt.show()

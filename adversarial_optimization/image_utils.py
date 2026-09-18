import io
from itertools import islice

import numpy as np
import torch
import torchvision.transforms as T
from datasets import load_dataset
from PIL import Image


def get_dataset_samples(
    dataset_name: str = "imagenet-1k",
    split: str = "validation",
    n_images: int = 10,
    streaming: bool = True,
) -> list:
    """
    Get n images from the dataset.
    Assuming the dataset is large, we set streaming to `True` to avoid loading everything into memory.
    """

    print(f"Loading {n_images} images from the {dataset_name} dataset")
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)

    return list(islice(dataset, n_images))


def from_bytes_to_image(sample):
    for n in range(len(sample)):
        sample[n]["image"] = Image.open(io.BytesIO(sample[n].pop("img_bytes"))).convert(
            "RGB"
        )

    return sample


def image_to_tensor(image, requires_grad: bool = False):
    to_tensor = T.ToTensor()

    image_tensor = to_tensor(image).unsqueeze(0)  # Add batch dimension

    return image_tensor.requires_grad_(requires_grad)


def tensor_to_image(image_tensor):
    """
    Convert a tensor to a PIL image.
    This is a placeholder function. Replace with actual conversion logic if needed.
    """
    to_pil = T.ToPILImage()

    image_tensor = image_tensor.squeeze(0).clamp(0, 1)

    return to_pil(image_tensor)  # Remove batch dimension


def deprocess_siglip2(proc_tensor: torch.Tensor) -> Image.Image:
    """Inverse for Siglip2 preprocess. Input: [1,3,H,W] or [3,H,W] normalized"""
    x = proc_tensor.detach().cpu()
    if x.dim() == 4:
        x = x[0]
    # undo normalization
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    x = x * std + mean  # now in [0,1] range if rescale undone below

    rescale = 0.00392156862745098
    x = x / rescale
    x = torch.clamp(x, 0.0, 255.0)
    arr = (x.permute(1, 2, 0).numpy()).astype(np.uint8)
    return Image.fromarray(arr)


def deprocess_resnet(proc_tensor: torch.Tensor) -> Image.Image:
    """Inverse for resnet preprocess (imagenet mean/std)."""
    x = proc_tensor.detach().cpu()
    if x.dim() == 4:
        x = x[0]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    x = x * std + mean  # now in [0,1]
    x = torch.clamp(x, 0.0, 1.0)
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def deprocess_vit(proc_tensor: torch.Tensor) -> Image.Image:
    """Inverse for ViT preprocess (mean/std=0.5)."""
    x = proc_tensor.detach().cpu()
    if x.dim() == 4:
        x = x[0]
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    x = x * std + mean
    x = torch.clamp(x, 0.0, 1.0)
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def deprocess_tensor_to_pil(tensor, model_name: str):
    if model_name == "siglip2":
        pil_image = deprocess_siglip2(tensor)
    elif model_name == "resnet50":
        pil_image = deprocess_resnet(tensor)
    elif model_name == "vit":
        pil_image = deprocess_vit(tensor)
    else:
        raise ValueError(
            f"Unknown preprocessor name: {model_name}",
            "\nSupported: siglip2, resnet50, vit",
        )

    return pil_image

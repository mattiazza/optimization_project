import torch
import torch.nn.functional as F
from PIL import Image


def compute_gradient(model, x, label_tensor):
    if x.grad is not None:
        x.grad.zero_()

    output = model(x)

    logits = output.logits if hasattr(output, "logits") else output
    loss = F.cross_entropy(logits, label_tensor)
    loss.backward()
    grad = x.grad.data
    pred = logits.argmax(dim=-1).item()
    return grad, pred


def fw_vanilla(
    model,
    img_ori,  #: Image.Image,
    target: torch.Tensor,
    epsilon: float = 1e-2,
    max_iter: int = 1000,
) -> tuple[Image.Image, int]:
    """
    Vanilla Frank-Wolfe algorithm for adversarial attacks.
    Args:
        model: The model to attack.
        img_ori: The original image tensor.
        target: The target class tensor.
        epsilon: The maximum perturbation allowed.
        max_iter: The maximum number of iterations.
    Returns:
        attacked_img: The adversarially perturbed image tensor.
    """

    # original_img = image_to_tensor(img_ori)
    # attacked_img = image_to_tensor(img_ori, requires_grad=True)
    original_img = img_ori
    attacked_img = img_ori.clone().detach().requires_grad_(True)

    it = 0

    for k in range(1, max_iter + 1):
        # print("Iteration:", k)
        # Compute the step size
        gamma = 2 / (2 + k)

        # Compute the gradient of the objective function
        grad, pred = compute_gradient(
            model, attacked_img, target
        )  # Placeholder for actual gradient computation

        print(f"Iteration {it} - Target: {target.item()} - Current prediction: {pred}")

        it = k
        if pred == target.item():
            print(
                f"\nEarly stopping at iteration {k} as the target class is reached.\n"
            )
            break

        # Compute the vertex of the constraint set
        v = -torch.sign(grad) * epsilon + original_img

        # Update the solution
        attacked_img = (1 - gamma) * attacked_img + gamma * v

        # Make img_attacked differentiable again
        attacked_img = attacked_img.detach().requires_grad_(True)

    # attacked_img = tensor_to_image(attacked_img)

    return attacked_img, it  # Returns a

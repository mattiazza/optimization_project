import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def compute_gradient_gamma(model, x, label_tensor, gamma, d):
    gamma = torch.tensor(gamma, dtype=torch.float32, requires_grad=True)

    # Initialize gradient at zero
    if gamma.grad is not None:
        gamma.grad.zero_()

    output = model(x + gamma * d)

    logits = output.logits if hasattr(output, "logits") else output
    # Cross-Entropy
    loss = F.cross_entropy(logits, label_tensor)
    # Backpropagation of the gradient of the loss
    grad_gamma = torch.autograd.grad(loss, gamma)[0]
    return grad_gamma


def line_search_bisection(model, x, y_t, d, gamma_max, threshold):
    gamma_low = 0.0
    gamma_high = gamma_max

    while gamma_high - gamma_low > threshold:
        gamma_mid = (gamma_low + gamma_high) / 2.0
        grad = compute_gradient_gamma(model, x, y_t, gamma_mid, d)
        print(f"grad: {grad}, gamma_mid: {gamma_mid}")

        # print(f"  Line search - grad: {grad.item():.4f}, gamma: {gamma_mid:.4f}")

        if grad > 0:
            gamma_high = gamma_mid
        else:
            gamma_low = gamma_mid

    return round((gamma_low + gamma_high) / 2.0, 3)


def compute_gradient(model, x, label_tensor):
    if x.grad is not None:
        x.grad.zero_()

    output = model(x)

    logits = output.logits if hasattr(output, "logits") else output
    loss = torch.nn.functional.cross_entropy(logits, label_tensor)
    loss.backward()
    grad = x.grad.data
    pred = logits.argmax(dim=-1).item()
    return grad, pred


def fw_away_steps(
    model,
    img_ori,  #: Image.Image,
    target: torch.Tensor,
    epsilon: float = 1e-2,
    max_iter: int = 1000,
) -> tuple[Image.Image, int]:
    """
    Frank-Wolfe algorithm with Away Steps for adversarial attacks.

    This variant maintains an active set of vertices and adaptively chooses between
    Frank-Wolfe steps (towards new extreme points) and Away steps (away from vertices
    in the active set) based on which provides better progress. Away steps help escape
    from suboptimal vertices and can lead to faster convergence.

    Args:
        model: The model to attack.
        img_ori: The original image tensor.
        target: The target class tensor.
        epsilon: The maximum perturbation allowed (L-infinity bound).
        max_iter: The maximum number of iterations.
    Returns:
        attacked_img: The adversarially perturbed image tensor.
        it: The number of iterations performed.
    """

    # original_img = image_to_tensor(img_ori)
    # attacked_img = image_to_tensor(img_ori, requires_grad=True)
    original_img = img_ori
    attacked_img = img_ori.clone().detach().requires_grad_(True)

    active_set = [[original_img, 1]]
    it = 0

    for k in range(max_iter):
        print("Iteration:", k)
        grad, pred = compute_gradient(model, attacked_img, target)

        print(f"Active set size: {len(active_set)}")
        it = k
        # print(f"Iteration {it} - Target: {target.item()} - Current prediction: {pred}")
        if pred == target.item():
            print(
                f"\nEarly stopping at iteration {k} as the target class is reached.\n"
            )
            break

        s = -torch.sign(grad) * epsilon + original_img  # L-inf FW step
        d_fw = s - attacked_img

        v_best = np.argmax(
            [torch.dot(grad.flatten(), vertex.flatten()) for vertex, _ in active_set]
        )
        d_sa = attacked_img - active_set[v_best][0]

        g_fw = torch.dot(-grad.flatten(), d_fw.flatten())
        g_as = torch.dot(grad.flatten(), d_sa.flatten())
        # print(f"  g_fw: {g_fw:.4f}, g_as: {g_as:.4f}")
        # Stopping condition
        if g_fw <= 1e-6:
            break

        if g_fw >= g_as:
            d = d_fw
            gamma_max = 1

            step = "fw"
            print("FW step")
        else:
            alpha = active_set[v_best][1]

            d = d_sa
            gamma_max = alpha / (1 - alpha)
            step = "as"
            print("Away step")

        # line search
        gamma = line_search_bisection(
            model,
            attacked_img,
            target,
            d,
            gamma_max,
            threshold=0.1,
        )

        attacked_img = attacked_img + gamma * d

        if step == "fw":
            if gamma == 1:
                active_set = [[s, 1]]
            else:
                active_set.append([s, 0])
                for i in range(len(active_set)):
                    active_set[i][1] = (1 - gamma) * active_set[i][1]
                    if active_set[i][0] is s:
                        active_set[i][1] += gamma

        else:  # Away Step
            if gamma == gamma_max:
                active_set.pop(v_best)

            else:
                for i in range(len(active_set)):
                    active_set[i][1] = (1 + gamma) * active_set[i][1]
                    if i == v_best:
                        active_set[i][1] -= gamma

        # Ensure x is a leaf with grad
        attacked_img = attacked_img.detach().clone().requires_grad_()

    # attacked_img = tensor_to_image(attacked_img)
    return attacked_img, it

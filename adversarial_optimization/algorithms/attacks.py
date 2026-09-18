import time

from adversarial_optimization.algorithms.fw_away_step import fw_away_steps
from adversarial_optimization.algorithms.fw_baseline import fw_vanilla
from adversarial_optimization.algorithms.fw_pairwise import fw_pairwise


def attack_image(
    model, img_ori, target, epsilon=0.1, max_iter=25, algorithm: str = "fw"
):
    """
    Perform an attack on the given image using the specified algorithm.
    """
    if algorithm == "fw":
        start_time = time.time()
        attacked_image, n_iteration = fw_vanilla(
            model=model,
            img_ori=img_ori,
            target=target,
            epsilon=epsilon,
            max_iter=max_iter,
        )
        end_time = time.time()
        attack_time = end_time - start_time
        return attacked_image, n_iteration, attack_time

    elif algorithm == "fw_as":
        start_time = time.time()
        attacked_image, n_iteration = fw_away_steps(
            model=model,
            img_ori=img_ori,
            target=target,
            epsilon=epsilon,
            max_iter=max_iter,
        )
        end_time = time.time()
        attack_time = end_time - start_time
        return attacked_image, n_iteration, attack_time

    elif algorithm == "fw_pair":
        start_time = time.time()
        attacked_image, n_iteration = fw_pairwise(
            model=model,
            img_ori=img_ori,
            target=target,
            epsilon=epsilon,
            max_iter=max_iter,
        )
        end_time = time.time()
        attack_time = end_time - start_time
        return attacked_image, n_iteration, attack_time
    else:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. Supported algorithms are: 'fw', 'fw_as', 'fw_pair'"
        )

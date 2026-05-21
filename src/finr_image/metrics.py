import math

import numpy as np


def psnr(reference: np.ndarray, prediction: np.ndarray, data_range: float = 1.0) -> float:
    mse = float(np.mean(np.square(reference - prediction)))
    if mse == 0.0:
        return float("inf")
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def ssim_rgb(reference: np.ndarray, prediction: np.ndarray) -> float | None:
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        return None

    min_side = min(reference.shape[0], reference.shape[1])
    if min_side < 3:
        return None
    win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
    if win_size < 3:
        return None

    scores = []
    for channel in range(reference.shape[-1]):
        scores.append(
            structural_similarity(
                reference[..., channel],
                prediction[..., channel],
                data_range=1.0,
                win_size=win_size,
            )
        )
    return float(np.mean(scores))

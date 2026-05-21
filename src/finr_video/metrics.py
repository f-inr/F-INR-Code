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

    min_side = min(reference.shape[1], reference.shape[2])
    if min_side < 3:
        return None
    win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
    if win_size < 3:
        return None

    scores = []
    for channel in range(reference.shape[0]):
        scores.append(
            structural_similarity(
                reference[channel, ...],
                prediction[channel, ...],
                data_range=1.0,
                win_size=win_size,
            )
        )
    return float(np.mean(scores))


def frame_metrics(
    reference_video: np.ndarray,
    prediction_video: np.ndarray,
    frame_index: int,
) -> tuple[float, float | None]:
    frame_index = min(frame_index, reference_video.shape[1] - 1)
    reference_frame = reference_video[:, frame_index, :, :]
    prediction_frame = prediction_video[:, frame_index, :, :]
    return psnr(reference_frame, prediction_frame), ssim_rgb(reference_frame, prediction_frame)


def video_psnr(reference_video: np.ndarray, prediction_video: np.ndarray) -> float:
    return psnr(reference_video, prediction_video)

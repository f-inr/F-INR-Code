import numpy as np


def relative_l2(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    denominator = np.linalg.norm(target.reshape(-1))
    if denominator == 0.0:
        return float(np.linalg.norm(prediction.reshape(-1)))
    return float(np.linalg.norm((prediction - target).reshape(-1)) / denominator)

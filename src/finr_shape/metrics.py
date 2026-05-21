import jax.numpy as jnp
import numpy as np
from scipy.spatial import KDTree


def relative_l2(prediction: np.ndarray | jnp.ndarray, target: np.ndarray | jnp.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    numerator = np.linalg.norm(prediction - target)
    denominator = np.linalg.norm(target)
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return float(numerator / denominator)


def calculate_iou(sdf_pred: np.ndarray, sdf_gt: np.ndarray) -> float:
    binary_pred = np.asarray(sdf_pred) <= 0
    binary_gt = np.asarray(sdf_gt) <= 0
    intersection = np.logical_and(binary_pred, binary_gt).sum()
    union = np.logical_or(binary_pred, binary_gt).sum()
    return float(intersection / union) if union > 0 else 0.0


def calculate_metrics_from_voxels(
    sdf_pred: np.ndarray,
    sdf_gt: np.ndarray,
    surface_threshold: float = 0.5,
) -> tuple[float, float, float]:
    grid = np.indices(sdf_pred.shape).reshape(3, -1).T
    pred_surface_voxels = grid[np.abs(np.asarray(sdf_pred).ravel()) <= surface_threshold]
    gt_surface_voxels = grid[np.abs(np.asarray(sdf_gt).ravel()) <= surface_threshold]

    if len(pred_surface_voxels) == 0 or len(gt_surface_voxels) == 0:
        return float("inf"), float("inf"), calculate_iou(sdf_pred, sdf_gt)

    tree_pred = KDTree(pred_surface_voxels)
    tree_gt = KDTree(gt_surface_voxels)

    dist_pred_to_gt, _ = tree_pred.query(gt_surface_voxels)
    dist_gt_to_pred, _ = tree_gt.query(pred_surface_voxels)

    chamfer_dist = float(np.mean(dist_pred_to_gt) + np.mean(dist_gt_to_pred))
    hausdorff_dist = float(max(np.max(dist_pred_to_gt), np.max(dist_gt_to_pred)))
    iou = calculate_iou(sdf_pred, sdf_gt)
    return chamfer_dist, hausdorff_dist, iou

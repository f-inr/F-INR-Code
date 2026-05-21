import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from finr_shape import (  # noqa: E402
    AxisConfig,
    BaselineShapeINR,
    ShapeEncodingConfig,
    build_shape_model,
    downsample_gradient_grid,
    downsample_sdf_grid,
    load_gradient_grid,
    load_sdf_grid,
    train_baseline_shape_model,
    train_shape_model,
    truncate_sdf,
)

from finr_common import count_params, load_json_config, maybe_override, resolve_output_dir, resolve_project_path, write_json  # noqa: E402


DEFAULT_CONFIG = {
    "sdf_path": None,
    "gradient_path": None,
    "model": "finr",
    "decomposition": "cp",
    "backend": "relu",
    "encoding": "raw",
    "pos_enc": 0,
    "axis_backends": ["relu", "relu", "relu"],
    "axis_encodings": ["raw", "raw", "raw"],
    "axis_pos_enc": [0, 0, 0],
    "hidden_dims": [256, 256, 256],
    "rank": 64,
    "steps": 5_000,
    "eval_every": 500,
    "learning_rate": 1e-4,
    "seed": 0,
    "truncate": 0.1,
    "downsample_stride": 1,
    "near_surface_threshold": 0.01,
    "near_surface_weight": 1000.0,
    "use_eikonal": False,
    "eikonal_weight": 1.0,
    "use_grad_supervision": False,
    "grad_weight": 1.0,
    "num_rbfs": 64,
    "hash_levels": 16,
    "hash_features_per_level": 2,
    "hash_min_resolution": 16,
    "hash_max_resolution": 524_288,
    "hash_table_size": 2**14,
    "output_dir": str(resolve_output_dir(None, "sdf_public")),
    "experiment_name": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Public SDF F-INR runner for reproducible experiments.")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config file.")
    parser.add_argument("--sdf-path", type=Path, default=None, help="Path to the SDF voxel grid.")
    parser.add_argument("--gradient-path", type=Path, default=None, help="Optional gradient supervision grid.")
    parser.add_argument("--model", choices=["finr", "baseline"], default=None, help="Model family to train.")
    parser.add_argument("--mode", "--decomposition", dest="decomposition", choices=["cp", "tt", "tucker", "tr"], default=None, help="Tensor decomposition for the factorized SDF model.")
    parser.add_argument("--backend", choices=["relu", "siren", "siren2", "wire", "finer"], default=None, help="Baseline backend.")
    parser.add_argument("--encoding", choices=["raw", "positional", "hash"], default=None, help="Baseline encoding.")
    parser.add_argument("--pos-enc", type=int, default=None, help="Baseline positional encoding size.")
    parser.add_argument("--axis-backends", nargs=3, default=None, help="Per-axis backends for x, y, z.")
    parser.add_argument("--axis-encodings", nargs=3, default=None, help="Per-axis encodings for x, y, z.")
    parser.add_argument("--axis-pos-enc", nargs=3, type=int, default=None, help="Per-axis positional encoding sizes.")
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=None, help="Hidden dimensions.")
    parser.add_argument("--rank", type=int, default=None, help="Rank for the factorized SDF model.")
    parser.add_argument("--steps", type=int, default=None, help="Optimization steps.")
    parser.add_argument("--eval-every", type=int, default=None, help="Metric logging interval.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--truncate", type=float, default=None, help="Optional SDF truncation.")
    parser.add_argument("--downsample-stride", type=int, default=None, help="Stride-based grid downsampling.")
    parser.add_argument("--near-surface-threshold", type=float, default=None, help="Near-surface threshold.")
    parser.add_argument("--near-surface-weight", type=float, default=None, help="Near-surface loss weight.")
    parser.add_argument("--use-eikonal", action="store_true", help="Enable eikonal regularization.")
    parser.add_argument("--eikonal-weight", type=float, default=None, help="Eikonal regularization weight.")
    parser.add_argument("--use-grad-supervision", action="store_true", help="Enable gradient supervision.")
    parser.add_argument("--grad-weight", type=float, default=None, help="Gradient supervision weight.")
    parser.add_argument("--num-rbfs", type=int, default=None, help="RBF count for NeuRBF backends.")
    parser.add_argument("--hash-levels", type=int, default=None, help="Hash levels.")
    parser.add_argument("--hash-features-per-level", type=int, default=None, help="Features per hash level.")
    parser.add_argument("--hash-min-resolution", type=int, default=None, help="Minimum hash resolution.")
    parser.add_argument("--hash-max-resolution", type=int, default=None, help="Maximum hash resolution.")
    parser.add_argument("--hash-table-size", type=int, default=None, help="Hash table size.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for outputs.")
    parser.add_argument("--experiment-name", type=str, default=None, help="Optional run name override.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    config = dict(DEFAULT_CONFIG)
    config.update(load_json_config(args.config))

    config["sdf_path"] = maybe_override(config.get("sdf_path"), None if args.sdf_path is None else str(args.sdf_path))
    config["gradient_path"] = maybe_override(config.get("gradient_path"), None if args.gradient_path is None else str(args.gradient_path))
    config["model"] = maybe_override(config.get("model"), args.model)
    config["decomposition"] = maybe_override(config.get("decomposition"), args.decomposition)
    config["backend"] = maybe_override(config.get("backend"), args.backend)
    config["encoding"] = maybe_override(config.get("encoding"), args.encoding)
    config["pos_enc"] = maybe_override(config.get("pos_enc"), args.pos_enc)
    config["axis_backends"] = maybe_override(config.get("axis_backends"), args.axis_backends)
    config["axis_encodings"] = maybe_override(config.get("axis_encodings"), args.axis_encodings)
    config["axis_pos_enc"] = maybe_override(config.get("axis_pos_enc"), args.axis_pos_enc)
    config["hidden_dims"] = maybe_override(config.get("hidden_dims"), args.hidden_dims)
    config["rank"] = maybe_override(config.get("rank"), args.rank)
    config["steps"] = maybe_override(config.get("steps"), args.steps)
    config["eval_every"] = maybe_override(config.get("eval_every"), args.eval_every)
    config["learning_rate"] = maybe_override(config.get("learning_rate"), args.learning_rate)
    config["seed"] = maybe_override(config.get("seed"), args.seed)
    config["truncate"] = maybe_override(config.get("truncate"), args.truncate)
    config["downsample_stride"] = maybe_override(config.get("downsample_stride"), args.downsample_stride)
    config["near_surface_threshold"] = maybe_override(config.get("near_surface_threshold"), args.near_surface_threshold)
    config["near_surface_weight"] = maybe_override(config.get("near_surface_weight"), args.near_surface_weight)
    config["use_eikonal"] = True if args.use_eikonal else config.get("use_eikonal", False)
    config["eikonal_weight"] = maybe_override(config.get("eikonal_weight"), args.eikonal_weight)
    config["use_grad_supervision"] = True if args.use_grad_supervision else config.get("use_grad_supervision", False)
    config["grad_weight"] = maybe_override(config.get("grad_weight"), args.grad_weight)
    config["num_rbfs"] = maybe_override(config.get("num_rbfs"), args.num_rbfs)
    config["hash_levels"] = maybe_override(config.get("hash_levels"), args.hash_levels)
    config["hash_features_per_level"] = maybe_override(config.get("hash_features_per_level"), args.hash_features_per_level)
    config["hash_min_resolution"] = maybe_override(config.get("hash_min_resolution"), args.hash_min_resolution)
    config["hash_max_resolution"] = maybe_override(config.get("hash_max_resolution"), args.hash_max_resolution)
    config["hash_table_size"] = maybe_override(config.get("hash_table_size"), args.hash_table_size)
    config["output_dir"] = str(resolve_output_dir(maybe_override(config.get("output_dir"), args.output_dir), "sdf_public"))
    config["experiment_name"] = maybe_override(config.get("experiment_name"), args.experiment_name)

    if config["sdf_path"] is None:
        raise ValueError("`sdf_path` must be provided via --sdf-path or --config.")
    return config


def normalize_config(config: dict) -> dict:
    normalized = dict(DEFAULT_CONFIG)
    normalized.update(config)
    if normalized["sdf_path"] is None:
        raise ValueError("`sdf_path` must be provided.")
    normalized["output_dir"] = str(resolve_output_dir(normalized.get("output_dir"), "sdf_public"))
    return normalized


def build_axis_configs(config: dict) -> tuple[AxisConfig, AxisConfig, AxisConfig]:
    names = ("x", "y", "z")
    return tuple(
        AxisConfig(
            name=name,
            hidden_dims=tuple(int(width) for width in config["hidden_dims"]),
            backend=config["axis_backends"][index],
            encoding=config["axis_encodings"][index],
            pos_enc=int(config["axis_pos_enc"][index]),
            hash_levels=int(config["hash_levels"]),
            hash_features_per_level=int(config["hash_features_per_level"]),
            hash_min_resolution=int(config["hash_min_resolution"]),
            hash_max_resolution=int(config["hash_max_resolution"]),
            hash_table_size=int(config["hash_table_size"]),
            num_rbfs=int(config["num_rbfs"]),
        )
        for index, name in enumerate(names)
    )


def save_preview_slice(path: Path, grid: np.ndarray) -> None:
    mid = grid.shape[2] // 2
    image = grid[:, :, mid]
    image = image - image.min()
    denom = image.max()
    if denom > 0:
        image = image / denom
    Image.fromarray(np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)).save(path)


def save_outputs(output_dir: Path, grid: np.ndarray, config: dict, result, params_count: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "input.npy", grid)
    np.save(output_dir / "reconstruction.npy", result.reconstruction)
    save_preview_slice(output_dir / "input_slice.png", grid)
    save_preview_slice(output_dir / "reconstruction_slice.png", result.reconstruction)

    with (output_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "loss", "relative_l2", "iou"])
        for record in result.history:
            writer.writerow([record.step, record.loss, record.relative_l2, record.iou])

    summary = {
        "model": config["model"],
        "decomposition": config["decomposition"],
        "backend": config["backend"],
        "encoding": config["encoding"],
        "rank": config["rank"],
        "parameter_count": params_count,
        "elapsed_seconds": result.elapsed_seconds,
        "final_step": result.history[-1].step,
        "final_loss": result.history[-1].loss,
        "final_relative_l2": result.history[-1].relative_l2,
        "final_iou": result.history[-1].iou,
    }
    write_json(output_dir / "metrics.json", summary)
    write_json(output_dir / "effective_config.json", config)
    return summary


def run_sdf_experiment(config: dict) -> tuple[dict, object]:
    config = normalize_config(config)
    sdf_path = resolve_project_path(config["sdf_path"])
    gradient_path = resolve_project_path(config["gradient_path"])
    output_dir_root = resolve_output_dir(config["output_dir"], "sdf_public")
    grid = load_sdf_grid(sdf_path)
    gradients = None
    if gradient_path is not None:
        gradients = load_gradient_grid(gradient_path)

    stride = int(config["downsample_stride"])
    if stride > 1:
        grid = downsample_sdf_grid(grid, stride=stride)
        if gradients is not None:
            gradients = downsample_gradient_grid(gradients, stride=stride)

    truncate_value = float(config["truncate"])
    if truncate_value > 0:
        grid = truncate_sdf(grid, truncation=truncate_value)

    train_config = ShapeEncodingConfig(
        steps=int(config["steps"]),
        learning_rate=float(config["learning_rate"]),
        eval_every=int(config["eval_every"]),
        seed=int(config["seed"]),
        near_surface_threshold=float(config["near_surface_threshold"]),
        near_surface_weight=float(config["near_surface_weight"]),
        use_eikonal=bool(config["use_eikonal"]),
        eikonal_weight=float(config["eikonal_weight"]),
        use_grad_supervision=bool(config["use_grad_supervision"]),
        grad_weight=float(config["grad_weight"]),
    )

    if config["model"] == "finr":
        model = build_shape_model(
            decomposition=config["decomposition"],
            axis_configs=build_axis_configs(config),
            rank=int(config["rank"]),
        )
        result = train_shape_model(model, grid, train_config, gradients=gradients)
    else:
        if bool(config["use_eikonal"]) or bool(config["use_grad_supervision"]):
            raise ValueError("Baseline SDF runner currently supports pure reconstruction experiments only. Set `use_eikonal` and `use_grad_supervision` to false.")
        model = BaselineShapeINR(
            hidden_dims=tuple(int(width) for width in config["hidden_dims"]),
            backend=config["backend"],
            encoding=config["encoding"],
            pos_enc=int(config["pos_enc"]),
            hash_levels=int(config["hash_levels"]),
            hash_features_per_level=int(config["hash_features_per_level"]),
            hash_min_resolution=int(config["hash_min_resolution"]),
            hash_max_resolution=int(config["hash_max_resolution"]),
            hash_table_size=int(config["hash_table_size"]),
        )
        result = train_baseline_shape_model(model, grid, train_config)

    run_name = config["experiment_name"]
    if run_name is None:
        run_name = (
            f"{sdf_path.stem}_{config['model']}"
            f"_mode-{config['decomposition']}_backend-{config['backend']}"
            f"_rank-{config['rank']}_seed-{config['seed']}"
        )
    output_dir = output_dir_root / run_name
    effective_config = dict(config)
    effective_config["sdf_path"] = str(sdf_path)
    effective_config["gradient_path"] = None if gradient_path is None else str(gradient_path)
    effective_config["output_dir"] = str(output_dir_root)
    summary = save_outputs(output_dir, grid, effective_config, result, count_params(result.params))
    summary["output_dir"] = str(output_dir)
    return summary, result


def main() -> None:
    config = build_config(parse_args())
    summary, _ = run_sdf_experiment(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

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

from finr_image import (  # noqa: E402
    BaselineImageINR,
    FactorizedImageINR,
    ImageEncodingConfig,
    load_image,
    train_baseline_image_model,
    train_factorized_image_model,
)

from finr_common import count_params, load_json_config, maybe_override, resolve_output_dir, resolve_project_path, write_json  # noqa: E402


DEFAULT_CONFIG = {
    "img_path": None,
    "model": "finr",
    "backend": "relu",
    "encoding": "raw",
    "rank": 64,
    "hidden_dims": [256, 256, 256],
    "pos_enc": 0,
    "steps": 5_000,
    "eval_every": 500,
    "learning_rate": 1e-4,
    "seed": 0,
    "downsample_stride": 1,
    "num_rbfs": 64,
    "hash_levels": 16,
    "hash_features_per_level": 2,
    "hash_min_resolution": 16,
    "hash_max_resolution": 524_288,
    "hash_table_size": 2**14,
    "output_dir": str(resolve_output_dir(None, "image_public")),
    "experiment_name": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Public image F-INR runner for reproducible experiments.")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config file.")
    parser.add_argument("--img-path", type=Path, default=None, help="Path to the RGB image.")
    parser.add_argument("--model", choices=["finr", "baseline"], default=None, help="Model family to train.")
    parser.add_argument(
        "--backend",
        choices=["relu", "siren", "siren2", "wire", "finer", "neurbf"],
        default=None,
        help="Axis-network backend.",
    )
    parser.add_argument(
        "--encoding",
        choices=["raw", "positional", "hash"],
        default=None,
        help="Coordinate encoding strategy.",
    )
    parser.add_argument("--rank", type=int, default=None, help="Rank for the factorized image model.")
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=None, help="Hidden dimensions.")
    parser.add_argument("--pos-enc", type=int, default=None, help="Notebook-style positional encoding size.")
    parser.add_argument("--steps", type=int, default=None, help="Optimization steps.")
    parser.add_argument("--eval-every", type=int, default=None, help="Metric logging interval.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--downsample-stride", type=int, default=None, help="Image downsampling stride.")
    parser.add_argument("--num-rbfs", type=int, default=None, help="RBF count for NeuRBF.")
    parser.add_argument("--hash-levels", type=int, default=None, help="Number of hash levels.")
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

    config["img_path"] = maybe_override(config.get("img_path"), None if args.img_path is None else str(args.img_path))
    config["model"] = maybe_override(config.get("model"), args.model)
    config["backend"] = maybe_override(config.get("backend"), args.backend)
    config["encoding"] = maybe_override(config.get("encoding"), args.encoding)
    config["rank"] = maybe_override(config.get("rank"), args.rank)
    config["hidden_dims"] = maybe_override(config.get("hidden_dims"), args.hidden_dims)
    config["pos_enc"] = maybe_override(config.get("pos_enc"), args.pos_enc)
    config["steps"] = maybe_override(config.get("steps"), args.steps)
    config["eval_every"] = maybe_override(config.get("eval_every"), args.eval_every)
    config["learning_rate"] = maybe_override(config.get("learning_rate"), args.learning_rate)
    config["seed"] = maybe_override(config.get("seed"), args.seed)
    config["downsample_stride"] = maybe_override(config.get("downsample_stride"), args.downsample_stride)
    config["num_rbfs"] = maybe_override(config.get("num_rbfs"), args.num_rbfs)
    config["hash_levels"] = maybe_override(config.get("hash_levels"), args.hash_levels)
    config["hash_features_per_level"] = maybe_override(config.get("hash_features_per_level"), args.hash_features_per_level)
    config["hash_min_resolution"] = maybe_override(config.get("hash_min_resolution"), args.hash_min_resolution)
    config["hash_max_resolution"] = maybe_override(config.get("hash_max_resolution"), args.hash_max_resolution)
    config["hash_table_size"] = maybe_override(config.get("hash_table_size"), args.hash_table_size)
    config["output_dir"] = str(resolve_output_dir(maybe_override(config.get("output_dir"), args.output_dir), "image_public"))
    config["experiment_name"] = maybe_override(config.get("experiment_name"), args.experiment_name)

    if config["img_path"] is None:
        raise ValueError("`img_path` must be provided via --img-path or --config.")
    if config["encoding"] is None:
        config["encoding"] = "positional" if int(config["pos_enc"]) > 0 else "raw"
    return config


def normalize_config(config: dict) -> dict:
    normalized = dict(DEFAULT_CONFIG)
    normalized.update(config)
    if normalized["img_path"] is None:
        raise ValueError("`img_path` must be provided.")
    if normalized["encoding"] is None:
        normalized["encoding"] = "positional" if int(normalized["pos_enc"]) > 0 else "raw"
    normalized["output_dir"] = str(resolve_output_dir(normalized.get("output_dir"), "image_public"))
    return normalized


def save_outputs(output_dir: Path, image: np.ndarray, config: dict, result, params_count: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)).save(output_dir / "input.png")
    Image.fromarray(np.clip(result.reconstruction * 255.0, 0.0, 255.0).astype(np.uint8)).save(output_dir / "reconstruction.png")

    with (output_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "loss", "psnr", "ssim"])
        for record in result.history:
            writer.writerow([record.step, record.loss, record.psnr, record.ssim])

    summary = {
        "model": config["model"],
        "backend": config["backend"],
        "encoding": config["encoding"],
        "rank": config["rank"],
        "parameter_count": params_count,
        "elapsed_seconds": result.elapsed_seconds,
        "final_step": result.history[-1].step,
        "final_loss": result.history[-1].loss,
        "final_psnr": result.history[-1].psnr,
        "final_ssim": result.history[-1].ssim,
    }
    write_json(output_dir / "metrics.json", summary)
    write_json(output_dir / "effective_config.json", config)
    return summary


def run_image_experiment(config: dict) -> tuple[dict, object]:
    config = normalize_config(config)
    image_path = resolve_project_path(config["img_path"])
    output_dir_root = resolve_output_dir(config["output_dir"], "image_public")
    image = load_image(image_path, downsample_stride=int(config["downsample_stride"]))
    train_config = ImageEncodingConfig(
        steps=int(config["steps"]),
        learning_rate=float(config["learning_rate"]),
        eval_every=int(config["eval_every"]),
        seed=int(config["seed"]),
    )
    hidden_dims = tuple(int(width) for width in config["hidden_dims"])

    if config["model"] == "finr":
        model = FactorizedImageINR(
            hidden_dims=hidden_dims,
            rank=int(config["rank"]),
            out_channels=3,
            pos_enc=int(config["pos_enc"]),
            backend=config["backend"],
            encoding=config["encoding"],
            hash_levels=int(config["hash_levels"]),
            hash_features_per_level=int(config["hash_features_per_level"]),
            hash_min_resolution=int(config["hash_min_resolution"]),
            hash_max_resolution=int(config["hash_max_resolution"]),
            hash_table_size=int(config["hash_table_size"]),
            num_rbfs=int(config["num_rbfs"]),
        )
        result = train_factorized_image_model(model, image, train_config)
    else:
        model = BaselineImageINR(
            hidden_dims=hidden_dims,
            out_channels=3,
            pos_enc=int(config["pos_enc"]),
            backend=config["backend"],
            encoding=config["encoding"],
            hash_levels=int(config["hash_levels"]),
            hash_features_per_level=int(config["hash_features_per_level"]),
            hash_min_resolution=int(config["hash_min_resolution"]),
            hash_max_resolution=int(config["hash_max_resolution"]),
            hash_table_size=int(config["hash_table_size"]),
        )
        result = train_baseline_image_model(model, image, train_config)

    run_name = config["experiment_name"]
    if run_name is None:
        run_name = (
            f"{image_path.stem}_{config['model']}"
            f"_backend-{config['backend']}_encoding-{config['encoding']}"
            f"_rank-{config['rank']}_seed-{config['seed']}"
        )
    output_dir = output_dir_root / run_name
    effective_config = dict(config)
    effective_config["img_path"] = str(image_path)
    effective_config["output_dir"] = str(output_dir_root)
    summary = save_outputs(output_dir, image, effective_config, result, count_params(result.params))
    summary["output_dir"] = str(output_dir)
    return summary, result


def main() -> None:
    config = build_config(parse_args())
    summary, _ = run_image_experiment(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

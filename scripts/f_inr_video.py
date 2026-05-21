import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from finr_video import (  # noqa: E402
    AxisConfig,
    BaselineVideoINR,
    VideoEncodingConfig,
    build_video_model,
    load_video_from_image_dir,
    load_video_from_mp4,
    load_video_from_npy,
    save_video_mp4,
    train_baseline_video_model,
    train_video_model,
)

from finr_common import count_params, load_json_config, maybe_override, resolve_output_dir, resolve_project_path, write_json  # noqa: E402


DEFAULT_CONFIG = {
    "video_path": None,
    "source_type": "mp4",
    "model": "finr",
    "decomposition": "cp",
    "backend": "relu",
    "encoding": "raw",
    "pos_enc": 0,
    "axis_backends": ["relu", "relu", "relu"],
    "axis_encodings": ["raw", "positional", "positional"],
    "axis_pos_enc": [0, 10, 10],
    "hidden_dims": [256, 256],
    "rank": 64,
    "steps": 3_000,
    "eval_every": 300,
    "learning_rate": 1e-4,
    "seed": 0,
    "eval_frame_index": 0,
    "target_height": 96,
    "target_width": 96,
    "max_frames": 16,
    "frame_rate": 10,
    "num_rbfs": 64,
    "hash_levels": 16,
    "hash_features_per_level": 2,
    "hash_min_resolution": 16,
    "hash_max_resolution": 524_288,
    "hash_table_size": 2**14,
    "output_dir": str(resolve_output_dir(None, "video_public")),
    "experiment_name": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Public video F-INR runner for reproducible experiments.")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config file.")
    parser.add_argument("--video-path", type=Path, default=None, help="Path to the input video source.")
    parser.add_argument("--source-type", choices=["mp4", "image_dir", "npy"], default=None, help="How to interpret the source.")
    parser.add_argument("--model", choices=["finr", "baseline"], default=None, help="Model family to train.")
    parser.add_argument("--mode", "--decomposition", dest="decomposition", choices=["cp", "tt", "tucker", "tr"], default=None, help="Tensor decomposition for the factorized video model.")
    parser.add_argument("--backend", choices=["relu", "siren", "siren2", "wire", "finer"], default=None, help="Baseline backend.")
    parser.add_argument("--encoding", choices=["raw", "positional", "hash"], default=None, help="Baseline encoding.")
    parser.add_argument("--pos-enc", type=int, default=None, help="Baseline positional encoding size.")
    parser.add_argument("--axis-backends", nargs=3, default=None, help="Per-axis backends for time, height, width.")
    parser.add_argument("--axis-encodings", nargs=3, default=None, help="Per-axis encodings for time, height, width.")
    parser.add_argument("--axis-pos-enc", nargs=3, type=int, default=None, help="Per-axis positional encoding sizes.")
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=None, help="Hidden dimensions.")
    parser.add_argument("--rank", type=int, default=None, help="Rank for the factorized video model.")
    parser.add_argument("--steps", type=int, default=None, help="Optimization steps.")
    parser.add_argument("--eval-every", type=int, default=None, help="Metric logging interval.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--eval-frame-index", type=int, default=None, help="Frame index for periodic metrics.")
    parser.add_argument("--target-height", type=int, default=None, help="Resize height.")
    parser.add_argument("--target-width", type=int, default=None, help="Resize width.")
    parser.add_argument("--max-frames", type=int, default=None, help="Frame cap.")
    parser.add_argument("--frame-rate", type=int, default=None, help="Saved reconstruction frame rate.")
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

    config["video_path"] = maybe_override(config.get("video_path"), None if args.video_path is None else str(args.video_path))
    config["source_type"] = maybe_override(config.get("source_type"), args.source_type)
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
    config["eval_frame_index"] = maybe_override(config.get("eval_frame_index"), args.eval_frame_index)
    config["target_height"] = maybe_override(config.get("target_height"), args.target_height)
    config["target_width"] = maybe_override(config.get("target_width"), args.target_width)
    config["max_frames"] = maybe_override(config.get("max_frames"), args.max_frames)
    config["frame_rate"] = maybe_override(config.get("frame_rate"), args.frame_rate)
    config["num_rbfs"] = maybe_override(config.get("num_rbfs"), args.num_rbfs)
    config["hash_levels"] = maybe_override(config.get("hash_levels"), args.hash_levels)
    config["hash_features_per_level"] = maybe_override(config.get("hash_features_per_level"), args.hash_features_per_level)
    config["hash_min_resolution"] = maybe_override(config.get("hash_min_resolution"), args.hash_min_resolution)
    config["hash_max_resolution"] = maybe_override(config.get("hash_max_resolution"), args.hash_max_resolution)
    config["hash_table_size"] = maybe_override(config.get("hash_table_size"), args.hash_table_size)
    config["output_dir"] = str(resolve_output_dir(maybe_override(config.get("output_dir"), args.output_dir), "video_public"))
    config["experiment_name"] = maybe_override(config.get("experiment_name"), args.experiment_name)

    if config["video_path"] is None:
        raise ValueError("`video_path` must be provided via --video-path or --config.")
    return config


def normalize_config(config: dict) -> dict:
    normalized = dict(DEFAULT_CONFIG)
    normalized.update(config)
    if normalized["video_path"] is None:
        raise ValueError("`video_path` must be provided.")
    normalized["output_dir"] = str(resolve_output_dir(normalized.get("output_dir"), "video_public"))
    return normalized


def load_video(config: dict) -> np.ndarray:
    path = resolve_project_path(config["video_path"])
    if config["source_type"] == "mp4":
        return load_video_from_mp4(
            path,
            target_height=int(config["target_height"]) if config["target_height"] is not None else None,
            target_width=int(config["target_width"]) if config["target_width"] is not None else None,
            max_frames=int(config["max_frames"]) if config["max_frames"] is not None else None,
        )
    if config["source_type"] == "image_dir":
        return load_video_from_image_dir(
            path,
            target_height=int(config["target_height"]) if config["target_height"] is not None else None,
            target_width=int(config["target_width"]) if config["target_width"] is not None else None,
            max_frames=int(config["max_frames"]) if config["max_frames"] is not None else None,
        )
    return load_video_from_npy(path)


def build_axis_configs(config: dict) -> tuple[AxisConfig, AxisConfig, AxisConfig]:
    names = ("time", "height", "width")
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


def save_outputs(output_dir: Path, video: np.ndarray, config: dict, result, params_count: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "input.npy", video)
    np.save(output_dir / "reconstruction.npy", result.reconstruction)
    save_video_mp4(video, output_dir / "input.mp4", frame_rate=int(config["frame_rate"]))
    save_video_mp4(result.reconstruction, output_dir / "reconstruction.mp4", frame_rate=int(config["frame_rate"]))

    with (output_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "loss", "frame_psnr", "frame_ssim", "video_psnr"])
        for record in result.history:
            writer.writerow([record.step, record.loss, record.frame_psnr, record.frame_ssim, record.video_psnr])

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
        "final_frame_psnr": result.history[-1].frame_psnr,
        "final_frame_ssim": result.history[-1].frame_ssim,
        "final_video_psnr": result.history[-1].video_psnr,
    }
    write_json(output_dir / "metrics.json", summary)
    write_json(output_dir / "effective_config.json", config)
    return summary


def run_video_experiment(config: dict) -> tuple[dict, object]:
    config = normalize_config(config)
    video_path = resolve_project_path(config["video_path"])
    output_dir_root = resolve_output_dir(config["output_dir"], "video_public")
    video = load_video(config)
    train_config = VideoEncodingConfig(
        steps=int(config["steps"]),
        learning_rate=float(config["learning_rate"]),
        eval_every=int(config["eval_every"]),
        seed=int(config["seed"]),
        eval_frame_index=int(config["eval_frame_index"]),
    )

    if config["model"] == "finr":
        model = build_video_model(
            decomposition=config["decomposition"],
            axis_configs=build_axis_configs(config),
            rank=int(config["rank"]),
            out_channels=int(video.shape[0]),
        )
        result = train_video_model(model, video, train_config)
    else:
        model = BaselineVideoINR(
            hidden_dims=tuple(int(width) for width in config["hidden_dims"]),
            out_channels=int(video.shape[0]),
            backend=config["backend"],
            encoding=config["encoding"],
            pos_enc=int(config["pos_enc"]),
            hash_levels=int(config["hash_levels"]),
            hash_features_per_level=int(config["hash_features_per_level"]),
            hash_min_resolution=int(config["hash_min_resolution"]),
            hash_max_resolution=int(config["hash_max_resolution"]),
            hash_table_size=int(config["hash_table_size"]),
        )
        result = train_baseline_video_model(model, video, train_config)

    run_name = config["experiment_name"]
    if run_name is None:
        run_name = (
            f"{video_path.stem}_{config['model']}"
            f"_mode-{config['decomposition']}_backend-{config['backend']}"
            f"_rank-{config['rank']}_seed-{config['seed']}"
        )
    output_dir = output_dir_root / run_name
    effective_config = dict(config)
    effective_config["video_path"] = str(video_path)
    effective_config["output_dir"] = str(output_dir_root)
    summary = save_outputs(output_dir, video, effective_config, result, count_params(result.params))
    summary["output_dir"] = str(output_dir)
    return summary, result


def main() -> None:
    config = build_config(parse_args())
    summary, _ = run_video_experiment(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

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

from finr_common import count_params, load_json_config, maybe_override, resolve_output_dir, resolve_project_path, write_json  # noqa: E402
from finr_sim import (  # noqa: E402
    AxisConfig,
    SimulationEncodingConfig,
    build_simulation_model,
    load_navier_stokes_super_resolution_data,
    train_simulation_model,
)


DEFAULT_CONFIG = {
    "data_path": None,
    "snapshot_pattern": "assets/simulations/w0_{index}.mat",
    "snapshot_indices": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "observation_times": [0.1, 0.2, 0.29, 0.4, 0.49, 0.59, 0.69, 0.8, 0.9, 0.99],
    "low_res_size": 32,
    "decomposition": "cp",
    "axis_backends": ["relu", "relu", "relu"],
    "axis_encodings": ["raw", "positional", "positional"],
    "axis_pos_enc": [0, 10, 10],
    "hidden_dims": [256, 256, 256],
    "rank": 64,
    "steps": 10_000,
    "eval_every": 500,
    "learning_rate": 1e-4,
    "seed": 0,
    "collocation_time_resolution": 128,
    "collocation_space_resolution": 128,
    "offset_count": 8,
    "offset_iter": 100,
    "viscosity": 0.01,
    "divergence_weight": 5_000.0,
    "initial_weight": 10_000.0,
    "observation_weight": 10_000.0,
    "num_rbfs": 64,
    "hash_levels": 16,
    "hash_features_per_level": 2,
    "hash_min_resolution": 16,
    "hash_max_resolution": 524_288,
    "hash_table_size": 2**14,
    "output_dir": str(resolve_output_dir(None, "simulation_public")),
    "experiment_name": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Public simulation super-resolution F-INR runner.")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config file.")
    parser.add_argument("--data-path", type=Path, default=None, help="Path to the `w_data.mat` file.")
    parser.add_argument("--snapshot-pattern", type=str, default=None, help="Pattern for low-resolution snapshot mats, e.g. `assets/simulations/w0_{index}.mat`.")
    parser.add_argument("--snapshot-indices", nargs="+", type=int, default=None, help="Snapshot indices used for sparse observations.")
    parser.add_argument("--observation-times", nargs="+", type=float, default=None, help="Observation times corresponding to `snapshot_indices`.")
    parser.add_argument("--low-res-size", type=int, default=None, help="Observed spatial resolution per axis.")
    parser.add_argument("--mode", "--decomposition", dest="decomposition", choices=["cp", "tt", "tucker", "tr"], default=None, help="Tensor decomposition.")
    parser.add_argument("--axis-backends", nargs=3, default=None, help="Per-axis backends for time, x, y.")
    parser.add_argument("--axis-encodings", nargs=3, default=None, help="Per-axis encodings for time, x, y.")
    parser.add_argument("--axis-pos-enc", nargs=3, type=int, default=None, help="Per-axis positional encodings for time, x, y.")
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=None, help="Hidden dimensions.")
    parser.add_argument("--rank", type=int, default=None, help="Tensor rank.")
    parser.add_argument("--steps", type=int, default=None, help="Optimization steps.")
    parser.add_argument("--eval-every", type=int, default=None, help="Evaluation interval.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--collocation-time-resolution", type=int, default=None, help="Collocation resolution in time.")
    parser.add_argument("--collocation-space-resolution", type=int, default=None, help="Collocation resolution in x and y.")
    parser.add_argument("--offset-count", type=int, default=None, help="Number of offset collocation grids.")
    parser.add_argument("--offset-iter", type=int, default=None, help="Steps per offset switch.")
    parser.add_argument("--viscosity", type=float, default=None, help="Viscosity coefficient.")
    parser.add_argument("--divergence-weight", type=float, default=None, help="Incompressibility penalty weight.")
    parser.add_argument("--initial-weight", type=float, default=None, help="Initial-condition weight.")
    parser.add_argument("--observation-weight", type=float, default=None, help="Sparse-observation weight.")
    parser.add_argument("--num-rbfs", type=int, default=None, help="RBF count for NeuRBF backends.")
    parser.add_argument("--hash-levels", type=int, default=None, help="Hash levels.")
    parser.add_argument("--hash-features-per-level", type=int, default=None, help="Hash features per level.")
    parser.add_argument("--hash-min-resolution", type=int, default=None, help="Minimum hash resolution.")
    parser.add_argument("--hash-max-resolution", type=int, default=None, help="Maximum hash resolution.")
    parser.add_argument("--hash-table-size", type=int, default=None, help="Hash table size.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for outputs.")
    parser.add_argument("--experiment-name", type=str, default=None, help="Optional run name override.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    config = dict(DEFAULT_CONFIG)
    config.update(load_json_config(args.config))

    config["data_path"] = maybe_override(config.get("data_path"), None if args.data_path is None else str(args.data_path))
    config["snapshot_pattern"] = maybe_override(config.get("snapshot_pattern"), args.snapshot_pattern)
    config["snapshot_indices"] = maybe_override(config.get("snapshot_indices"), args.snapshot_indices)
    config["observation_times"] = maybe_override(config.get("observation_times"), args.observation_times)
    config["low_res_size"] = maybe_override(config.get("low_res_size"), args.low_res_size)
    config["decomposition"] = maybe_override(config.get("decomposition"), args.decomposition)
    config["axis_backends"] = maybe_override(config.get("axis_backends"), args.axis_backends)
    config["axis_encodings"] = maybe_override(config.get("axis_encodings"), args.axis_encodings)
    config["axis_pos_enc"] = maybe_override(config.get("axis_pos_enc"), args.axis_pos_enc)
    config["hidden_dims"] = maybe_override(config.get("hidden_dims"), args.hidden_dims)
    config["rank"] = maybe_override(config.get("rank"), args.rank)
    config["steps"] = maybe_override(config.get("steps"), args.steps)
    config["eval_every"] = maybe_override(config.get("eval_every"), args.eval_every)
    config["learning_rate"] = maybe_override(config.get("learning_rate"), args.learning_rate)
    config["seed"] = maybe_override(config.get("seed"), args.seed)
    config["collocation_time_resolution"] = maybe_override(config.get("collocation_time_resolution"), args.collocation_time_resolution)
    config["collocation_space_resolution"] = maybe_override(config.get("collocation_space_resolution"), args.collocation_space_resolution)
    config["offset_count"] = maybe_override(config.get("offset_count"), args.offset_count)
    config["offset_iter"] = maybe_override(config.get("offset_iter"), args.offset_iter)
    config["viscosity"] = maybe_override(config.get("viscosity"), args.viscosity)
    config["divergence_weight"] = maybe_override(config.get("divergence_weight"), args.divergence_weight)
    config["initial_weight"] = maybe_override(config.get("initial_weight"), args.initial_weight)
    config["observation_weight"] = maybe_override(config.get("observation_weight"), args.observation_weight)
    config["num_rbfs"] = maybe_override(config.get("num_rbfs"), args.num_rbfs)
    config["hash_levels"] = maybe_override(config.get("hash_levels"), args.hash_levels)
    config["hash_features_per_level"] = maybe_override(config.get("hash_features_per_level"), args.hash_features_per_level)
    config["hash_min_resolution"] = maybe_override(config.get("hash_min_resolution"), args.hash_min_resolution)
    config["hash_max_resolution"] = maybe_override(config.get("hash_max_resolution"), args.hash_max_resolution)
    config["hash_table_size"] = maybe_override(config.get("hash_table_size"), args.hash_table_size)
    config["output_dir"] = str(resolve_output_dir(maybe_override(config.get("output_dir"), args.output_dir), "simulation_public"))
    config["experiment_name"] = maybe_override(config.get("experiment_name"), args.experiment_name)

    if config["data_path"] is None:
        raise ValueError("`data_path` must be provided via --data-path or --config.")
    return config


def build_axis_configs(config: dict) -> tuple[AxisConfig, AxisConfig, AxisConfig]:
    names = ("time", "x", "y")
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


def save_outputs(output_dir: Path, config: dict, data, result, params_count: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "predicted_vorticity.npy", result.vorticity)
    np.save(output_dir / "reference_vorticity.npy", data.w_eval)

    with (output_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "loss", "relative_l2"])
        for record in result.history:
            writer.writerow([record.step, record.loss, record.relative_l2])

    summary = {
        "decomposition": config["decomposition"],
        "rank": config["rank"],
        "parameter_count": params_count,
        "elapsed_seconds": result.elapsed_seconds,
        "final_step": result.history[-1].step,
        "final_loss": result.history[-1].loss,
        "final_relative_l2": result.history[-1].relative_l2,
    }
    write_json(output_dir / "metrics.json", summary)
    write_json(output_dir / "effective_config.json", config)
    return summary


def run_simulation_experiment(config: dict) -> tuple[dict, object]:
    data_path = resolve_project_path(config["data_path"])
    data = load_navier_stokes_super_resolution_data(
        data_path=data_path,
        snapshot_pattern=config["snapshot_pattern"],
        snapshot_indices=tuple(int(index) for index in config["snapshot_indices"]),
        observation_times=tuple(float(value) for value in config["observation_times"]),
        low_res_size=int(config["low_res_size"]),
    )

    train_config = SimulationEncodingConfig(
        steps=int(config["steps"]),
        learning_rate=float(config["learning_rate"]),
        eval_every=int(config["eval_every"]),
        seed=int(config["seed"]),
        collocation_time_resolution=int(config["collocation_time_resolution"]),
        collocation_space_resolution=int(config["collocation_space_resolution"]),
        offset_count=int(config["offset_count"]),
        offset_iter=int(config["offset_iter"]),
        viscosity=float(config["viscosity"]),
        divergence_weight=float(config["divergence_weight"]),
        initial_weight=float(config["initial_weight"]),
        observation_weight=float(config["observation_weight"]),
    )

    model = build_simulation_model(
        decomposition=config["decomposition"],
        axis_configs=build_axis_configs(config),
        rank=int(config["rank"]),
    )
    result = train_simulation_model(model, data, train_config)

    output_dir_root = resolve_output_dir(config["output_dir"], "simulation_public")
    run_name = config["experiment_name"]
    if run_name is None:
        run_name = (
            f"{data_path.stem}_mode-{config['decomposition']}"
            f"_backend-{config['axis_backends'][0]}-{config['axis_backends'][1]}-{config['axis_backends'][2]}"
            f"_rank-{config['rank']}_seed-{config['seed']}"
        )
    output_dir = output_dir_root / run_name

    effective_config = dict(config)
    effective_config["data_path"] = str(data_path)
    effective_config["output_dir"] = str(output_dir_root)
    summary = save_outputs(output_dir, effective_config, data, result, count_params(result.params))
    summary["output_dir"] = str(output_dir)
    return summary, result


def main() -> None:
    config = build_config(parse_args())
    summary, _ = run_simulation_experiment(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

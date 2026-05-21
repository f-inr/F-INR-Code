# F-INR
Initial code for 4 Applications 

- image encoding
- video encoding
- 3D SDF reconstruction
- Navier-Stokes super-resolution


## Layout

- `src/`: clean task packages
- `scripts/`: public entry points
- `configs/`: JSON configs for plug and play
- `assets/`: representative image, video, SDF, and simulation assets
- `notebooks/`: side-by-side F-INR vs baseline notebooks

## Environment

```bash
conda env create -f environment.yml
conda activate finr
pip install -e .
```


## Scripts

Image:

```bash
python scripts/f_inr_img.py --config configs/image/kodim23_finr.json
python scripts/f_inr_img.py --config configs/image/kodim23_baseline.json
python scripts/f_inr_img.py \
  --img-path assets/images/kodim23.png \
  --model finr \
  --backend wire \
  --encoding positional \
  --rank 128 \
  --steps 2000 \
  --output-dir publishable_runs/image_wire_rank128
```

Video:

```bash
python scripts/f_inr_video.py --config configs/video/ground_truth_finr.json
python scripts/f_inr_video.py --config configs/video/ground_truth_baseline.json
python scripts/f_inr_video.py \
  --video-path assets/videos/video.mp4 \
  --source-type mp4 \
  --model finr \
  --mode tt \
  --rank 32 \
  --axis-backends relu relu wire \
  --axis-encodings raw positional positional \
  --output-dir publishable_runs/video_tt_rank32
```

SDF:

```bash
python scripts/f_inr_sdf.py --config configs/sdf/armadillo_finr.json
python scripts/f_inr_sdf.py --config configs/sdf/armadillo_baseline.json
python scripts/f_inr_sdf.py \
  --sdf-path assets/sdf/voxels_sdfs_200_armadillo.npy \
  --model finr \
  --mode tucker \
  --rank 16 \
  --axis-backends relu relu relu \
  --axis-encodings positional positional positional \
  --downsample-stride 4 \
  --output-dir publishable_runs/sdf_tucker_rank16
```

Simulation super-resolution:

```bash
python scripts/f_inr_sim.py --config configs/simulation/ns_superres_finr.json
python scripts/f_inr_sim.py \
  --data-path assets/simulations/w_data.mat \
  --snapshot-pattern w0_{index}.mat \
  --mode cp \
  --rank 32 \
  --axis-backends relu relu wire \
  --axis-encodings raw positional positional \
  --low-res-size 32 \
  --output-dir publishable_runs/simulation_cp_rank32
```

Each run writes:

- `effective_config.json`
- `metrics.json`
- `history.csv`
- task-specific reconstructions

## Main Knobs

Image:
- `--model finr|baseline`
- `--backend relu|siren|siren2|wire|finer|neurbf`
- `--encoding raw|positional|hash`
- `--rank`
- `--hidden-dims`

Video:
- `--model finr|baseline`
- `--mode cp|tt|tucker|tr`
- `--backend` and `--encoding` for baseline
- `--axis-backends time height width`
- `--axis-encodings time height width`
- `--axis-pos-enc time height width`
- `--rank`

SDF:
- `--model finr|baseline`
- `--mode cp|tt|tucker|tr`
- `--backend` and `--encoding` for baseline
- `--axis-backends x y z`
- `--axis-encodings x y z`
- `--axis-pos-enc x y z`
- `--rank`
- `--truncate`
- `--downsample-stride`
- `--near-surface-threshold`
- `--near-surface-weight`
- `--use-eikonal`
- `--eikonal-weight`
- `--use-grad-supervision`

Baseline SDF runs in the public script are reconstruction-only. Eikonal and gradient-supervision options are for the factorized SDF model.

Simulation super-resolution:
- `--mode cp|tt|tucker|tr`
- `--axis-backends time x y`
- `--axis-encodings time x y`
- `--axis-pos-enc time x y`
- `--rank`
- `--low-res-size`
- `--collocation-time-resolution`
- `--collocation-space-resolution`
- `--offset-count`
- `--offset-iter`
- `--divergence-weight`
- `--initial-weight`
- `--observation-weight`

The simulation runner is the publishable F-INR / PINN path only. It uses the Navier-Stokes residual, incompressibility penalty, initial-condition loss, and sparse low-resolution observation loss from the legacy `simulations` workflow.

## Notebooks

- `notebooks/image_finr_vs_baseline.ipynb`
- `notebooks/video_finr_vs_baseline.ipynb`
- `notebooks/sdf_finr_vs_baseline.ipynb`

Each notebook uses the matching JSON configs in `configs/` and the assets in `assets/`.

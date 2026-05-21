from pathlib import Path

import cv2
import jax.numpy as jnp
import numpy as np


def to_channel_first(video: np.ndarray, normalize: bool = True) -> np.ndarray:
    array = np.asarray(video, dtype=np.float32)
    if normalize:
        array = array / 255.0
    return np.transpose(array, (3, 0, 1, 2))


def load_video_from_mp4(
    video_path: str | Path,
    target_height: int | None = None,
    target_width: int | None = None,
    max_frames: int | None = None,
    normalize: bool = True,
) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if target_height is not None and target_width is not None:
            frame = cv2.resize(frame, (target_width, target_height))
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if max_frames is not None and len(frames) >= max_frames:
            break

    cap.release()
    return to_channel_first(np.asarray(frames), normalize=normalize)


def load_video_from_image_dir(
    image_dir: str | Path,
    target_height: int | None = None,
    target_width: int | None = None,
    max_frames: int | None = None,
    normalize: bool = True,
) -> np.ndarray:
    image_dir = Path(image_dir)
    frame_paths = sorted(image_dir.glob("*.png"), key=lambda path: int(path.stem))
    if max_frames is not None:
        frame_paths = frame_paths[:max_frames]

    frames = []
    for path in frame_paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        if target_height is not None and target_width is not None:
            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    return to_channel_first(np.asarray(frames), normalize=normalize)


def load_video_from_npy(
    array_path: str | Path,
    channel_first: bool = True,
) -> np.ndarray:
    array = np.load(array_path).astype(np.float32)
    if channel_first:
        return array
    return np.transpose(array, (3, 0, 1, 2))


def video_axis_coordinates(video: np.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    _, frames, height, width = video.shape
    time_axis = jnp.linspace(-1.0, 1.0, frames, dtype=jnp.float32)[:, None]
    height_axis = jnp.linspace(-1.0, 1.0, height, dtype=jnp.float32)[:, None]
    width_axis = jnp.linspace(-1.0, 1.0, width, dtype=jnp.float32)[:, None]
    return time_axis, height_axis, width_axis


def flattened_video_coordinates(video: np.ndarray) -> jnp.ndarray:
    _, frames, height, width = video.shape
    time_axis = np.linspace(-1.0, 1.0, frames, dtype=np.float32)
    height_axis = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    width_axis = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    grid_t, grid_h, grid_w = np.meshgrid(time_axis, height_axis, width_axis, indexing="ij")
    stacked = np.stack(
        [grid_t.reshape(-1), grid_h.reshape(-1), grid_w.reshape(-1)],
        axis=1,
    )
    return jnp.asarray(stacked)


def save_video_mp4(
    video: np.ndarray,
    output_path: str | Path,
    frame_rate: int = 30,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    channel_last = np.transpose(video, (1, 2, 3, 0))
    if channel_last.dtype != np.uint8:
        channel_last = np.clip(channel_last * 255.0, 0.0, 255.0).astype(np.uint8)

    frames, height, width, _ = channel_last.shape
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        frame_rate,
        (width, height),
    )
    for frame in channel_last:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

from typing import Literal

from flax import linen as nn

from finr_video import AxisConfig, build_video_model


SimulationMode = Literal["cp", "tt", "tucker", "tr"]


def build_simulation_model(
    decomposition: SimulationMode,
    axis_configs: tuple[AxisConfig, AxisConfig, AxisConfig],
    rank: int,
) -> nn.Module:
    return build_video_model(
        decomposition=decomposition,
        axis_configs=axis_configs,
        rank=rank,
        out_channels=2,
    )

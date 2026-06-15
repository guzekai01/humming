import functools
import logging
import os
from typing import TYPE_CHECKING

import torch

from humming.config import GemmType
from humming.tune.base import DeviceHeuristics
from humming.tune.sm8x import (
    Sm80Heuristics,
    Sm86Heuristics,
    Sm87Heuristics,
    Sm89Heuristics,
)
from humming.tune.sm75 import Sm75Heuristics
from humming.tune.sm90 import Sm90Heuristics
from humming.tune.sm90_h20 import Sm90H20Heuristics
from humming.tune.sm100 import Sm100Heuristics

if TYPE_CHECKING:
    from humming.layer import HummingLayerMeta

logger = logging.getLogger("sglang.humming")


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


@functools.lru_cache(maxsize=256)
def _log_forced_batch_invariant_once(
    meta_desc: str,
    gemm_type: GemmType,
    shape_m: int | None,
    use_f16_accum: bool,
    config_desc: str,
) -> None:
    logger.warning(
        "[humming] HUMMING_FORCE_BATCH_INVARIANT=1 "
        f"gemm_type={gemm_type.value} shape_m={shape_m} "
        f"use_f16_accum={use_f16_accum} meta={meta_desc} config={config_desc}",
    )


def _meta_desc(meta: "HummingLayerMeta") -> str:
    return (
        f"a={meta.a_dtype},b={meta.b_dtype},m=?,n={meta.shape_n},k={meta.shape_k},"
        f"experts={meta.num_experts},input_scale_group={meta.input_scale_group_size},"
        f"weight_scale_group={meta.weight_scale_group_size},"
        f"fused_e8m0={meta.use_fused_e8m0_scale}"
    )


def _config_desc(config):
    if isinstance(config, dict):
        return (
            f"block_shape={config.get('block_shape')},"
            f"warp_shape={config.get('warp_shape')},"
            f"use_stream_k={config.get('use_stream_k')},"
            f"use_tma={config.get('use_tma', False)},"
            f"use_warp_spec={config.get('use_warp_spec', False)},"
            f"use_mbarrier={config.get('use_mbarrier', False)}"
        )
    preview = []
    for item in config[:3]:
        min_m, max_m, cfg = item
        preview.append(f"[{min_m},{max_m}):{_config_desc(cfg)}")
    suffix = "" if len(config) <= 3 else f",... total={len(config)}"
    return "; ".join(preview) + suffix

heuristics_map: dict[int, type[DeviceHeuristics]] = {
    75: Sm75Heuristics,
    80: Sm80Heuristics,
    86: Sm86Heuristics,
    87: Sm87Heuristics,
    89: Sm89Heuristics,
    90: Sm90Heuristics,
    100: Sm100Heuristics,
    103: Sm100Heuristics,
    120: Sm89Heuristics,
    121: Sm89Heuristics,
}


def get_heuristics_class(
    sm_version: int | tuple[int, int] | None = None,
    device: int | torch.device | None = None,
) -> type[DeviceHeuristics]:
    if sm_version is None:
        sm_version = torch.cuda.get_device_capability(device)
    if isinstance(sm_version, tuple):
        sm_version = sm_version[0] * 10 + sm_version[1]
    assert isinstance(sm_version, int)
    name = torch.cuda.get_device_name(device)
    if "H20" in name and "H200" not in name:
        return Sm90H20Heuristics

    return heuristics_map[sm_version]


@functools.lru_cache(maxsize=1024)
def _get_heuristics_config_cached(
    meta: "HummingLayerMeta | dict",
    shape_m: int | None = None,
    use_f16_accum: bool = False,
    use_batch_invariant: bool = False,
    gemm_type: str | GemmType = "dense",
    force_batch_invariant: bool = False,
):
    from humming.layer import HummingLayerMeta

    if isinstance(gemm_type, str):
        gemm_type = GemmType(gemm_type)
    if force_batch_invariant:
        use_batch_invariant = True

    if isinstance(meta, dict):
        meta = HummingLayerMeta(**meta)
    heuristics_cls = get_heuristics_class()
    if isinstance(shape_m, int):
        config = heuristics_cls.get_config(
            meta=meta,
            shape_m=shape_m,
            use_f16_accum=use_f16_accum,
            use_batch_invariant=use_batch_invariant,
            gemm_type=gemm_type,
        )
        if force_batch_invariant:
            _log_forced_batch_invariant_once(
                _meta_desc(meta), gemm_type, shape_m, use_f16_accum, _config_desc(config)
            )
        return config
    else:
        configs = heuristics_cls.get_configs(
            meta=meta,
            use_f16_accum=use_f16_accum,
            use_batch_invariant=use_batch_invariant,
            gemm_type=gemm_type,
        )
        if force_batch_invariant:
            _log_forced_batch_invariant_once(
                _meta_desc(meta), gemm_type, None, use_f16_accum, _config_desc(configs)
            )
        return configs


def get_heuristics_config(
    meta: "HummingLayerMeta | dict",
    shape_m: int | None = None,
    use_f16_accum: bool = False,
    use_batch_invariant: bool = False,
    gemm_type: str | GemmType = "dense",
):
    normalized_gemm_type = GemmType(gemm_type) if isinstance(gemm_type, str) else gemm_type
    force_batch_invariant = (
        _env_flag("HUMMING_FORCE_BATCH_INVARIANT")
        and normalized_gemm_type != GemmType.DENSE
    )
    return _get_heuristics_config_cached(
        meta=meta,
        shape_m=shape_m,
        use_f16_accum=use_f16_accum,
        use_batch_invariant=use_batch_invariant,
        gemm_type=normalized_gemm_type,
        force_batch_invariant=force_batch_invariant,
    )

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attach D's authenticated native host extents on the EngineCore owner."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kitepond.core.contracts import Wait
from kitepond.models.glm5next.host_cold_extent import ColdExtentLayout
from kitepond.models.glm5next.region_socket import join_region
from kitepond.models.glm5next.shared_region import NativeHiSparseMemfdRegion

from vllm.v1.hisparse.runtime import get_hisparse_host_pool

from .glm_host_backing import _Launch
from .native_glm_cold_provider import NativeGlmRankMemory

if TYPE_CHECKING:
    from .glm_terminal_bootstrap import TerminalRankEnrollment
    from .glm_terminal_worker import GlmTerminalWorker


def register_worker_cold_memory(worker: GlmTerminalWorker) -> ColdExtentLayout:
    """Recognize the real native registered tensor on its owning rank."""
    config = worker.model_runner.kv_cache_config
    pool = get_hisparse_host_pool(worker.vllm_config, config)
    memory = NativeGlmRankMemory.from_registered_cold_backing(pool.registered)
    if isinstance(memory, Wait):
        raise ValueError("D worker has no enrolled native GLM cold extent")
    views = worker.kitepond_native_state_views()
    if (
        views.descriptor != memory.region.descriptor
        or views.rank != memory.rank
        or memory.mla_bytes
        != config.hisparse_host_num_blocks * config.hisparse_host_block_stride
        or hasattr(worker, "_kitepond_native_cold_memory")
    ):
        raise ValueError("GLM cold extent differs from its registered native owner")
    worker._kitepond_native_cold_memory = memory
    assert memory._extent is not None
    return memory._extent.layout


@dataclass(frozen=True)
class NativeGlmColdRegions:
    """Borrow existing bytes for FSL2; the D workers remain the owners."""

    layout: ColdExtentLayout
    regions: tuple[NativeHiSparseMemfdRegion, ...]
    layout_sha256: str

    @classmethod
    def open(
        cls, config, reports: tuple[TerminalRankEnrollment, ...]
    ) -> NativeGlmColdRegions:
        """Open only reports whose terminal credentials already authenticated."""
        launch = _Launch.from_config(config)
        if launch.role != "authority" or launch.cold_extent is None:
            raise ValueError("GLM cold storage needs explicit D native extents")
        reports = tuple(sorted(reports, key=lambda report: report.rank))
        if tuple(report.rank for report in reports) != (0, 1, 2, 3):
            raise ValueError("GLM cold startup requires exactly four native extents")
        layout = reports[0].cold_layout
        if (
            type(layout) is not ColdExtentLayout
            or launch.cold_extent.layout(layout.mla_bytes) != layout
            or any(
                report.cold_layout != layout
                or report.host_layout_sha256 != reports[0].host_layout_sha256
                or (report.identity.engine_id, report.identity.boot_id)
                != (launch.authority_engine_id, launch.authority_boot_id)
                for report in reports
            )
        ):
            raise ValueError("GLM cold startup has different worker extent identities")
        with ExitStack() as failure:
            regions = []
            for report in reports:
                region = join_region(
                    launch.socket_directory,
                    authority_engine_id=launch.authority_engine_id,
                    authority_boot_id=launch.authority_boot_id,
                    source_rank=report.rank,
                    layout_sha256=report.host_layout_sha256,
                )
                failure.callback(region.close)
                if (
                    region.descriptor.fingerprint() != report.region_sha256
                    or len(region.mapping) != layout.extent
                    or region.owner
                ):
                    raise ValueError("GLM cold startup received a different D region")
                regions.append(region)
            authority = cls(layout, tuple(regions), reports[0].host_layout_sha256)
            failure.pop_all()
        return authority

    def close(self) -> None:
        with ExitStack() as cleanup:
            for region in self.regions:
                cleanup.callback(region.close)

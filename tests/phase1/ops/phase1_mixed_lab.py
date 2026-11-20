# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the sealed C1 and mixed lab on already-started, isolated native 4P4D."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import secrets
import shutil
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpx
from kitepond.integrations.vllm.cold_prefill_http import is_control_token
from kitepond.ops.capture import SOURCES_SCHEMA, _process_start
from kitepond.ops.cli import _read_manifest
from kitepond.ops.glm_cold_probe import ProbeConfig, ProbeFailure, run_probe
from kitepond.ops.glm_warm_probe import (
    DecodeIdentity,
    _prefill_counters,
    run_warm_probe,
)
from kitepond.ops.wheel import sha256_file
from phase1_soak_driver import Population, Workload, _emit, _run_live

SCHEMA = "kitepond.glm-native-mixed-lab.v1"
RANKS = [0, 1, 2, 3]
PHASES = ("cold", "warm", "nvme")
PORTS = {"prefill": 18163, "decode": 18162}
MODEL_LIMIT = 8192
MIXED_SECONDS = 7200
MODEL_LABEL = "nvidia/GLM-5.3-Flash-NVFP4"


def _require(valid, reason):
    if not valid:
        raise ValueError(reason)


def _read(path):
    return json.loads(path.read_bytes())


def _arg(argv, name):
    _require(argv.count(name) == 1, f"Role must define {name} once")
    return argv[argv.index(name) + 1]


def _role_config(pair, role):
    argv = pair["roles"][role]["argv"]
    _require(
        _arg(argv, "--host") == "127.0.0.1"
        and int(_arg(argv, "--port")) == PORTS[role]
        and int(_arg(argv, "--max-model-len")) == MODEL_LIMIT
        and int(_arg(argv, "--max-num-seqs")) == 4
        and _arg(argv, "--served-model-name") == MODEL_LABEL
        and "--no-enable-prefix-caching" in argv
        and _arg(argv, "--tensor-parallel-size") == "4"
        and _arg(argv, "--decode-context-parallel-size") == "4"
        and "--enable-expert-parallel" in argv,
        "The sealed role exceeds the verified 8K/C4 lab geometry",
    )
    additional = json.loads(_arg(argv, "--additional-config"))
    _require(
        additional["kitepond"]["stage"] == role
        and additional["kitepond"]["boot_id"] == pair["authority_boot_id"]
        and additional["kitepond_glm_cold_prefill"]["max_pending"] == 4,
        "Sealed native pending budget or engine identity differs",
    )
    return additional


def read_seals(args):
    raw = _read(args.plan)
    _require(
        raw.keys()
        == {
            "schema",
            "pair_sha256",
            "release_sha256",
            "config_sha256",
            "seed",
            "token_range",
            "populations",
            "suffix_tokens",
            "timeout_seconds",
            "sample_seconds",
            "drain_timeout_seconds",
            "min_host_available_bytes",
            "max_device_used_fraction",
            "max_waiting_per_role",
            "min_nvme_free_bytes",
            "nvme_probe_sha256",
        }
        and raw["schema"] == SCHEMA
        and all(
            is_control_token(raw[key])
            for key in (
                "pair_sha256",
                "release_sha256",
                "config_sha256",
                "nvme_probe_sha256",
            )
        ),
        "Unrecognized or incomplete native mixed lab plan",
    )
    _require(
        raw["pair_sha256"] == sha256_file(args.pair)
        and raw["release_sha256"] == sha256_file(args.manifest),
        "Pair and release hashes must match the lab plan",
    )
    release = _read_manifest(args.manifest)
    filename = release["wheel"]["filename"]
    _require(
        Path(filename).name == filename
        and sha256_file(args.manifest.with_name(filename))
        == release["wheel"]["sha256"],
        "The sealed Kitepond wheel does not match the release manifest",
    )
    pair = _read(args.pair)
    _require(
        pair["mode"] == "native-warm-return"
        and release["base"]["image"].endswith(pair["base_image"])
        and set(pair["roles"]) == set(PORTS),
        "The lab requires the named native warm pair and exact base",
    )
    role_config = {role: _role_config(pair, role) for role in PORTS}
    _require(
        role_config["decode"]["kitepond_glm_nvme_control"]["max_pending"] >= 2
        and role_config["decode"]["kitepond_glm_cold"]["root"] == pair["nvme"]["root"]
        and role_config["decode"]["kitepond_glm_cold"]["boot_id"]
        == pair["authority_boot_id"],
        "The full pair must enable the real D cold tier",
    )
    work = Workload(
        raw["seed"],
        *raw["token_range"],
        tuple(Population(**row) for row in raw["populations"]),
    )
    _require(
        all(
            type(raw[key]) is int and raw[key] > 0
            for key in (
                "suffix_tokens",
                "timeout_seconds",
                "sample_seconds",
                "drain_timeout_seconds",
                "min_nvme_free_bytes",
            )
        )
        and raw["sample_seconds"] < raw["timeout_seconds"] < 120
        and all(
            row.prompt_tokens + 2 * row.output_tokens + raw["suffix_tokens"]
            <= MODEL_LIMIT
            for row in work.populations
        ),
        "Warm turn and both outputs must fit the actual model budget",
    )
    return raw, release, pair, work


def _process_source(runtime, port, role, boot_id):
    pid = int((runtime / role / "PID").read_text(encoding="ascii"))
    return {
        "port": port,
        "pid": pid,
        "process_start_ticks": _process_start(pid),
        "engine_epoch": f"{boot_id}/{role}",
        "model_name": MODEL_LABEL,
        "engine_ids": ["0"],
    }


def _capture_sources(args, pair):
    _require(
        args.runtime.is_dir() and args.runtime.stat().st_mode & 0o077 == 0,
        "Native lab needs the private, same-container runtime",
    )
    ownership = _read(args.runtime / "OWNERSHIP.json")
    _require(
        ownership["plan_sha256"] == sha256_file(args.pair)
        and ownership["engine_boot_id"] == pair["authority_boot_id"]
        and ownership["gpu_count"] == 8
        and ownership["models_ready"] is True,
        "Ownership evidence does not describe this sealed pair",
    )
    return {
        "schema": SOURCES_SCHEMA,
        "release_manifest_sha256": sha256_file(args.manifest),
        "deployment": pair["authority_engine_id"],
        "roles": {
            role: _process_source(
                args.runtime, port, role, pair["mooncake_api"][boot_key]
            )
            for role, port, boot_key in (
                ("prefill", PORTS["prefill"], "p_boot_id"),
                ("decode", PORTS["decode"], "d_boot_id"),
            )
        },
    }


def _checked_turn(turn):
    ctx = {
        name: turn[name]
        for name in (
            "prompt_tokens",
            "prefill_api_prompt_tokens",
            "verified_ready_ranks",
            "ready_tokens_per_rank",
            "output_tokens",
            "probe_wall_seconds",
        )
    }
    _require(
        turn["status"] == "completed"
        and type(ctx["prompt_tokens"]) is int
        and ctx["ready_tokens_per_rank"] == ctx["prompt_tokens"] - 1
        and ctx["prefill_api_prompt_tokens"]
        in (ctx["prompt_tokens"], ctx["ready_tokens_per_rank"])
        and ctx["verified_ready_ranks"] == 4
        and type(ctx["output_tokens"]) is int
        and ctx["output_tokens"] > 0,
        "Probe omitted the native READY or output accounting",
    )
    return ctx


def _checked_warm(reply):
    turns = [_checked_turn(reply[name]) for name in ("first_turn", "second_turn")]
    reused = reply["reused_prefix_tokens"]
    _require(
        reply["settled_reverse_ranks"] == RANKS
        and reply["prefill_released"] is True
        and reply["decode_released"] is True
        and 0 < reused < turns[-1]["ready_tokens_per_rank"]
        and reply["prefill_second_turn_source_tokens"]
        == {
            "local_compute": turns[-1]["ready_tokens_per_rank"] - reused,
            "local_cache_hit": 0,
            "external_kv_transfer": reused,
        },
        "Warm turn lacks an attributable D→P fence or exact reuse accounting",
    )
    return turns, reused


def _nvme_module(path, expected):
    _require(
        not path.is_symlink() and sha256_file(path) == expected,
        "The native NVMe client differs from its sealed source",
    )
    data = path.read_bytes()
    _require(hashlib.sha256(data).hexdigest() == expected, "NVMe client changed")
    module = ModuleType("_kitepond_lab_nvme_client")
    exec(compile(data, str(path), "exec"), module.__dict__)
    return module


class MixedWorkload:
    def __init__(self, args, plan, pair, config):
        self.args, self.plan, self.config = args, plan, config
        peer = pair["mooncake_api"]
        self.decoder = DecodeIdentity(
            peer["d_engine_id"],
            peer["d_boot_id"],
            pair["checkpoint_identity"]["model_sha256"],
            plan["config_sha256"],
        )
        self.storage_root = Path(pair["nvme"]["root"]).resolve(strict=True)
        _require(
            self.storage_root == args.runtime / "kitepond-nvme"
            and not self.storage_root.stat().st_mode & 0o077,
            "NVMe lab must use this candidate's private persistent path",
        )
        self.counts = dict.fromkeys(PHASES, 0)
        self.phase = 0
        self.nvme_bytes = 0
        self.nvme_generation = 0
        self.nvme_namespace = f"lab-{self.decoder.boot_id[:12]}-{secrets.token_hex(8)}"
        self.nvme_device = self.storage_root.stat().st_dev
        self.nvme_initial_free = shutil.disk_usage(self.storage_root).free
        self.expected_prefill = 0
        self.nvme = None
        if args.mode != "c1":
            self.nvme = _nvme_module(args.nvme_probe, plan["nvme_probe_sha256"])
            self.nvme_token = os.environ.get(self.nvme.TOKEN_ENV)
            _require(
                is_control_token(self.nvme_token),
                "Missing private native NVMe token, mixed run is unavailable",
            )
            self.nvme_key = self.nvme.private_key(
                Path(os.environ["KITEPOND_GLM_NVME_OPERATOR_KEY_FILE"])
            )
            self.catalog_key = self.nvme.private_key(
                Path(os.environ["KITEPOND_GLM_NVME_CATALOG_KEY_FILE"])
            )

    async def _nvme_cycle(self, client, item, tokens, suffix):
        from kitepond.integrations.vllm.glm_cold_catalog import GlmColdCatalog
        from kitepond.integrations.vllm.glm_cold_runtime import ColdDirectoryScope

        space_before = shutil.disk_usage(self.storage_root).free
        _require(
            self.storage_root.stat().st_dev == self.nvme_device
            and space_before >= self.plan["min_nvme_free_bytes"],
            "Persistent disk changed or its reserve is exhausted",
        )
        self.nvme_generation += 1
        name, generation = self.nvme_namespace, self.nvme_generation
        evidence = self.args.output / f"nvme-{self.phase}"
        evidence.mkdir(mode=0o700)
        turn_path = evidence / "prompt.json"
        suffix_path = evidence / "suffix.json"
        oracle_path = evidence / "private-oracle.json"
        # Keep raw tokens in a private, run-owned temporary transcript. Only
        # the counts, digests and rank proof survive successful validation.
        _emit(turn_path, tokens)
        _emit(suffix_path, suffix)
        fields = SimpleNamespace(
            engine=self.decoder.engine_id,
            boot=self.decoder.boot_id,
            model=self.config.model,
            artifact=oracle_path,
            prefill_engine=self.config.prefill_engine_id,
            model_sha256=self.decoder.model_fingerprint,
            config_sha256=self.decoder.config_fingerprint,
            prompt=turn_path,
            suffix=suffix_path,
            session=name,
            generation=generation,
            seed=self.config.seed,
            output_tokens=item.output_tokens,
            timeout=self.config.timeout_seconds,
        )
        call = self.nvme.command(client, fields, self.nvme_key, self.nvme_token)
        completed = False
        try:
            archive = await self.nvme.archive(client, fields, call)
            catalog = GlmColdCatalog(
                self.storage_root,
                engine_id=fields.engine,
                boot_id=fields.boot,
                model_fingerprint=fields.model_sha256,
                config_fingerprint=fields.config_sha256,
                signing_key=self.catalog_key,
            )
            entry = catalog.reopen(ColdDirectoryScope(name, generation))
            _require(
                entry is not None
                and catalog.verify(entry)
                and entry.source_boot_id == entry.target_boot_id == fields.boot
                and entry.accepted_prefix_sha256 == archive["prefix_sha256"],
                "Closed checkpoint lacks the matching authenticated four-rank catalog",
            )
            restore = await self.nvme.restore(client, fields, call, same_boot=True)
            _require(
                archive["status"] == "committed"
                and restore["status"] == "completed"
                and restore["accepted_native"] is True
                and restore["matches_warm_token_ids"] is True
                and restore["source_boot"] == restore["restore_boot"] == fields.boot
                and restore["restored_prefix_tokens"] == archive["boundary"]
                and restore["completion_tokens"] == item.output_tokens,
                "Same-epoch NVMe path lacks accepted native D output",
            )
            for request in (
                archive["source_request_id"],
                restore["restore_request_id"],
            ):
                _require(
                    await call("forget", request) == {"state": "forgotten"},
                    "NVMe operator retained a completed operation",
                )
            ranks = [
                sum(part.nbytes for part in rank.parts)
                for rank in entry.checkpoint.plans
            ]
            _require(
                len(ranks) == 4 and all(ranks), "NVMe manifest lacks a physical rank"
            )
            _emit(
                evidence / "closed-proof.json",
                {
                    "archive": archive,
                    "restore": restore,
                    "aggregate_sha256": entry.aggregate_sha256,
                    "rank_manifest_payload_bytes": ranks,
                    "storage_device": self.nvme_device,
                    "filesystem_free_bytes_before": space_before,
                    "filesystem_free_bytes_after": shutil.disk_usage(
                        self.storage_root
                    ).free,
                    "proof_scope": "same_role_boot_fsl2_payload_then_native_decode",
                },
            )
            warm = archive["warm"] | {
                "settled_reverse_ranks": archive["settled_reverse_ranks"]
            }
            turns, reused = _checked_warm(warm)
            self.nvme_bytes += sum(ranks)
            completed = True
            return turns, reused, restore["completion_tokens"], entry.aggregate_sha256
        except RuntimeError as error:
            restore_failure = getattr(self.nvme, "RestoreProbeFailure", None)
            if restore_failure is not None and isinstance(error, restore_failure):
                _emit(evidence / "restore-failure.json", error.receipt)
            raise
        finally:
            if completed:
                for path in (turn_path, suffix_path, oracle_path):
                    path.unlink(missing_ok=True)

    async def request(self, client, item, tokens, auth, api_key):
        phase = PHASES[self.phase % len(PHASES)]
        self.phase += 1
        kv_restore_output = 0
        checkpoint = None
        config = replace(
            self.config, output_tokens=item.output_tokens, capture_token_ids=True
        )
        # A copied tail has no impact on global APC because the first eight
        # tokens are the stream's fresh nonce. This is only the second turn.
        suffix = tokens[-self.plan["suffix_tokens"] :]
        try:
            before = await _prefill_counters(client, self.config.prefill_port)
            if phase == "cold":
                turns = [
                    _checked_turn(
                        await run_probe(client, config, tokens, auth, api_key)
                    )
                ]
                reused = 0
            elif phase == "warm":
                result = await run_warm_probe(
                    client, config, self.decoder, tokens, suffix, auth, api_key
                )
                turns, reused = _checked_warm(result)
            else:
                turns, reused, kv_restore_output, checkpoint = await self._nvme_cycle(
                    client, item, tokens, suffix
                )
            after = await _prefill_counters(client, self.config.prefill_port)
            observed = {key: after[key] - before[key] for key in before}
            logical = sum(turn["ready_tokens_per_rank"] for turn in turns)
            _require(
                observed
                == {
                    "local_compute": logical - reused,
                    "local_cache_hit": 0,
                    "external_kv_transfer": reused,
                },
                "Per-phase P source increments disagree with native READY",
            )
        except ProbeFailure:
            raise
        except (
            httpx.HTTPError,
            TimeoutError,
            KeyError,
            OSError,
            TypeError,
            ValueError,
            RuntimeError,
        ) as issue:
            raise ProbeFailure(f"lab_{phase}", "", "", type(issue).__name__) from None
        self.counts[phase] += 1
        self.expected_prefill += logical
        return {
            "status": "completed",
            "phase": phase,
            "prefill_api_prompt_tokens": sum(
                turn["prefill_api_prompt_tokens"] for turn in turns
            ),
            "output_tokens": sum(turn["output_tokens"] for turn in turns)
            + kv_restore_output,
            "settled_turns": turns,
            "prefill_source_delta": observed,
            "native_nvme_restore_output_tokens": kv_restore_output,
            "checkpoint_aggregate_sha256": checkpoint,
        }


def _eligible(summary, run, expected, *, complete_window):
    return (
        summary["completed"] > 0
        and summary["failed"] == 0
        and not summary["stopped_by_guard"]
        and not summary["resource_guard_errors"]
        and summary["peak_client_inflight"] == 1
        and summary["native_import_and_lease_drained"]
        and summary["native_output_counts_match"]
        and summary["prefill_engine_logical_tokens"] == run.expected_prefill
        and all(run.counts[phase] > 0 for phase in expected)
        and (not complete_window or summary["reached_active_deadline"])
    )


def _verify_preflight(path, mode, paired, counts):
    receipt = _read(path)
    _require(
        receipt["schema"] == SCHEMA
        and receipt["mode"] == mode
        and receipt["evidence_complete"] is True
        and receipt["provenance"] == paired
        and receipt["completed_by_phase"] == counts
        and receipt["report_sha256"] == sha256_file(path.with_name("report.json"))
        and receipt["summary_sha256"] == sha256_file(path.with_name("summary.json")),
        f"Mixed lab needs a complete {mode} receipt from these live processes",
    )


async def _main(args, raw, release, pair, work):
    sources = _capture_sources(args, pair)
    boot = pair["authority_boot_id"]
    paired = {
        "pair": raw["pair_sha256"],
        "manifest": raw["release_sha256"],
        "wheel": release["wheel"]["sha256"],
        "boot": boot,
        "native_checkpoint_model_sha256": pair["checkpoint_identity"]["model_sha256"],
        "native_role_boots": {
            "prefill": pair["mooncake_api"]["p_boot_id"],
            "decode": pair["mooncake_api"]["d_boot_id"],
        },
        "process_sources": sources["roles"],
        "lab_plan": sha256_file(args.plan),
        "lab_driver": sha256_file(Path(__file__)),
        "soak_driver": sha256_file(Path(__file__).with_name("phase1_soak_driver.py")),
    }
    if args.mode != "c1":
        _verify_preflight(
            args.c1_receipt,
            "c1",
            paired,
            {"cold": 1, "warm": 0, "nvme": 0},
        )
    if args.mode == "mixed":
        _verify_preflight(
            args.smoke_receipt,
            "smoke",
            paired,
            {"cold": 1, "warm": 1, "nvme": 1},
        )
    gpu_map = _read(args.gpus)
    _require(
        set(gpu_map) == {str(i) for i in range(8)},
        "GPU identities must come from the sealed eight-GPU launch",
    )
    config = ProbeConfig(
        PORTS["prefill"],
        PORTS["decode"],
        f"http://127.0.0.1:{pair['roles']['prefill']['mooncake_bootstrap_port']}",
        pair["mooncake_api"]["p_engine_id"],
        MODEL_LABEL,
        timeout_seconds=raw["timeout_seconds"],
        seed=raw["seed"],
    )
    run = MixedWorkload(args, raw, pair, config)
    # Existing capture/soak owns before/after/drain and process-epoch accounting.
    _emit(args.output.with_suffix(".sources.json"), sources)
    args.sources = args.output.with_suffix(".sources.json")
    active_seconds = {
        "c1": raw["timeout_seconds"],
        "smoke": 3 * raw["timeout_seconds"],
        "mixed": MIXED_SECONDS,
    }
    guard = {
        "concurrency": 1,
        "active_seconds": active_seconds[args.mode],
        "sample_seconds": raw["sample_seconds"],
        "drain_timeout_seconds": raw["drain_timeout_seconds"],
        "max_failures": 0,
        "resource_limits": {
            "gpu_uuids": list(gpu_map.values()),
            "min_host_available_bytes": raw["min_host_available_bytes"],
            "max_device_used_fraction": raw["max_device_used_fraction"],
            "max_waiting_per_role": raw["max_waiting_per_role"],
            **(
                {"memory_event_paths": {"candidate": "/sys/fs/cgroup/memory.events"}}
                if Path("/sys/fs/cgroup/memory.events").is_file()
                else {}
            ),
        },
    }
    has_mixed_phases = args.mode != "c1"
    attempts = {"c1": 1, "smoke": 3, "mixed": None}
    summary = await _run_live(
        args,
        guard,
        work,
        config,
        raw["release_sha256"],
        workload_probe=run.request,
        max_attempts=attempts[args.mode],
        peak_prompt_tokens=max(
            row.prompt_tokens + row.output_tokens + raw["suffix_tokens"]
            if has_mixed_phases
            else row.prompt_tokens
            for row in work.populations
        ),
        scope=f"observed_local_native_lab_{args.mode}",
    )
    expected = PHASES if has_mixed_phases else ("cold",)
    result = {
        "schema": SCHEMA,
        "mode": args.mode,
        "provenance": paired,
        "evidence_complete": _eligible(
            summary, run, expected, complete_window=args.mode == "mixed"
        ),
        "completed_by_phase": run.counts,
        "verified_nvme_rank_manifest_payload_bytes": run.nvme_bytes,
        "storage_identity": {
            "device": run.nvme_device,
            "initial_free_bytes": run.nvme_initial_free,
            "final_free_bytes": shutil.disk_usage(run.storage_root).free,
        },
        "counts_scope": "same_processes_c1_exclusive_by_phase",
        "release_qualification": "not_evaluated_native_overlay_lab_only",
        "report_sha256": sha256_file(args.output / "report.json"),
        "summary_sha256": sha256_file(args.output / "summary.json"),
    }
    _emit(args.output / "lab-proof.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("c1", "smoke", "mixed"))
    for name in ("manifest", "pair", "plan", "runtime", "gpus", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--c1-receipt", type=Path)
    parser.add_argument("--smoke-receipt", type=Path)
    parser.add_argument("--nvme-probe", type=Path)
    args = parser.parse_args()
    try:
        _require(
            args.mode == "c1" or (args.c1_receipt and args.nvme_probe),
            "Mixed mode requires both the same-boot C1 receipt and sealed NVMe client",
        )
        _require(
            args.mode != "mixed" or args.smoke_receipt,
            "Measured soak requires a complete same-boot mixed smoke receipt",
        )
        args.runtime = args.runtime.resolve(strict=True)
        result = asyncio.run(_main(args, *read_seals(args)))
    except (OSError, KeyError, TypeError, ValueError, httpx.HTTPError) as issue:
        parser.exit(
            1, f"Native lab could not finish ({type(issue).__name__}): {issue}\n"
        )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["evidence_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

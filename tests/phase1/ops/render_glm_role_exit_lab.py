# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Render one sealed native P/D role-exit arm from a qualified private pair."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import subprocess
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

from phase1_role_exit_instrument import MODULE, instrument

CASES = ("prefill_process_exit", "decode_process_exit")
BOOTSTRAP_ANCHOR = "\nstart_boot()\ntry:\n    decode = start_role"
BODY = r"""
role_exit = plan["role_exit_test"]
action = role_exit["action"]
case = role_exit["case"]
assert action in ("inject", "verify") and case in ("prefill_process_exit", "decode_process_exit")
root = plan_path.parent
start_boot()
result_directory = state / "role-exit"
if action == "inject":
    # Only in this ephemeral container, use the sealed instrument for the real
    # transport fences. The recovery arm keeps the installed wheel untouched.
    installed = Path("/tmp/kit/kitepond/integrations/mooncake/live_receive.py")
    original = installed.read_bytes()
    patched = (root / "launch" / "live_receive.role_exit.py").read_bytes()
    material = json.loads((root / "ROLE_EXIT_MATERIAL.json").read_bytes())
    assert hashlib.sha256(original).hexdigest() == material["original_sha256"]
    assert hashlib.sha256(patched).hexdigest() == material["instrumented_sha256"]
    installed.write_bytes(patched)
    gate = state / "role-exit-gate"
    gate.mkdir(mode=0o700)
    role_env["KITEPOND_GLM_ROLE_EXIT_GATE"] = str(gate)
    role_env["PYTHONPATH"] = "/tmp/kit:/paired/launch"
try:
    decode = start_role("decode", gpu_uuids)
    prefill = start_role("prefill", gpu_uuids)
    proof = observe(prefill.pid, decode.pid)
    proof.update({"plan_sha256": plan_digest, "engine_boot_id": boot,
                  "gpu_count": 8, "models_ready": True})
    emit("OWNERSHIP.json", proof)
    env = dict(os.environ) | role_env
    cache = state / "role-exit-cpu-cache"
    cache.mkdir(mode=0o700)
    env.update(CUDA_VISIBLE_DEVICES="", VLLM_TARGET_DEVICE="cpu",
               XDG_CACHE_HOME=str(cache), NUMBA_CACHE_DIR=str(cache),
               OPENBLAS_NUM_THREADS="1")
    command = [sys.executable, "-B", str(root / "launch" / "phase1_role_exit_probe.py"),
               action, "--case", case, "--manifest", str(root / "release" / "release.json"),
               "--pair", str(plan_path), "--prompt", str(root / "ROLE_EXIT_PROMPT.json"),
               "--runtime", str(state), "--output", str(result_directory)]
    if action == "verify":
        command.extend(("--prior", str(root / "prior")))
    with (state / "ROLE_EXIT.stderr").open("xb") as stderr:
        subprocess.run(command, env=env, stderr=stderr, check=True, timeout=150)
finally:
    close_boot()
    if action == "inject" and (result_directory / "result.json").is_file():
        teardown = command.copy()
        teardown[3] = "teardown"
        subprocess.run(teardown, env=env, check=True, timeout=20)
"""


def load(path):
    return json.loads(path.read_bytes())


def put(path, payload):
    if isinstance(payload, dict):
        payload = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if isinstance(payload, str):
        payload = payload.encode()
    temporary = path.with_suffix(path.suffix + ".new")
    with temporary.open("xb") as output:
        output.write(payload)
    os.replace(temporary, path)


def fresh_plan(path):
    text = path.read_text()
    plan = json.loads(text)
    domain = plan["mooncake_api"]
    active = {domain["d_boot_id"], domain["p_boot_id"]}
    assert plan["authority_boot_id"] == domain["d_boot_id"] and len(active) == 2
    for old in active:
        if not re.fullmatch(r"[0-9a-f]{32}", old) or text.count(old) < 2:
            raise ValueError("Expected explicit 32-digit sealed role epochs")
        new = secrets.token_hex(16)
        text = text.replace(old, new).replace(old[:12], new[:12])
    return json.loads(text)


def render(args):
    if (args.prior is not None) != (args.action == "verify"):
        raise ValueError(
            "Only the fresh recovery arm consumes the old verified evidence"
        )
    if args.source.resolve() == args.output.resolve():
        raise ValueError("Keep the released source immutable")
    subprocess.run(
        ["sha256sum", "-c", "--strict", "--quiet", "SHA256SUMS"],
        cwd=args.source,
        check=True,
    )
    prompt = load(args.prompt)
    if (
        type(prompt) is not list
        or not 1024 <= len(prompt) <= 4096
        or any(type(token) is not int or not 100 <= token < 32000 for token in prompt)
    ):
        raise ValueError("Expected a short synthetic raw ID prompt")
    if args.prior is not None:
        old, injection, teardown = (
            load(args.prior / name)
            for name in ("result.json", "injection.json", "teardown.json")
        )
        if (
            old["case"] != args.case
            or old["status"] != "injection_observed"
            or teardown["schema"] != "kitepond.glm-role-exit-teardown.v1"
            or len({old["case_id"], injection["case_id"], teardown["case_id"]}) != 1
            or teardown["old_result_sha256"]
            != sha256((args.prior / "result.json").read_bytes()).hexdigest()
            or old["injection_sha256"]
            != sha256((args.prior / "injection.json").read_bytes()).hexdigest()
        ):
            raise ValueError("Old pair exit and teardown must both be sealed")
    shutil.copytree(args.source, args.output, copy_function=os.link)
    plan = fresh_plan(args.output / "PLAN.json")
    plan["role_exit_test"] = {"action": args.action, "case": args.case}
    put(args.output / "PLAN.json", plan)
    put(args.output / "ROLE_EXIT_PROMPT.json", args.prompt.read_bytes())
    launch = args.output / "launch"
    entry = launch / "RUN_PAIR_INSIDE.py"
    source = entry.read_text()
    if source.count(BOOTSTRAP_ANCHOR) != 1:
        raise ValueError("Qualified private pair launcher changed its startup seam")
    rendered = source.split(BOOTSTRAP_ANCHOR, 1)[0] + "\n" + BODY
    compile(rendered, entry.name, "exec")
    put(entry, rendered)
    for name in (
        "phase1_role_exit_probe",
        "phase1_role_exit_gate",
        "phase1_fault_probe",
    ):
        library = Path(__file__).with_name(name + ".py")
        put(launch / library.name, library.read_bytes())
    if args.action == "inject":
        wheel = args.output / "release"
        wheel /= load(wheel / "release.json")["wheel"]["filename"]
        with ZipFile(wheel) as archive:
            original = archive.read(MODULE)
        patched = instrument(original)
        put(launch / "live_receive.role_exit.py", patched)
        put(
            args.output / "ROLE_EXIT_MATERIAL.json",
            {
                "schema": "kitepond.glm-role-exit-instrument.v1",
                "module": MODULE,
                "original_sha256": sha256(original).hexdigest(),
                "instrumented_sha256": sha256(patched).hexdigest(),
            },
        )
    else:
        prior = args.output / "prior"
        prior.mkdir(mode=0o700)
        for name in ("result.json", "injection.json", "teardown.json"):
            put(prior / name, (args.prior / name).read_bytes())
    files = sorted(
        path.relative_to(args.output)
        for path in args.output.rglob("*")
        if path.is_file() and path != args.output / "SHA256SUMS"
    )
    result = subprocess.check_output(
        ["sha256sum", "--", *map(str, files)], cwd=args.output
    )
    put(args.output / "SHA256SUMS", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--action", required=True, choices=("inject", "verify"))
    parser.add_argument("--case", required=True, choices=CASES)
    parser.add_argument("--prior", type=Path)
    render(parser.parse_args())


if __name__ == "__main__":
    main()

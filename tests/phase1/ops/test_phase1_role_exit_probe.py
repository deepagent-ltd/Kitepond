# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import os
import subprocess
import sys
from contextlib import ExitStack
from dataclasses import asdict, replace

import pytest
from kitepond.ops.wheel import sha256_file
from phase1_role_exit_gate import LinuxProcess, emit
from phase1_role_exit_probe import (
    CASES,
    await_exact_exit,
    candidate_group,
    finalize_teardown,
    kill_identity_group,
    teardown_context,
)

pytestmark = pytest.mark.skipif(
    not hasattr(os, "pidfd_open"), reason="Native role fault control requires Linux"
)


def _start_role(stack, runtime, role):
    command = [sys.executable, "-c", "from time import sleep; sleep(90)"]
    proc = subprocess.Popen(command, start_new_session=True)

    def close():
        if proc.poll() is None:
            proc.terminate()
        proc.wait(timeout=4)

    stack.callback(close)
    work = runtime / role
    work.mkdir()
    (work / "PID").write_text(str(proc.pid))
    return proc


@pytest.mark.parametrize("case", sorted(CASES))
def test_exact_role_exit_requires_old_namespace_teardown(tmp_path, case):
    runtime = tmp_path / "old"
    runtime.mkdir(mode=0o700)
    arm = runtime / "exit"
    arm.mkdir(mode=0o700)
    ownership = runtime / "OWNERSHIP.json"
    emit(ownership, {})
    target, survivor = CASES[case]
    with ExitStack() as stack:
        target_proc = _start_role(stack, runtime, target)
        peer_proc = _start_role(stack, runtime, survivor)
        leader, members = candidate_group(runtime, target, [target_proc.pid])
        other = LinuxProcess.read(peer_proc.pid)
        assert other is not None and other.live()
        assert len(members) == 1
        # A wrong start time cannot authorize use of an occupied PID.
        with pytest.raises(ValueError, match="group changed"):
            kill_identity_group(leader, (replace(leader, start_ticks=-1),))
        assert leader.live() and other.live()
        fired = kill_identity_group(leader, (replace(leader, state="R"),))
        ended = asyncio.run(await_exact_exit(leader, members))
        assert target_proc.wait(timeout=3) == -9
        assert other.live()
        assert ended >= fired
        epochs = dict.fromkeys((target, survivor), "old")
        context = teardown_context()
        landmarks = {
            "case_id": "synthetic-single-role",
            "ownership_sha256": sha256_file(ownership),
            "injection_context": context,
            "role_groups": {target: [asdict(leader)], survivor: [asdict(other)]},
        }
        emit(arm / "injection.json", landmarks)
        emit(
            arm / "result.json",
            {
                "status": "injection_observed",
                "case": case,
                "case_id": landmarks["case_id"],
                "artifact": "cpu-only",
                "before_role_epochs": epochs,
                "injection_sha256": sha256_file(arm / "injection.json"),
            },
        )
        with pytest.raises(ValueError, match="live or unverified"):
            finalize_teardown(runtime, arm)
        assert not (arm / "teardown.json").exists()
        peer_proc.terminate()
        peer_proc.wait(timeout=4)
        receipt = finalize_teardown(runtime, arm)
        assert receipt["closed_session_members"] == dict.fromkeys(epochs, 1)
        assert receipt["context"] == context
        assert receipt["epochs"] == epochs
        assert receipt["observed_monotonic_ns"] >= ended

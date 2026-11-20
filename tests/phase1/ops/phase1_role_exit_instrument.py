# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Instrument the exact released module only inside a sealed fault lab."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

MODULE = "kitepond/integrations/mooncake/live_receive.py"
ANCHORS = (
    (
        "            await _finish_before_cancel(asyncio.to_thread(scatter_and_fence))\n",
        "            from phase1_role_exit_gate import observe_fenced_window\n"
        "\n"
        "            await observe_fenced_window(attempt, window, remote_fenced, local_fenced)\n",
    ),
    (
        "                        self.pool.release(lease, writes_settled=settled)\n"
        "                    )\n"
        "            return WrittenRankWindow(\n",
        "                        self.pool.release(lease, writes_settled=settled)\n"
        "                    )\n"
        "                from phase1_role_exit_gate import source_event\n"
        "\n"
        "                source_event(self, window)\n"
        "            return WrittenRankWindow(\n",
    ),
    (
        "            self.release_source()\n            self._released = True\n",
        "            self.release_source()\n"
        "            self._released = True\n"
        "            from phase1_role_exit_gate import source_event\n"
        "\n"
        "            source_event(self)\n",
    ),
)


def instrument(source: bytes) -> bytes:
    text = source.decode()
    if any(text.count(anchor) != 1 for anchor, _ in ANCHORS):
        raise ValueError(
            "Released Mooncake module changed its fenced instrumentation seam"
        )
    text = text.replace(ANCHORS[0][0], ANCHORS[0][0] + ANCHORS[0][1], 1)
    for anchor, replacement in ANCHORS[1:]:
        text = text.replace(anchor, replacement, 1)
    compile(text, MODULE, "exec")
    return text.encode()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--wheel", required=True, type=Path)
    cli.add_argument("--output", required=True, type=Path)
    cli.add_argument("--module-sha256", required=True)
    args = cli.parse_args()
    with ZipFile(args.wheel) as archive:
        source = archive.read(MODULE)
    if sha256(source).hexdigest() != args.module_sha256:
        cli.error("Wheel module digest differs from the sealed dependency")
    payload = instrument(source)
    with args.output.open("xb") as output:
        output.write(payload)
    print(sha256(payload).hexdigest(), args.output.name)


if __name__ == "__main__":
    main()

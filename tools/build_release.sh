#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if [[ $# != 2 ]]; then
    printf 'Usage: %s VLLM_GIT_REPOSITORY OUTPUT_DIR\n' "$0" >&2
    exit 2
fi

root=$(cd "$(dirname "$0")/.." && pwd)
upstream=$(cd "$1" && pwd)
mkdir -p "$2"
output=$(cd "$2" && pwd)
scratch=$(mktemp -d)
trap 'rm -r -- "$scratch"' EXIT
mkdir "$scratch/source" "$scratch/sdist"

export SOURCE_DATE_EPOCH
SOURCE_DATE_EPOCH=$(git -C "$root" show -s --format=%ct HEAD)
export LC_ALL=C.UTF-8
git -C "$root" archive HEAD | tar -x -C "$scratch/source"
cd "$scratch/source"
patches=(compat/vllm-v034/*.patch)
target=$(sed -n '1s/^From \([0-9a-f]\{40\}\) .*/\1/p' "${patches[-1]}")
test -n "$target"

bash tools/stage_v034_runtime.sh "$upstream" "$scratch/runtime-sources"
python tools/seal_v034_runtime.py \
    --stock-manifest compat/vllm-v034/stock-runtime.json \
    --sources "$scratch/runtime-sources" --output src/kitepond/v034-runtime.zip \
    --target "$target"

release=$(python - <<'PY'
import tomllib
from pathlib import Path

project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
print(f"{project['name']}-{project['version']}")
PY
)
python -c 'import sys; from setuptools.build_meta import build_sdist; build_sdist(sys.argv[1])' "$scratch"
tar -xzf "$scratch/$release.tar.gz" -C "$scratch/sdist"

# Normalize the setuptools archive, including generated metadata timestamps.
tar --sort=name --mtime="@$SOURCE_DATE_EPOCH" --owner=0 --group=0 --numeric-owner \
    -C "$scratch/sdist" -cf - "$release" | gzip -n > "$output/$release.tar.gz"
cd "$scratch/sdist/$release"
python -c 'import sys; from setuptools.build_meta import build_wheel; build_wheel(sys.argv[1])' "$output"

cd "$output"
python - "$release-py3-none-any.whl" <<'PY'
import sys
from zipfile import ZipFile

with ZipFile(sys.argv[1]) as wheel:
    if not wheel.read("kitepond/v034-runtime.zip"):
        raise ValueError("The wheel must include the sealed vLLM runtime")
PY
sha256sum -- "$release-py3-none-any.whl" "$release.tar.gz" > SHA256SUMS

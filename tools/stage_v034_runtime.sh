#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if [[ $# != 2 ]]; then
    printf 'Usage: %s VLLM_GIT_REPOSITORY NEW_OUTPUT_DIR\n' "$0" >&2
    exit 2
fi

root=$(cd "$(dirname "$0")/.." && pwd)
repo=$(cd "$1" && pwd)
base=62d4bc3c7e33b1b6c906e233e6af2d3506c482ca
mkdir "$2"
output=$(cd "$2" && pwd)
staging=$(mktemp -d)
trap 'rm -r -- "$staging"' EXIT

git -C "$repo" archive "$base" vllm | tar -x -C "$staging"
for patch in "$root"/compat/vllm-v034/*.patch; do
    git -C "$staging" apply --include='vllm/**' "$patch"
    git -C "$staging" apply --numstat --include='vllm/**' "$patch" | cut -f3
done | sort -u | while IFS= read -r path; do
    [[ "$path" == vllm/*.py ]]
    install -D -m 0644 "$staging/$path" "$output/$path"
done

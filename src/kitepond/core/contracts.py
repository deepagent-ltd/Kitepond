# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass


@dataclass(frozen=True)
class Wait:
    """An operation that has not acquired a model lease or started native I/O."""

    reason: str

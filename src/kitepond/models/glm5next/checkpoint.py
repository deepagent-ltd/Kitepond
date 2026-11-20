# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact GLM target KDA checkpoint identity and boundary."""

import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class ConsumerCheckpointContract:
    """Operator-bound consumer metadata; never read from request HTTP fields."""

    model_fingerprint: str
    config_fingerprint: str
    hash_unit: int = 1024
    prefill_lookahead: int = 1
    finalized_margin: int = 0
    schema: str = "glm53f-native-consumer-prefix-v1"

    def __post_init__(self) -> None:
        if (
            self.schema != "glm53f-native-consumer-prefix-v1"
            or any(
                type(v) is not int
                for v in (self.hash_unit, self.prefill_lookahead, self.finalized_margin)
            )
            or self.hash_unit != 1024
            or self.prefill_lookahead != 1
            or self.finalized_margin != max(0, self.prefill_lookahead - 1)
            or any(
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
                for value in (self.model_fingerprint, self.config_fingerprint)
            )
        ):
            raise ValueError("Invalid locally bound GLM consumer checkpoint contract")

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(self.__dict__, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def checkpoint(self, valid_tokens: int) -> int:
        if type(valid_tokens) is not int or not 0 < valid_tokens < 1048576:
            raise ValueError("Invalid real producer valid-token boundary")
        # Native shared-prefix MTP leaves the final aligned block for replay.
        aligned = (valid_tokens - self.finalized_margin) // self.hash_unit
        return max(0, (aligned - 1) * self.hash_unit)


@dataclass(frozen=True)
class ProducerCheckpoint:
    request_id: str
    transfer_id: str
    epoch: int
    valid_tokens: int
    checkpoint_tokens: int
    contract_fingerprint: str
    group_blocks: tuple[tuple[int, int], ...]
    origin: str

    def validate(self, contract: ConsumerCheckpointContract) -> None:
        if (
            not self.request_id
            or not self.transfer_id
            or type(self.epoch) is not int
            or self.epoch < 0
            or self.valid_tokens <= 0
            or self.checkpoint_tokens <= 0
            or self.checkpoint_tokens != contract.checkpoint(self.valid_tokens)
            or self.contract_fingerprint != contract.fingerprint
            or self.origin not in {"cached_H_native_copy", "producer_H_COW_destination"}
            or not self.group_blocks
            or len({group for group, _ in self.group_blocks}) != len(self.group_blocks)
            or any(
                type(group) is not int
                or type(block) is not int
                or group < 0
                or block <= 0
                for group, block in self.group_blocks
            )
        ):
            raise ValueError("Invalid exact producer checkpoint metadata")

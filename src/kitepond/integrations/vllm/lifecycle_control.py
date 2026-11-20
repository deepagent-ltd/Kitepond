# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only lifecycle queries on the opted-in scheduler connector owner."""

from dataclasses import asdict
from threading import get_ident

from kitepond.core.policy import ColdJobScheduler, Mode

from .control_transport import ConnectorControlClient, ConnectorControlTransport
from .host_authority import NativeHostAuthority

LIFECYCLE_NAMESPACE = "kitepond.lifecycle.v1"
_QUERY = "kitepond.lifecycle-query.v1"
_RESULT = "kitepond.lifecycle-observation.v1"
_COMMON = {"schema", "engine_id", "boot_id"}
_FIELDS = {
    "health": set(),
    "ram_admission": {"reservation_tokens"},
    "cold_admission": {"mode"},
}
_REPLY_FIELDS = _COMMON | {"operation", "observations"}
_SUBJECTS = {
    "health": {"ram", "cold"},
    "ram_admission": {"ram"},
    "cold_admission": {"cold"},
}


def _validate_identity(engine_id: str, boot_id: str) -> None:
    if any(
        not isinstance(value, str)
        or not 0 < len(value) <= 256
        or value != value.strip()
        for value in (engine_id, boot_id)
    ):
        raise ValueError("Lifecycle query requires exact engine and boot identifiers")


class LifecycleControl:
    """Observe native state on the scheduler thread that owns the authorities."""

    def __init__(
        self,
        *,
        engine_id: str,
        boot_id: str,
        host: NativeHostAuthority | None = None,
        cold: ColdJobScheduler | None = None,
    ) -> None:
        _validate_identity(engine_id, boot_id)
        if host is not None and (host.engine_id, host.boot_id) != (engine_id, boot_id):
            raise ValueError("Host authority has a different engine or boot identity")
        self._owner = get_ident()
        self._identity = {"engine_id": engine_id, "boot_id": boot_id}
        self._host = host
        self._cold = cold

    @property
    def engine_id(self) -> str:
        return self._identity["engine_id"]

    def require_owner(self) -> None:
        if get_ident() != self._owner:
            raise RuntimeError(
                "Lifecycle observations require the scheduler owner thread"
            )

    def _ram(self, tokens: int | None = None) -> dict:
        if self._host is None:
            return {"status": "unavailable"}
        view = self._host.observe()
        report = {"status": "observed", **asdict(view)}
        if tokens is not None:
            needed = (tokens + view.block_tokens - 1) // view.block_tokens
            report.update(
                reservation_tokens=tokens,
                required_unshared_pages=needed,
                capacity="available" if needed <= view.available_pages else "wait",
            )
        return report

    def _lane(self, mode: Mode) -> dict:
        if self._cold is None:
            return {"mode": mode, "status": "unavailable"}
        current = self._cold.observe_lane(mode)
        return {
            "mode": mode,
            "status": "observed",
            "capacity": current.capacity,
            **asdict(current),
        }

    def handle(self, operation: str, payload: dict) -> dict:
        self.require_owner()
        if not isinstance(operation, str) or operation not in _FIELDS:
            raise ValueError("Unsupported Kitepond lifecycle query")
        if (
            not isinstance(payload, dict)
            or payload.keys() != _COMMON | _FIELDS[operation]
            or payload["schema"] != _QUERY
            or any(payload[name] != value for name, value in self._identity.items())
        ):
            raise ValueError("Lifecycle query schema or owner identity mismatch")

        if operation == "ram_admission":
            tokens = payload["reservation_tokens"]
            if type(tokens) is not int or not 0 < tokens < 2**63:
                raise ValueError("Reservation token count must be a positive int64")
            observations = {"ram": self._ram(tokens)}
        elif operation == "cold_admission":
            mode = payload["mode"]
            if mode not in ("RESTORE", "STORE"):
                raise ValueError("Unsupported cold I/O lane")
            observations = {"cold": self._lane(mode)}
        else:
            observations = {
                "ram": self._ram(),
                "cold": {mode: self._lane(mode) for mode in ("RESTORE", "STORE")},
            }
        return {
            "schema": _RESULT,
            "operation": operation,
            "observations": observations,
            **self._identity,
        }


class LifecycleQueryClient:
    """Read lifecycle observations through a single-core typed client."""

    def __init__(
        self, client: ConnectorControlClient, *, engine_id: str, boot_id: str
    ) -> None:
        _validate_identity(engine_id, boot_id)
        self._identity = {"engine_id": engine_id, "boot_id": boot_id}
        self._transport = ConnectorControlTransport(
            client, namespace=LIFECYCLE_NAMESPACE
        )

    async def _observe(self, operation: str, **fields) -> dict:
        result = await self._transport.call(
            operation, {"schema": _QUERY, **self._identity, **fields}
        )
        if (
            not isinstance(result, dict)
            or result.keys() != _REPLY_FIELDS
            or result["schema"] != _RESULT
            or result["operation"] != operation
            or any(result[name] != value for name, value in self._identity.items())
            or not isinstance(result["observations"], dict)
            or result["observations"].keys() != _SUBJECTS[operation]
        ):
            raise ValueError("Lifecycle observation schema or owner identity mismatch")
        return result

    async def health(self) -> dict:
        return await self._observe("health")

    async def ram_admission(self, reservation_tokens: int) -> dict:
        return await self._observe(
            "ram_admission", reservation_tokens=reservation_tokens
        )

    async def cold_admission(self, mode: Mode) -> dict:
        return await self._observe("cold_admission", mode=mode)

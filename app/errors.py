"""Domain errors mapped to stable HTTP responses by the API layer."""

from __future__ import annotations

from typing import Any


class ArrayError(Exception):
    """Base class for every expected storage-layer failure."""

    status_code = 500
    code = "array_error"

    def __init__(self, message: str, *, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.extra = detail or {}

    def as_detail(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, **self.extra}


class InvalidConfiguration(ArrayError):
    status_code = 422
    code = "invalid_configuration"


class InvalidBlockPayload(ArrayError):
    status_code = 422
    code = "invalid_block_payload"


class BlockOutOfRange(ArrayError):
    status_code = 404
    code = "block_out_of_range"


class DegradedReadOnly(ArrayError):
    status_code = 409
    code = "degraded_read_only"


class ServiceCrashed(ArrayError):
    status_code = 503
    code = "service_crashed"


class SimulatedPowerLoss(ArrayError):
    status_code = 503
    code = "simulated_power_loss"


class VolumeCorrupt(ArrayError):
    status_code = 500
    code = "volume_corrupt"

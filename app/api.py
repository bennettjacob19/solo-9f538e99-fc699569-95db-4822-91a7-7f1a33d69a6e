"""FastAPI surface for the XOR array drill.

Every operation is intentionally block oriented so the drill can be driven by
hand from ``/docs``:

* ``GET  /api/disks/{disk}/blocks/{index}``  -> read one 4 KiB block
* ``POST /api/disks/{disk}/blocks/{index}``  -> write one 4 KiB block
* ``POST /api/write``                        -> same, with a JSON body
* ``GET  /api/status`` / ``GET /api/journal`` -> inspect degraded + journal state

All storage calls are synchronous (``pread``/``pwrite``/``fsync``), so they are
dispatched to Starlette's thread pool.  That keeps ``fsync`` from stalling the
event loop and serializes correctly behind the array's own lock.
"""

from __future__ import annotations

import base64
import binascii
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, TypeVar

from fastapi import FastAPI, Query, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .array import POWER_LOSS_SCENARIOS, XorArray
from .config import BLOCK_SIZE, DISK_NAMES, Settings
from .errors import ArrayError, InvalidBlockPayload

T = TypeVar("T")

_write_description = (
    "Write one 4 KiB block. By default the write is acknowledged only after "
    "the data blocks, the parity block and the journal are all durable. Set "
    "`power_loss` to simulate a crash in the middle: `data` lands the two data "
    "blocks but leaves parity stale, `parity` lands only the parity block, "
    "`neither` persists only the journal, and `both` lands everything but never "
    "clears the journal. A simulated power loss stops the service; restart it "
    "to run recovery."
)


class WriteRequest(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"index": 0, "disk": "data0", "fill": 90, "power_loss": "none"},
                {
                    "index": 1,
                    "disk": "data1",
                    "data_b64": "<base64 of exactly 4096 bytes>",
                    "power_loss": "data",
                },
            ]
        }
    }

    index: int = Field(..., ge=0, description="Stripe index", examples=[0])
    disk: str = Field(
        ...,
        description="data0 or data1 (parity is derived)",
        examples=["data0"],
    )
    data_b64: str | None = Field(
        None,
        description="Base64 of exactly 4096 bytes",
        examples=["<base64 of exactly 4096 bytes>"],
    )
    data_hex: str | None = Field(
        None,
        description="Hex of exactly 4096 bytes (8192 hex characters)",
        examples=["<8192 hex characters>"],
    )
    fill: int | None = Field(
        None,
        ge=0,
        le=255,
        description="Alternative to data: fill every byte with this value",
        examples=[90],
    )
    power_loss: str = Field(
        "none",
        description=f"One of: {', '.join(POWER_LOSS_SCENARIOS)}",
        examples=["none"],
    )


def _decode_block(payload: WriteRequest) -> bytes:
    provided = [
        payload.data_b64 is not None,
        payload.data_hex is not None,
        payload.fill is not None,
    ]
    if sum(provided) != 1:
        raise InvalidBlockPayload("provide exactly one of data_b64, data_hex or fill")
    if payload.fill is not None:
        return bytes([payload.fill]) * BLOCK_SIZE
    try:
        if payload.data_b64 is not None:
            data = base64.b64decode(payload.data_b64, validate=True)
        else:
            data = bytes.fromhex(payload.data_hex or "")
    except (binascii.Error, ValueError) as exc:
        raise InvalidBlockPayload(f"could not decode block payload: {exc}") from exc
    if len(data) != BLOCK_SIZE:
        raise InvalidBlockPayload(
            f"a block must be exactly {BLOCK_SIZE} bytes",
            detail={"received_bytes": len(data), "expected_bytes": BLOCK_SIZE},
        )
    return data


def _parse_write(payload: Any) -> WriteRequest:
    try:
        return WriteRequest.model_validate(payload)
    except Exception as exc:  # pydantic ValidationError and JSON decode errors
        raise InvalidBlockPayload(f"invalid write payload: {exc}") from exc


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        # Construction performs disk setup + crash recovery; keep it off the
        # event loop as it may fsync every stripe of a rebuilt disk.
        app.state.array = await run_in_threadpool(XorArray, settings)
        yield

    app = FastAPI(
        title="XOR Array Drill",
        version="1.0.0",
        description=(
            "Three-disk XOR array (data0, data1, parity) with 4 KiB blocks, a "
            "single-slot redo journal and power-loss simulation. Restart with "
            "`OFFLINE_DISK=data0` or `OFFLINE_DISK=data1` for read-only "
            "degraded mode."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings

    @app.exception_handler(ArrayError)
    async def _array_error_handler(_: Request, exc: ArrayError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.as_detail()},
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Keep the drill's single error envelope even for FastAPI's own
        # body/path validation failures.
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "invalid_block_payload",
                    "message": "request validation failed",
                    "issues": jsonable_encoder(exc.errors()),
                }
            },
        )

    def arr(request: Request) -> XorArray:
        return request.app.state.array

    async def call(request: Request, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        return await run_in_threadpool(fn, *args, **kwargs)

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    # ------------------------------------------------------------------- ops

    @app.get("/healthz", tags=["ops"], responses={503: {"description": "Crashed"}})
    async def healthz(request: Request):
        array = arr(request)
        payload = {"ok": not array.crashed, "mode": array.mode}
        return JSONResponse(status_code=200 if payload["ok"] else 503, content=payload)

    @app.get("/api/status", tags=["ops"])
    async def status_(request: Request) -> dict[str, Any]:
        return arr(request).status()

    @app.get("/api/journal", tags=["ops"])
    async def journal(request: Request) -> dict[str, Any]:
        return arr(request).journal_info()

    @app.get("/api/verify", tags=["ops"])
    async def verify(request: Request) -> dict[str, Any]:
        return await call(request, arr(request).verify)

    # ---------------------------------------------------------------- blocks

    @app.get("/api/disks", tags=["blocks"])
    async def disks() -> dict[str, Any]:
        return {
            "disks": list(DISK_NAMES),
            "block_size": BLOCK_SIZE,
            "note": "parity is derived from data0 XOR data1 and is read-only",
        }

    @app.get("/api/disks/{disk}/blocks", tags=["blocks"])
    async def list_blocks(
        request: Request,
        disk: str,
        include_data: bool = Query(False, description="Include base64 payloads"),
    ) -> dict[str, Any]:
        array = arr(request)
        return await call(request, array.list_blocks, disk, include_data=include_data)

    @app.get("/api/disks/{disk}/blocks/{index}", tags=["blocks"])
    async def read_block(request: Request, disk: str, index: int) -> dict[str, Any]:
        return await call(request, arr(request).read_block, index, disk)

    @app.get("/api/disks/{disk}/blocks/{index}/raw", tags=["blocks"])
    async def read_block_raw(request: Request, disk: str, index: int) -> Response:
        result = await call(request, arr(request).read_block, index, disk)
        return Response(
            content=base64.b64decode(result["data_b64"]),
            media_type="application/octet-stream",
        )

    @app.get("/api/blocks/{index}", tags=["blocks"])
    async def read_block_query(
        request: Request,
        index: int,
        disk: str = Query(..., description="data0, data1 or parity"),
    ) -> dict[str, Any]:
        return await call(request, arr(request).read_block, index, disk)

    @app.api_route(
        "/api/disks/{disk}/blocks/{index}",
        methods=["POST", "PUT"],
        tags=["blocks"],
        description=_write_description,
        responses={409: {"description": "Degraded read-only mode"}},
    )
    async def write_block(
        request: Request,
        disk: str,
        index: int,
        power_loss: str | None = Query(
            None, description=f"One of: {', '.join(POWER_LOSS_SCENARIOS)}"
        ),
    ) -> dict[str, Any]:
        array = arr(request)
        content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if content_type in {"application/octet-stream", "binary/octet-stream"}:
            raw = await request.body()
            if len(raw) != BLOCK_SIZE:
                raise InvalidBlockPayload(
                    f"a block must be exactly {BLOCK_SIZE} bytes",
                    detail={"received_bytes": len(raw), "expected_bytes": BLOCK_SIZE},
                )
            return await call(
                request, array.write_block, index, disk, raw, power_loss=power_loss
            )

        try:
            body = await request.json()
        except Exception as exc:
            raise InvalidBlockPayload(f"invalid write payload: {exc}") from exc
        payload = _parse_write(body)
        if payload.index != index:
            raise InvalidBlockPayload(
                "path index and body index disagree",
                detail={"path_index": index, "body_index": payload.index},
            )
        # Compare canonical names so aliases like "0" and "data0" agree.
        if arr(request).validate_disk_name(payload.disk) != arr(request).validate_disk_name(disk):
            raise InvalidBlockPayload(
                "path disk and body disk disagree",
                detail={"path_disk": disk, "body_disk": payload.disk},
            )
        return await call(
            request,
            array.write_block,
            index,
            disk,
            _decode_block(payload),
            power_loss=power_loss or payload.power_loss,
        )

    @app.post("/api/write", tags=["blocks"], description=_write_description)
    async def write_any(request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
        except Exception as exc:
            raise InvalidBlockPayload(f"invalid write payload: {exc}") from exc
        payload = _parse_write(body)
        return await call(
            request,
            arr(request).write_block,
            payload.index,
            payload.disk,
            _decode_block(payload),
            power_loss=payload.power_loss,
        )

    return app


app = create_app()

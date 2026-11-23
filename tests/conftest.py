from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.config import BLOCK_SIZE, Settings


def block(fill: int) -> bytes:
    return bytes([fill]) * BLOCK_SIZE


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def make_settings(tmp_path, *, stripes: int = 4, offline: str | None = None) -> Settings:
    return Settings(data_dir=tmp_path, stripes=stripes, offline_disk=offline)


@pytest.fixture()
def client(tmp_path):
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)) as c:
        c.settings = settings
        yield c


def write(client: TestClient, index: int, disk: str, data: bytes, **kwargs):
    return client.post(
        f"/api/disks/{disk}/blocks/{index}",
        json={"index": index, "disk": disk, "data_b64": b64(data), **kwargs},
    )


def read(client: TestClient, index: int, disk: str) -> dict:
    response = client.get(f"/api/disks/{disk}/blocks/{index}")
    assert response.status_code == 200, response.text
    return response.json()


def read_bytes(client: TestClient, index: int, disk: str) -> bytes:
    return base64.b64decode(read(client, index, disk)["data_b64"])

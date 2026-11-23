from __future__ import annotations

import base64

from fastapi.testclient import TestClient

from app.api import create_app

from conftest import b64, block, make_settings, read, read_bytes, write


def test_healthy_write_reads_all_three_disks(client):
    payload = block(0xA5)
    response = write(client, 1, "data0", payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["journal_cleared"] is True
    assert body["power_loss"] == "none"

    assert read_bytes(client, 1, "data0") == payload
    assert read_bytes(client, 1, "data1") == block(0)
    assert read_bytes(client, 1, "parity") == payload

    assert read(client, 1, "parity")["consistent"] is True
    status = client.get("/api/status").json()
    assert status["mode"] == "healthy"
    assert status["journal"]["state"] == "clear"

    verify = client.get("/api/verify").json()
    assert verify["consistent"] is True
    assert verify["checked"] is True


def test_xor_parity_across_two_data_writes(client):
    a, b = block(0x11), block(0x22)
    assert write(client, 0, "data0", a).status_code == 200
    assert write(client, 0, "data1", b).status_code == 200
    expected = bytes(x ^ y for x, y in zip(a, b))
    assert read_bytes(client, 0, "parity") == expected
    result = read(client, 0, "data0")
    assert result["consistent"] is True


def test_write_rejects_wrong_size_and_parity_target(client):
    short = client.post(
        "/api/write",
        json={"index": 0, "disk": "data0", "data_b64": b64(b"too short")},
    )
    assert short.status_code == 422
    assert short.json()["error"]["code"] == "invalid_block_payload"

    parity = write(client, 0, "parity", block(1))
    assert parity.status_code == 422
    assert parity.json()["error"]["code"] == "invalid_configuration"


def test_raw_octet_stream_write(client):
    response = client.post(
        "/api/disks/data1/blocks/2",
        content=block(0x7F),
        headers={"content-type": "application/octet-stream"},
    )
    assert response.status_code == 200, response.text
    assert read_bytes(client, 2, "data1") == block(0x7F)


def test_out_of_range_index(client):
    response = client.get("/api/disks/data0/blocks/999")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "block_out_of_range"


def test_docs_and_openapi_expose_block_endpoints(client):
    docs = client.get("/docs")
    assert docs.status_code == 200
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    assert "/api/disks/{disk}/blocks/{index}" in paths
    assert {"get", "post"} <= set(paths["/api/disks/{disk}/blocks/{index}"])
    assert "/api/write" in paths
    params = {
        p["name"] for p in paths["/api/disks/{disk}/blocks/{index}"]["post"]["parameters"]
    }
    assert "power_loss" in params
    assert "/api/status" in paths and "/api/journal" in paths


def test_disk_aliases_and_path_body_agreement(client):
    via_alias = client.post(
        "/api/disks/0/blocks/1",
        json={"index": 1, "disk": "data0", "fill": 5},
    )
    assert via_alias.status_code == 200, via_alias.text
    assert read_bytes(client, 1, "data0") == block(5)

    mismatch = client.post(
        "/api/disks/data0/blocks/1",
        json={"index": 1, "disk": "data1", "fill": 5},
    )
    assert mismatch.status_code == 422
    assert mismatch.json()["error"]["code"] == "invalid_block_payload"

    wrong_index = client.post(
        "/api/disks/data0/blocks/1",
        json={"index": 2, "disk": "data0", "fill": 5},
    )
    assert wrong_index.status_code == 422


def test_healthz_reflects_crash(client):
    assert client.get("/healthz").status_code == 200
    assert client.get("/healthz").json() == {"ok": True, "mode": "healthy"}
    write(client, 0, "data0", block(1))
    crash = write(client, 0, "data0", block(2), power_loss="data")
    assert crash.status_code == 503
    health = client.get("/healthz")
    assert health.status_code == 503
    assert health.json() == {"ok": False, "mode": "crashed"}


def test_validation_errors_use_the_shared_envelope(client):
    missing = client.post("/api/write", json={"index": 0, "data_b64": b64(block(1))})
    assert missing.status_code == 422
    assert missing.json()["error"]["code"] == "invalid_block_payload"

    bad_fill = client.post(
        "/api/write", json={"index": 0, "disk": "data0", "fill": 999}
    )
    assert bad_fill.status_code == 422
    assert bad_fill.json()["error"]["code"] == "invalid_block_payload"

    negative = client.post(
        "/api/write", json={"index": -1, "disk": "data0", "fill": 1}
    )
    assert negative.status_code == 422
    assert negative.json()["error"]["code"] == "invalid_block_payload"


def test_degraded_write_rejects_before_touching_disk(tmp_path):
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)) as c:
        write(c, 0, "data0", block(3))

    from conftest import make_settings as ms

    with TestClient(create_app(ms(tmp_path, offline="data0"))) as c:
        refused = write(c, 0, "data1", block(4))
        assert refused.status_code == 409
        assert refused.json()["error"]["offline_disk"] == "data0"
        # the rejected write must not have changed data1
        assert read_bytes(c, 0, "data1") == block(0)


def test_fill_helper_and_block_listing(client):
    response = client.post(
        "/api/write", json={"index": 3, "disk": "data0", "fill": 9}
    )
    assert response.status_code == 200
    listing = client.get("/api/disks/parity/blocks?include_data=true").json()
    assert listing["disk"] == "parity"
    assert len(listing["blocks"]) == 4
    entry = listing["blocks"][3]
    assert entry["index"] == 3
    assert base64.b64decode(entry["data_b64"]) == block(9)
    assert entry["source"] == "native"
    assert entry["consistent"] is True

    sha_only = client.get("/api/disks/data0/blocks").json()
    assert "data_b64" not in sha_only["blocks"][0]
    assert sha_only["blocks"][3]["sha256"] == entry["sha256"]


def test_degraded_mode_is_read_only_and_reconstructs(tmp_path):
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)) as c:
        assert write(c, 0, "data0", block(0x31)).status_code == 200
        assert write(c, 1, "data0", block(0x32)).status_code == 200
        assert write(c, 1, "data1", block(0x44)).status_code == 200
        before = {i: read_bytes(c, i, "data0") for i in range(2)}
        before1 = {i: read_bytes(c, i, "data1") for i in range(2)}

    degraded = make_settings(tmp_path, offline="data0")
    with TestClient(create_app(degraded)) as c:
        status = c.get("/api/status").json()
        assert status["mode"] == "degraded"
        assert status["writable"] is False
        assert status["offline_disk"] == "data0"

        reconstructed = read(c, 0, "data0")
        assert reconstructed["source"] == "reconstructed"
        assert base64.b64decode(reconstructed["data_b64"]) == before[0]
        assert read_bytes(c, 1, "data0") == before[1]
        # online disks still read natively
        assert read(c, 1, "data1")["source"] == "native"
        assert read_bytes(c, 1, "data1") == before1[1]

        blocked = write(c, 2, "data1", block(0x99))
        assert blocked.status_code == 409
        assert blocked.json()["error"]["code"] == "degraded_read_only"
        verify = c.get("/api/verify").json()
        assert verify["checked"] is False

    # Online re-read after the offline drill shows the same data.
    with TestClient(create_app(make_settings(tmp_path))) as c:
        assert read_bytes(c, 0, "data0") == before[0]
        assert read_bytes(c, 1, "data0") == before[1]
        assert c.get("/api/verify").json()["consistent"] is True


def test_degraded_repeated_restart_is_stable(tmp_path):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        write(c, 0, "data0", block(1))
        write(c, 0, "data1", block(2))

    expected = {}
    for disk, idx in (("data0", 0), ("data1", 0)):
        with TestClient(create_app(make_settings(tmp_path, offline="data1"))) as c:
            expected[(disk, idx)] = read_bytes(c, idx, disk)
    # two consecutive starts with the same offline disk
    for _ in range(2):
        with TestClient(create_app(make_settings(tmp_path, offline="data1"))) as c:
            for (disk, idx), value in expected.items():
                assert read_bytes(c, idx, disk) == value


def test_write_power_loss_requires_restart_and_recovery(tmp_path):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        write(c, 0, "data0", block(0x10))
        write(c, 0, "data1", block(0x20))
        old0, old1 = read_bytes(c, 0, "data0"), read_bytes(c, 0, "data1")

        crashed = write(c, 0, "data0", block(0x77), power_loss="parity")
        assert crashed.status_code == 503
        detail = crashed.json()["error"]
        assert detail["code"] == "simulated_power_loss"
        assert detail["landed"] == ["parity"]

        # service refuses further work until restarted
        assert c.get("/api/status").status_code == 200
        blocked = c.get("/api/disks/data0/blocks/0")
        assert blocked.status_code == 503
        assert blocked.json()["error"]["code"] == "service_crashed"
        blocked_write = write(c, 0, "data1", block(0x01))
        assert blocked_write.status_code == 503

    with TestClient(create_app(make_settings(tmp_path))) as c:
        status = c.get("/api/status").json()
        assert status["mode"] == "healthy"
        assert status["last_recovery"] is not None
        assert status["last_recovery"]["recovered"] is True
        assert status["journal"]["state"] == "clear"
        assert read_bytes(c, 0, "data0") == block(0x77)
        # data1 kept its confirmed value, so parity was rebuilt from both
        assert read_bytes(c, 0, "data1") == old1
        assert read(c, 0, "parity")["consistent"] is True
        assert c.get("/api/verify").json()["consistent"] is True
        assert (old0, old1) != (block(0x77), old1)

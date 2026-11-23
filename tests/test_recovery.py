from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.config import BLOCK_SIZE
from app.errors import VolumeCorrupt
from app.journal import HEADER_SIZE, PAYLOAD_SIZE, TOTAL_SIZE

from conftest import block, make_settings, read, read_bytes, write


def _snapshot(client: TestClient, stripes: int, disks=("data0", "data1", "parity")):
    return {
        (disk, index): read_bytes(client, index, disk)
        for index in range(stripes)
        for disk in disks
    }


def _seed(client: TestClient, stripes: int) -> None:
    for index in range(stripes):
        assert write(client, index, "data0", block(0x10 + index)).status_code == 200
        assert write(client, index, "data1", block(0xA0 + index)).status_code == 200


def _crash(tmp_path, scenario: str, index: int = 1, fill: int = 0x5E) -> None:
    with TestClient(create_app(make_settings(tmp_path))) as c:
        _seed(c, 4)
        response = write(c, index, "data0", block(fill), power_loss=scenario)
        assert response.status_code == 503, response.text
        assert response.json()["error"]["code"] == "simulated_power_loss"


def _assert_consistent(client: TestClient, stripes: int) -> None:
    result = client.get("/api/verify").json()
    assert result["consistent"] is True
    assert all(item["consistent"] for item in result["stripes"])


def _assert_untouched(client: TestClient, before: dict, stripes: int, skip: int) -> None:
    for index in range(stripes):
        if index == skip:
            continue
        for disk in ("data0", "data1", "parity"):
            assert read_bytes(client, index, disk) == before[(disk, index)], (
                disk,
                index,
            )


@pytest.mark.parametrize("scenario", ["data", "data0", "data1", "parity", "neither", "both"])
def test_recovery_rolls_forward_interrupted_stripe(tmp_path, scenario):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        _seed(c, 4)
        before = _snapshot(c, 4)
        new_value = block(0x5E)

    with TestClient(create_app(make_settings(tmp_path))) as c:
        response = write(c, 1, "data0", new_value, power_loss=scenario)
        assert response.status_code == 503
        assert response.json()["error"]["restart_required"] is True

    with TestClient(create_app(make_settings(tmp_path))) as c:
        status = c.get("/api/status").json()
        assert status["mode"] == "healthy"
        assert status["last_recovery"]["recovered"] is True
        assert status["last_recovery"]["stripe"] == 1
        assert status["journal"]["state"] == "clear"
        assert read_bytes(c, 1, "data0") == new_value
        # the untouched data block of the same stripe keeps its confirmed value
        assert read_bytes(c, 1, "data1") == before[("data1", 1)]
        _assert_consistent(c, 4)
        _assert_untouched(c, before, 4, skip=1)


def test_crash_keeps_confirmed_contents_elsewhere(tmp_path):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        _seed(c, 4)
        confirmed = _snapshot(c, 4)

    with TestClient(create_app(make_settings(tmp_path))) as c:
        write(c, 2, "data0", block(0xFF), power_loss="data")

    with TestClient(create_app(make_settings(tmp_path))) as c:
        for index in range(4):
            if index == 2:
                continue
            for disk in ("data0", "data1", "parity"):
                assert read_bytes(c, index, disk) == confirmed[(disk, index)]
        assert read_bytes(c, 2, "data1") == confirmed[("data1", 2)]
        _assert_consistent(c, 4)


def test_both_landed_but_journal_not_cleared_is_idempotent(tmp_path):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        _seed(c, 4)
    _crash(tmp_path, "both", index=3, fill=0x44)
    first = None
    for _ in range(2):
        with TestClient(create_app(make_settings(tmp_path))) as c:
            current = _snapshot(c, 4)
            assert read_bytes(c, 3, "data0") == block(0x44)
            if first is not None:
                assert current == first
            first = current
            assert c.get("/api/status").json()["journal"]["state"] == "clear"


def test_recovery_with_offline_disk_is_repeatable(tmp_path):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        _seed(c, 4)
        confirmed = _snapshot(c, 4)

    # Data blocks landed, parity stale, then the machine died.
    with TestClient(create_app(make_settings(tmp_path))) as c:
        write(c, 1, "data0", block(0x6B), power_loss="data")

    offline = make_settings(tmp_path, offline="data0")
    snapshots = []
    for _ in range(2):
        with TestClient(create_app(offline)) as c:
            status = c.get("/api/status").json()
            assert status["mode"] == "degraded"
            assert status["journal"]["state"] == "prepared"
            assert status["last_recovery"]["journal_cleared"] is False
            view = _snapshot(c, 4)
            # logical view is rolled forward, including the offline disk
            assert view[("data0", 1)] == block(0x6B)
            assert view[("data0", 0)] == confirmed[("data0", 0)]
            snapshots.append(view)
            # reads of the offline disk are reconstructed
            assert read(c, 1, "data0")["source"] == "reconstructed"
            # read-only
            assert write(c, 1, "data1", block(0x01)).status_code == 409
    assert snapshots[0] == snapshots[1]

    # A later start without the offline flag finishes recovery.
    with TestClient(create_app(make_settings(tmp_path))) as c:
        assert c.get("/api/status").json()["journal"]["state"] == "clear"
        assert read_bytes(c, 1, "data0") == block(0x6B)
        assert _snapshot(c, 4) == snapshots[0]
        _assert_consistent(c, 4)


def test_offline_disk_never_read_from_its_file(tmp_path):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        _seed(c, 4)

    # Sabotage the physical file of the disk we are about to take offline.
    # A degraded read must reconstruct from the other data disk + parity.
    with open(tmp_path / "data0.img", "r+b") as handle:
        handle.seek(BLOCK_SIZE)
        handle.write(b"\xff" * BLOCK_SIZE)

    with TestClient(create_app(make_settings(tmp_path, offline="data0"))) as c:
        for index in range(4):
            value = read_bytes(c, index, "data0")
            assert value == block(0x10 + index)
            assert read(c, index, "data0")["source"] == "reconstructed"
        assert c.get("/api/verify").json()["checked"] is False


def test_torn_journal_header_means_nothing_landed(tmp_path):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        _seed(c, 4)
        confirmed = _snapshot(c, 4)

    # Crash before the journal intent became durable: no disk byte may change.
    with TestClient(create_app(make_settings(tmp_path))) as c:
        write(c, 1, "data0", block(0x77), power_loss="neither")

    journal = tmp_path / "journal.img"
    with open(journal, "r+b") as handle:
        handle.seek(0)
        handle.write(b"\x00" * HEADER_SIZE)

    with TestClient(create_app(make_settings(tmp_path))) as c:
        info = c.get("/api/journal").json()
        assert info["state"] in {"empty", "invalid_header", "clear"}
        # The interrupted block is allowed to be old *or* new. Here it is old.
        assert _snapshot(c, 4) == confirmed
        _assert_consistent(c, 4)
        status = c.get("/api/status").json()
        assert status["last_recovery"] is None


def test_corrupt_journal_payload_refuses_to_start(tmp_path):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        _seed(c, 4)
    with TestClient(create_app(make_settings(tmp_path))) as c:
        write(c, 1, "data0", block(0x33), power_loss="neither")

    journal = tmp_path / "journal.img"
    with open(journal, "r+b") as handle:
        handle.seek(HEADER_SIZE + 10)
        original = handle.read(1)
        handle.seek(HEADER_SIZE + 10)
        handle.write(bytes([original[0] ^ 0xFF]))

    with pytest.raises(VolumeCorrupt):
        with TestClient(create_app(make_settings(tmp_path))) as c:
            c.get("/api/status")


def test_journal_footprint_is_two_blocks_plus_metadata(tmp_path):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        _seed(c, 4)
        info = c.get("/api/journal").json()

    journal = tmp_path / "journal.img"
    assert journal.exists()
    assert journal.stat().st_size == TOTAL_SIZE
    assert TOTAL_SIZE == HEADER_SIZE + 2 * BLOCK_SIZE
    assert PAYLOAD_SIZE == 2 * BLOCK_SIZE
    assert HEADER_SIZE <= 512
    # The journal is a separate file; the disk images hold payload only.
    for name in ("data0", "data1", "parity"):
        image = tmp_path / f"{name}.img"
        assert image.stat().st_size == 4 * BLOCK_SIZE
        assert image != journal
    assert info["payload_blocks"] == 2
    assert info["max_bytes"] == journal.stat().st_size


def test_missing_disk_image_is_rebuilt_when_starting_healthy(tmp_path):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        _seed(c, 4)
        expected = _snapshot(c, 4)

    (tmp_path / "data1.img").unlink()

    with TestClient(create_app(make_settings(tmp_path))) as c:
        status = c.get("/api/status").json()
        assert status["rebuilt_disks"] == ["data1"]
        assert _snapshot(c, 4) == expected
        _assert_consistent(c, 4)


def test_missing_disk_can_be_started_offline(tmp_path):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        _seed(c, 4)
        expected = _snapshot(c, 4)

    (tmp_path / "data0.img").unlink()

    with TestClient(create_app(make_settings(tmp_path, offline="data0"))) as c:
        status = c.get("/api/status").json()
        assert status["mode"] == "degraded"
        assert status["rebuilt_disks"] == []
        for index in range(4):
            assert read_bytes(c, index, "data0") == expected[("data0", index)]
        assert not (tmp_path / "data0.img").exists()


def test_concurrent_writes_keep_every_stripe_consistent(tmp_path):
    import threading

    with TestClient(create_app(make_settings(tmp_path))) as c:
        # TestClient is thread safe enough for this coarse serialization check.
        errors = []

        def worker(offset: int) -> None:
            try:
                for index in range(4):
                    write(c, index, "data0" if offset % 2 else "data1", block(offset + index))
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(n * 10,)) for n in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors
        _assert_consistent(c, 4)


def test_journal_unknown_state_refuses_to_start(tmp_path):
    from app.journal import STATE_PREPARED, _pack_header

    with TestClient(create_app(make_settings(tmp_path))) as c:
        _seed(c, 4)
    with TestClient(create_app(make_settings(tmp_path))) as c:
        write(c, 1, "data0", block(0x21), power_loss="neither")

    # Forge a CRC-valid header with an unknown state byte.
    header = _pack_header(
        state=STATE_PREPARED + 99,
        flags=0,
        stripe=1,
        data_len=BLOCK_SIZE,
        seq=9,
        created_ns=1,
        data0_sha=b"\x00" * 32,
        data1_sha=b"\x00" * 32,
    )
    with open(tmp_path / "journal.img", "r+b") as handle:
        handle.write(header)

    with pytest.raises(VolumeCorrupt):
        with TestClient(create_app(make_settings(tmp_path))) as c:
            c.get("/api/status")


def test_journal_stripe_outside_array_refuses_to_start(tmp_path):
    from app.journal import Journal, new_record

    journal = Journal(tmp_path / "journal.img")
    journal.write_prepared(
        new_record(stripe=9, seq=1, data0=block(1), data1=block(2))
    )
    with pytest.raises(VolumeCorrupt):
        with TestClient(create_app(make_settings(tmp_path))) as c:
            c.get("/api/status")

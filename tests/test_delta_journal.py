"""Tests for the small-write delta journal and its recovery guarantees.

Covers:

* writes changing <= 64 bytes are journaled as delta records whose prepared
  journal write stays under 5 KiB including the header;
* the neither / data / parity power-loss breakpoints all roll forward;
* reads stay correct with either data disk offline, across repeated offline
  restarts and after the disk comes back online;
* replaying one delta record reads at most two member-disk blocks;
* larger writes keep using full-stripe records;
* legacy v1 full records on old volumes still replay;
* all other stripes keep their last confirmed contents.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.config import BLOCK_SIZE
from app.journal import (
    HEADER_SIZE,
    JOURNAL_PREPARE_BUDGET,
    SMALL_WRITE_LIMIT,
    DeltaEntry,
    Journal,
    apply_entries,
    new_record,
)

from conftest import block, make_settings, read, read_bytes, write


def _base_block(seed: int = 0x30) -> bytearray:
    # A high-entropy base so every stripe's blocks differ byte by byte.
    return bytearray((seed * 31 + i) & 0xFF for i in range(BLOCK_SIZE))


def _patch(base: bytes, changes: dict[int, int]) -> bytes:
    buf = bytearray(base)
    for offset, value in changes.items():
        buf[offset] = value
    return bytes(buf)


def _changes(n: int, *, start: int = 100) -> dict[int, int]:
    return {start + 7 * i: 0x10 + i for i in range(n)}


def _guaranteed_changes(base: bytes, n: int, *, start: int = 100) -> dict[int, int]:
    """Build ``n`` replacements that provably differ from ``base``."""
    out: dict[int, int] = {}
    i = 0
    while len(out) < n:
        offset = start + 7 * i
        value = (base[offset] ^ 0xA5 ^ i) & 0xFF
        if value != base[offset] and offset not in out:
            out[offset] = value
        i += 1
    return out


def _prepare_stripe(tmp_path, index: int, *, fill: int = 0x40) -> tuple[bytes, bytes]:
    """Write one fully populated stripe and return the two confirmed blocks."""
    with TestClient(create_app(make_settings(tmp_path))) as c:
        d0 = _base_block(fill)
        d1 = _base_block(fill + 1)
        assert write(c, index, "data0", bytes(d0)).status_code == 200
        assert write(c, index, "data1", bytes(d1)).status_code == 200
    return bytes(d0), bytes(d1)


def _full_snapshot(client: TestClient, stripes: int) -> dict:
    return {
        (disk, idx): read_bytes(client, idx, disk)
        for idx in range(stripes)
        for disk in ("data0", "data1", "parity")
    }


class ReadCounter:
    """Count member-disk block reads performed while inside ``with``."""

    def __init__(self, array) -> None:
        self.array = array
        self.count = 0
        self._original = array._read_raw

    def __enter__(self) -> "ReadCounter":
        original = self._original

        def counting(name: str, index: int) -> bytes:
            self.count += 1
            return original(name, index)

        self.array._read_raw = counting
        return self

    def __exit__(self, *exc) -> None:
        self.array._read_raw = self._original


def test_small_write_uses_delta_under_five_kib(tmp_path):
    d0, d1 = _prepare_stripe(tmp_path, 1)
    changes = _guaranteed_changes(d0, 10)
    new_d0 = _patch(d0, changes)

    with TestClient(create_app(make_settings(tmp_path))) as c:
        response = write(c, 1, "data0", new_d0)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["journal_record_kind"] == "delta"
        assert body["changed_bytes"] == len(changes)
        assert body["journal_prepared_bytes"] == HEADER_SIZE + len(changes) * 4
        assert body["journal_prepared_bytes"] < JOURNAL_PREPARE_BUDGET
        assert body["journal_prepared_bytes"] <= 5 * 1024

        info = c.get("/api/journal").json()
        assert info["state"] == "clear"
        assert info["small_write_limit"] == SMALL_WRITE_LIMIT
        assert info["prepare_budget_bytes"] == JOURNAL_PREPARE_BUDGET

        assert read_bytes(c, 1, "data0") == new_d0
        assert read_bytes(c, 1, "data1") == d1
        assert read(c, 1, "parity")["consistent"] is True

    # The journal image itself stays preallocated for both record formats.
    assert (tmp_path / "journal.img").stat().st_size == HEADER_SIZE + 2 * BLOCK_SIZE


@pytest.mark.parametrize("changed", [1, 32, 64])
def test_delta_budget_holds_up_to_64_bytes(tmp_path, changed):
    d0, _ = _prepare_stripe(tmp_path, 0)
    new_d0 = _patch(d0, _guaranteed_changes(d0, changed))
    with TestClient(create_app(make_settings(tmp_path))) as c:
        body = write(c, 0, "data0", new_d0).json()
        assert body["journal_record_kind"] == "delta"
        assert body["changed_bytes"] == changed
        assert body["journal_prepared_bytes"] < 5 * 1024


def test_more_than_64_changed_bytes_falls_back_to_full_record(tmp_path):
    d0, _ = _prepare_stripe(tmp_path, 0)
    new_d0 = _patch(d0, _guaranteed_changes(d0, 65))
    with TestClient(create_app(make_settings(tmp_path))) as c:
        body = write(c, 0, "data0", new_d0).json()
        assert body["journal_record_kind"] == "full"
        assert body["changed_bytes"] == 65
        assert body["journal_prepared_bytes"] == HEADER_SIZE + 2 * BLOCK_SIZE
        assert read_bytes(c, 0, "data0") == new_d0
        assert read(c, 0, "parity")["consistent"] is True


@pytest.mark.parametrize("scenario", ["neither", "data", "parity"])
def test_delta_recovers_from_each_breakpoint(tmp_path, scenario):
    stripes = 4
    d0, d1 = _prepare_stripe(tmp_path, 2)
    # Populate every other stripe too, to prove they stay confirmed.
    with TestClient(create_app(make_settings(tmp_path))) as c:
        for idx in range(stripes):
            if idx == 2:
                continue
            write(c, idx, "data0", bytes(_base_block(0x50 + idx)))
            write(c, idx, "data1", bytes(_base_block(0x60 + idx)))
        confirmed = _full_snapshot(c, stripes)

    changes = _guaranteed_changes(d0, 12, start=333)
    new_d0 = _patch(d0, changes)
    with TestClient(create_app(make_settings(tmp_path))) as c:
        response = write(c, 2, "data0", new_d0, power_loss=scenario)
        assert response.status_code == 503, response.text
        detail = response.json()["error"]
        assert detail["journal_record_kind"] == "delta"
        assert detail["journal_prepared_bytes"] < 5 * 1024
        info = c.get("/api/journal").json()
        assert info["state"] == "prepared"
        assert info["record_kind"] == "delta"
        assert info["delta_entries"] == len(changes)

    with TestClient(create_app(make_settings(tmp_path))) as c:
        status = c.get("/api/status").json()
        assert status["mode"] == "healthy"
        recovery = status["last_recovery"]
        assert recovery["recovered"] is True
        assert recovery["record_kind"] == "delta"
        assert recovery["member_block_reads"] <= 2
        assert status["journal"]["state"] == "clear"
        assert read_bytes(c, 2, "data0") == new_d0
        assert read_bytes(c, 2, "data1") == d1
        assert read(c, 2, "parity")["consistent"] is True
        # Other stripes keep their last confirmed contents.
        for idx in range(stripes):
            if idx == 2:
                continue
            for disk in ("data0", "data1", "parity"):
                assert read_bytes(c, idx, disk) == confirmed[(disk, idx)]
        assert c.get("/api/verify").json()["consistent"] is True


@pytest.mark.parametrize("offline", ["data0", "data1"])
def test_delta_readable_with_either_data_disk_offline(tmp_path, offline):
    d0, d1 = _prepare_stripe(tmp_path, 1)
    target = "data0"
    changes = _guaranteed_changes(d0, 8)
    new_d0 = _patch(d0, changes)

    with TestClient(create_app(make_settings(tmp_path))) as c:
        response = write(c, 1, target, new_d0, power_loss="neither")
        assert response.status_code == 503

    settings = make_settings(tmp_path, offline=offline)
    views = []
    for _ in range(2):  # same disk offline across two restarts
        with TestClient(create_app(settings)) as c:
            status = c.get("/api/status").json()
            assert status["mode"] == "degraded"
            assert status["journal"]["state"] == "prepared"
            assert status["last_recovery"]["record_kind"] == "delta"
            assert status["last_recovery"]["journal_cleared"] is False
            assert status["last_recovery"]["member_block_reads"] <= 2
            # The offline disk's logical view is the rolled-forward value.
            assert read_bytes(c, 1, "data0") == new_d0
            assert read_bytes(c, 1, "data1") == d1
            assert read(c, 1, offline)["source"] == "reconstructed"
            assert write(c, 0, "data0", block(1)).status_code == 409
            views.append(_full_snapshot(c, 4))
    assert views[0] == views[1]

    # Back online: the journal finishes and every disk holds the new value.
    with TestClient(create_app(make_settings(tmp_path))) as c:
        status = c.get("/api/status").json()
        assert status["journal"]["state"] == "clear"
        assert read_bytes(c, 1, "data0") == new_d0
        assert read_bytes(c, 1, "data1") == d1
        assert c.get("/api/verify").json()["consistent"] is True
        assert _full_snapshot(c, 4) == views[0]


def test_delta_targeting_data1_also_recovers_offline(tmp_path):
    d0, d1 = _prepare_stripe(tmp_path, 0)
    changes = _guaranteed_changes(d1, 5, start=900)
    new_d1 = _patch(d1, changes)

    with TestClient(create_app(make_settings(tmp_path))) as c:
        assert write(c, 0, "data1", new_d1, power_loss="parity").status_code == 503

    # The target disk itself is offline during recovery.
    with TestClient(create_app(make_settings(tmp_path, offline="data1"))) as c:
        assert read_bytes(c, 0, "data1") == new_d1
        assert read(c, 0, "data1")["source"] == "reconstructed"
        assert read_bytes(c, 0, "data0") == d0
        assert c.get("/api/status").json()["last_recovery"]["member_block_reads"] <= 2

    with TestClient(create_app(make_settings(tmp_path))) as c:
        assert read_bytes(c, 0, "data1") == new_d1
        assert c.get("/api/verify").json()["consistent"] is True


def test_delta_replay_reads_at_most_two_member_blocks(tmp_path):
    d0, _ = _prepare_stripe(tmp_path, 1)
    new_d0 = _patch(d0, _guaranteed_changes(d0, 20))

    with TestClient(create_app(make_settings(tmp_path))) as c:
        assert write(c, 1, "data0", new_d0, power_loss="data").status_code == 503

    app = create_app(make_settings(tmp_path))
    with TestClient(app) as c:
        counter = ReadCounter(app.state.array)
        with counter:
            status = c.get("/api/status").json()
        assert status["last_recovery"]["record_kind"] == "delta"
        assert counter.count <= 2


def test_delta_apply_entries_is_idempotent():
    base = bytes(_base_block(7))
    entries = tuple(
        DeltaEntry(offset=off, value=val, parity_value=val ^ 0x5A)
        for off, val in _guaranteed_changes(base, 10).items()
    )
    once = apply_entries(base, entries)
    twice = apply_entries(once, entries)
    assert once == twice


def test_legacy_v1_full_record_still_replays(tmp_path):
    d0, d1 = _prepare_stripe(tmp_path, 1)
    new0 = bytes(_base_block(0x99))
    new1 = d1

    # Forge a prepared record in the original v1 format, as an upgraded
    # service would find it on an old volume.
    journal = Journal(tmp_path / "journal.img")
    journal.write_prepared  # noqa: B018 - ensure the class is the current one
    record = new_record(stripe=1, seq=99, data0=new0, data1=new1)
    # Force the legacy on-disk magic/version through the v1 packer.
    from app import journal as journal_mod

    import hashlib

    header = journal_mod._pack_header(
        state=journal_mod.STATE_PREPARED,
        flags=0,
        stripe=1,
        data_len=BLOCK_SIZE,
        seq=99,
        created_ns=record.created_ns,
        data0_sha=hashlib.sha256(new0).digest(),
        data1_sha=hashlib.sha256(new1).digest(),
    )
    fd = os.open(tmp_path / "journal.img", os.O_RDWR)
    try:
        os.pwrite(fd, new0, HEADER_SIZE)
        os.pwrite(fd, new1, HEADER_SIZE + BLOCK_SIZE)
        os.pwrite(fd, header, 0)
        os.fsync(fd)
    finally:
        os.close(fd)

    with TestClient(create_app(make_settings(tmp_path))) as c:
        status = c.get("/api/status").json()
        assert status["last_recovery"]["recovered"] is True
        assert status["last_recovery"]["record_kind"] == "full"
        assert status["journal"]["state"] == "clear"
        assert read_bytes(c, 1, "data0") == new0
        assert read_bytes(c, 1, "data1") == d1
        assert c.get("/api/verify").json()["consistent"] is True


def test_corrupt_delta_payload_refuses_to_start(tmp_path):
    from app.errors import VolumeCorrupt

    d0, _ = _prepare_stripe(tmp_path, 0)
    with TestClient(create_app(make_settings(tmp_path))) as c:
        assert write(c, 0, "data0", _patch(d0, _guaranteed_changes(d0, 4)), power_loss="neither").status_code == 503

    journal_path = tmp_path / "journal.img"
    with open(journal_path, "r+b") as handle:
        handle.seek(HEADER_SIZE)
        original = handle.read(1)
        handle.seek(HEADER_SIZE)
        handle.write(bytes([original[0] ^ 0xFF]))

    with pytest.raises(VolumeCorrupt):
        with TestClient(create_app(make_settings(tmp_path))) as c:
            c.get("/api/status")


def test_full_record_power_loss_still_recovers(tmp_path):
    # Larger change + every breakpoint must keep working via full records.
    _prepare_stripe(tmp_path, 0)
    for scenario in ("neither", "data", "parity"):
        with TestClient(create_app(make_settings(tmp_path))) as c:
            current = read_bytes(c, 0, "data0")
            payload = _patch(current, _guaranteed_changes(current, 200))
            response = write(c, 0, "data0", payload, power_loss=scenario)
            assert response.status_code == 503
            assert response.json()["error"]["journal_record_kind"] == "full"
        with TestClient(create_app(make_settings(tmp_path))) as c:
            assert read_bytes(c, 0, "data0") == payload
            assert c.get("/api/verify").json()["consistent"] is True
            assert c.get("/api/status").json()["journal"]["state"] == "clear"


def test_delta_after_rebuilt_missing_disk_converges(tmp_path):
    d0, d1 = _prepare_stripe(tmp_path, 1)
    changes = _guaranteed_changes(d0, 16)
    new_d0 = _patch(d0, changes)

    with TestClient(create_app(make_settings(tmp_path))) as c:
        assert write(c, 1, "data0", new_d0, power_loss="parity").status_code == 503

    # Replace "the other" data disk while the delta intent is still pending.
    (tmp_path / "data1.img").unlink()
    with TestClient(create_app(make_settings(tmp_path))) as c:
        status = c.get("/api/status").json()
        assert status["rebuilt_disks"] == ["data1"]
        assert read_bytes(c, 1, "data0") == new_d0
        assert read_bytes(c, 1, "data1") == d1
        assert c.get("/api/verify").json()["consistent"] is True


def test_delta_switching_offline_disk_between_restarts(tmp_path):
    d0, d1 = _prepare_stripe(tmp_path, 1)
    new_d0 = _patch(d0, _guaranteed_changes(d0, 11))

    with TestClient(create_app(make_settings(tmp_path))) as c:
        assert write(c, 1, "data0", new_d0, power_loss="neither").status_code == 503

    # Offline target, then offline the other disk, then healthy: the retained
    # journal must produce the same rolled-forward view every time.
    with TestClient(create_app(make_settings(tmp_path, offline="data0"))) as c:
        assert read_bytes(c, 1, "data0") == new_d0
        assert read_bytes(c, 1, "data1") == d1
        view_a = _full_snapshot(c, 4)
        assert c.get("/api/status").json()["journal"]["state"] == "prepared"

    with TestClient(create_app(make_settings(tmp_path, offline="data1"))) as c:
        assert read_bytes(c, 1, "data0") == new_d0
        assert read_bytes(c, 1, "data1") == d1
        assert c.get("/api/status").json()["journal"]["state"] == "prepared"

    with TestClient(create_app(make_settings(tmp_path))) as c:
        assert c.get("/api/status").json()["journal"]["state"] == "clear"
        assert _full_snapshot(c, 4) == view_a
        assert c.get("/api/verify").json()["consistent"] is True


def test_consecutive_delta_writes_chain_correctly(tmp_path):
    d0, _ = _prepare_stripe(tmp_path, 0)
    first = _patch(d0, _guaranteed_changes(d0, 7, start=10))
    second = _patch(first, _guaranteed_changes(first, 5, start=2000))

    with TestClient(create_app(make_settings(tmp_path))) as c:
        assert write(c, 0, "data0", first).status_code == 200
        assert read_bytes(c, 0, "data0") == first
        assert c.get("/api/verify").json()["consistent"] is True

    with TestClient(create_app(make_settings(tmp_path))) as c:
        assert write(c, 0, "data0", second, power_loss="parity").status_code == 503

    with TestClient(create_app(make_settings(tmp_path))) as c:
        assert read_bytes(c, 0, "data0") == second
        assert c.get("/api/verify").json()["consistent"] is True
        info = c.get("/api/journal").json()
        assert info["state"] == "clear"


def test_delta_journal_metadata_reports_shape(tmp_path):
    d0, _ = _prepare_stripe(tmp_path, 2)
    new_d0 = _patch(d0, _guaranteed_changes(d0, 3))

    with TestClient(create_app(make_settings(tmp_path))) as c:
        assert write(c, 2, "data0", new_d0, power_loss="data").status_code == 503
        info = c.get("/api/journal").json()
        assert info["state"] == "prepared"
        assert info["record_kind"] == "delta"
        assert info["target_disk"] == "data0"
        assert info["delta_entries"] == 3
        assert info["prepared_bytes"] == 512 + 3 * 4
        assert info["small_write_limit"] == 64
        assert info["prepare_budget_bytes"] == 5 * 1024
        assert info["file_bytes"] == 512 + 2 * BLOCK_SIZE


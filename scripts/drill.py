#!/usr/bin/env python3
"""One-command power-loss drill using local uvicorn subprocesses.

    python scripts/drill.py

It creates a throwaway data directory and drives two drills against real
uvicorn processes:

1. **full-stripe drill** (unchanged from the original service): healthy writes
   -> a full-record crash with both data blocks landed and parity stale ->
   repeated degraded restarts with a data disk offline -> healthy recovery.
2. **small-write (delta) drill**: a collector-style write that changes only a
   few bytes of one 4 KiB block.  The prepared journal write is asserted to
   stay under 5 KiB including its header, and the ``neither`` / ``data`` /
   ``parity`` power-loss breakpoints are each exercised in turn.  Every case
   first persists the journal intent and then stops with, respectively, no
   member block written, both data blocks written, or only the parity block
   written.  Reads are checked with either data disk offline, across two
   consecutive offline restarts and after the disk comes back online.

For the Docker variant, follow the same steps by hand in ``/docs`` or use
``docker compose up`` with ``OFFLINE_DISK``; see README.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BLOCK = 4096
STRIPES = 4
SMALL_WRITE_LIMIT = 64
PREPARE_BUDGET = 5 * 1024


class Server:
    def __init__(self, data_dir: Path, *, offline: str = "none") -> None:
        self.data_dir = data_dir
        self.offline = offline
        self.process: subprocess.Popen | None = None
        self.port = self._free_port()

    @staticmethod
    def _free_port() -> int:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    def __enter__(self) -> "Server":
        env = dict(os.environ)
        env.update(
            DATA_DIR=str(self.data_dir),
            STRIPES=str(STRIPES),
            OFFLINE_DISK=self.offline,
            FSYNC="1",
            PYTHONPATH=str(ROOT),
        )
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--log-level",
                "warning",
            ],
            cwd=ROOT,
            env=env,
        )
        deadline = time.time() + 20
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("service exited during startup")
            try:
                self.get("/healthz")
                return self
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(0.15)
        raise RuntimeError("service did not become ready")

    def __exit__(self, *exc) -> None:
        if self.process and self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=10)

    def request(self, method: str, path: str, payload: dict | None = None):
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode()
            headers["content-type"] = "application/json"
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read() or b"{}")

    def get(self, path: str):
        return self.request("GET", path)

    def block(self, index: int, disk: str) -> dict:
        status, body = self.get(f"/api/disks/{disk}/blocks/{index}")
        assert status == 200, body
        return body

    def block_hash(self, index: int, disk: str) -> str:
        return self.block(index, disk)["sha256"]

    def snapshot(self) -> dict[str, str]:
        return {
            f"{disk}/{index}": self.block_hash(index, disk)
            for disk in ("data0", "data1", "parity")
            for index in range(STRIPES)
        }


def ok(condition: bool, message: str) -> None:
    print(("  ok   " if condition else "  FAIL ") + message)
    if not condition:
        sys.exit(1)


def small_block(base: bytes, n: int = 9, *, start: int = 200) -> bytes:
    """Return a copy of ``base`` with exactly ``n`` bytes replaced."""
    buf = bytearray(base)
    for i in range(n):
        offset = start + 13 * i
        value = base[offset] ^ 0xA5 ^ i
        buf[offset] = value
    return bytes(buf)


def drill_full_records(data_dir: Path) -> None:
    print("phase A1: healthy full-stripe writes")
    with Server(data_dir) as srv:
        for disk, fill in (("data0", 0x11), ("data1", 0x22)):
            status, _ = srv.request(
                "POST", "/api/write", {"index": 0, "disk": disk, "fill": fill}
            )
            ok(status == 200, f"wrote stripe 0 of {disk}")
        ok(srv.get("/api/verify")[1]["consistent"], "array verifies")
        confirmed = srv.snapshot()
        ok(
            srv.get("/api/status")[1]["journal"]["state"] == "clear",
            "journal cleared after confirmed write",
        )

    print("\nphase A2: full-record crash, data blocks landed, parity stale")
    with Server(data_dir) as srv:
        status, body = srv.request(
            "POST",
            "/api/write",
            {"index": 1, "disk": "data0", "fill": 0xC3, "power_loss": "data"},
        )
        ok(status == 503, "service returns 503 and stops")
        ok(body["error"]["landed"] == ["data0", "data1"], "both data blocks landed")
        ok(body["error"]["journal_record_kind"] == "full", "record kind is full")
        ok(srv.get("/api/disks/data1/blocks/1")[0] == 503, "crashed service refuses reads")

    print("\nphase A3: restart with data1 offline, twice (degraded, read-only)")
    degraded_view = None
    for attempt in (1, 2):
        with Server(data_dir, offline="data1") as srv:
            status = srv.get("/api/status")[1]
            ok(status["mode"] == "degraded", f"restart {attempt}: degraded mode")
            ok(status["writable"] is False, f"restart {attempt}: read-only")
            ok(
                status["journal"]["state"] == "prepared",
                f"restart {attempt}: journal kept for the offline disk",
            )
            ok(
                srv.block(1, "data1")["source"] == "reconstructed",
                f"restart {attempt}: offline disk read is reconstructed",
            )
            status, _ = srv.request(
                "POST", "/api/write", {"index": 2, "disk": "data0", "fill": 1}
            )
            ok(status == 409, f"restart {attempt}: write returns 409")
            view = srv.snapshot()
            if degraded_view is None:
                degraded_view = view
            else:
                ok(view == degraded_view, f"restart {attempt}: byte-identical view")
    assert degraded_view is not None
    for key, value in confirmed.items():
        if not key.endswith("/1"):
            ok(degraded_view[key] == value, f"confirmed block unchanged: {key}")

    print("\nphase A4: healthy restart finishes recovery")
    with Server(data_dir) as srv:
        status = srv.get("/api/status")[1]
        ok(status["mode"] == "healthy", "healthy mode")
        ok(status["journal"]["state"] == "clear", "journal cleared")
        ok(status["last_recovery"]["recovered"], "recovery reported")
        ok(srv.get("/api/verify")[1]["consistent"], "array verifies")
        ok(srv.snapshot() == degraded_view, "final bytes match the degraded view")


def drill_delta_records(data_dir: Path) -> None:
    index = 3
    print("\nphase B1: seed the delta stripe and capture its confirmed view")
    with Server(data_dir) as srv:
        base = bytes((0x5A + i) & 0xFF for i in range(BLOCK))
        status, _ = srv.request(
            "POST",
            "/api/write",
            {
                "index": index,
                "disk": "data0",
                "data_b64": base64.b64encode(base).decode(),
            },
        )
        ok(status == 200, "seeded the stripe")
        old_d0 = base
        new_d0 = small_block(base)
        changed = sum(a != b for a, b in zip(old_d0, new_d0))
        ok(changed <= SMALL_WRITE_LIMIT, f"small write changes {changed} <= 64 bytes")
        other_stripes = {
            key: value
            for key, value in srv.snapshot().items()
            if not key.endswith(f"/{index}")
        }

    for scenario, expected_landed in (
        ("neither", []),
        ("data", ["data0", "data1"]),
        ("parity", ["parity"]),
    ):
        print(f"\nphase B2 ({scenario}): delta intent durable, breakpoint '{scenario}'")
        # Restore the confirmed base first so each breakpoint starts identically.
        with Server(data_dir) as srv:
            status, _ = srv.request(
                "POST",
                "/api/write",
                {
                    "index": index,
                    "disk": "data0",
                    "data_b64": base64.b64encode(old_d0).decode(),
                },
            )
            ok(status == 200, "stripe reset to its confirmed base")

        with Server(data_dir) as srv:
            status, body = srv.request(
                "POST",
                "/api/write",
                {
                    "index": index,
                    "disk": "data0",
                    "data_b64": base64.b64encode(new_d0).decode(),
                    "power_loss": scenario,
                },
            )
            error = body["error"]
            ok(status == 503, f"{scenario}: service returns 503 and stops")
            ok(error["landed"] == expected_landed, f"{scenario}: landed {expected_landed}")
            ok(error["journal_record_kind"] == "delta", f"{scenario}: delta record")
            ok(
                error["journal_prepared_bytes"] < PREPARE_BUDGET,
                f"{scenario}: prepared journal write "
                f"{error['journal_prepared_bytes']} B < 5 KiB (header included)",
            )
            info = srv.get("/api/journal")[1]
            ok(info["state"] == "prepared", f"{scenario}: intent is durable")
            ok(info["record_kind"] == "delta", f"{scenario}: journal metadata says delta")

        print(f"phase B3 ({scenario}): reads while each data disk may be offline")
        for offline in ("data0", "data1"):
            views = []
            for attempt in (1, 2):  # same disk offline across two restarts
                with Server(data_dir, offline=offline) as srv:
                    status = srv.get("/api/status")[1]
                    ok(status["mode"] == "degraded", f"{scenario}/{offline} restart {attempt}: degraded")
                    ok(
                        status["last_recovery"]["member_block_reads"] <= 2,
                        f"{scenario}/{offline} restart {attempt}: <= 2 member block reads",
                    )
                    ok(
                        srv.block(index, "data0")["data_b64"]
                        == base64.b64encode(new_d0).decode(),
                        f"{scenario}/{offline} restart {attempt}: new value is readable",
                    )
                    ok(
                        srv.block(index, offline)["source"] == "reconstructed",
                        f"{scenario}/{offline} restart {attempt}: offline read reconstructed",
                    )
                    status, _ = srv.request(
                        "POST",
                        "/api/write",
                        {"index": 0, "disk": "data1", "fill": 1},
                    )
                    ok(status == 409, f"{scenario}/{offline} restart {attempt}: read-only")
                    views.append(srv.snapshot())
            ok(views[0] == views[1], f"{scenario}/{offline}: offline restarts are byte-identical")

        print(f"phase B4 ({scenario}): disk back online, full convergence")
        with Server(data_dir) as srv:
            status = srv.get("/api/status")[1]
            ok(status["journal"]["state"] == "clear", f"{scenario}: journal cleared")
            ok(
                srv.block(index, "data0")["data_b64"]
                == base64.b64encode(new_d0).decode(),
                f"{scenario}: data0 holds the new value",
            )
            ok(srv.get("/api/verify")[1]["consistent"], f"{scenario}: array verifies")
            snapshot = srv.snapshot()
            for key, value in other_stripes.items():
                ok(snapshot[key] == value, f"{scenario}: other stripes confirmed ({key})")

    print("\nphase B5: a larger change still uses a full record at every breakpoint")
    for scenario in ("neither", "data", "parity"):
        # Reset to a known, very different base so the next payload differs in
        # far more than 64 bytes regardless of the previous iteration.
        base = bytes(((scenario_index * 17) & 0xFF for scenario_index in range(BLOCK)))
        big = bytes(((i * 7 + 0x55) & 0xFF for i in range(BLOCK)))
        with Server(data_dir) as srv:
            status, _ = srv.request(
                "POST",
                "/api/write",
                {
                    "index": index,
                    "disk": "data0",
                    "data_b64": base64.b64encode(base).decode(),
                },
            )
            ok(status == 200, f"full/{scenario}: base stripe reset")
        with Server(data_dir) as srv:
            status, body = srv.request(
                "POST",
                "/api/write",
                {
                    "index": index,
                    "disk": "data0",
                    "data_b64": base64.b64encode(big).decode(),
                    "power_loss": scenario,
                },
            )
            ok(status == 503, f"full/{scenario}: service stops")
            ok(
                body["error"]["journal_record_kind"] == "full",
                f"full/{scenario}: large change uses a full record",
            )
        with Server(data_dir) as srv:
            ok(
                srv.get("/api/verify")[1]["consistent"],
                f"full/{scenario}: recovery verifies",
            )
            ok(
                srv.block(index, "data0")["data_b64"]
                == base64.b64encode(big).decode(),
                f"full/{scenario}: large write rolled forward",
            )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="xor-drill-") as tmp:
        data_dir = Path(tmp) / "data"
        print(f"data dir: {data_dir}")
        drill_full_records(data_dir)
        drill_delta_records(data_dir)
        print("\nall drill phases passed")


if __name__ == "__main__":
    main()

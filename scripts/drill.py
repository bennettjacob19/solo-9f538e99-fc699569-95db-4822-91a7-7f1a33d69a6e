#!/usr/bin/env python3
"""One-command power-loss drill using local uvicorn subprocesses.

    python scripts/drill.py

It creates a throwaway data directory, starts the service, writes confirmed
data, crashes mid-write (data blocks landed, parity stale), kills the process,
restarts with a data disk offline, checks the reconstructed view and read-only
mode, restarts twice more to prove idempotence, then restarts healthy and
verifies the array — printing every assertion along the way.

For the Docker variant, follow the same steps by hand in ``/docs`` or use
``docker compose up`` with ``OFFLINE_DISK``; see README.
"""

from __future__ import annotations

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
            STRIPES="4",
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

    def block_hash(self, index: int, disk: str) -> str:
        status, body = self.get(f"/api/disks/{disk}/blocks/{index}")
        assert status == 200, body
        return body["sha256"]

    def snapshot(self) -> dict[str, str]:
        out = {}
        for disk in ("data0", "data1", "parity"):
            for index in range(4):
                out[f"{disk}/{index}"] = self.block_hash(index, disk)
        return out


def ok(condition: bool, message: str) -> None:
    print(("  ok   " if condition else "  FAIL ") + message)
    if not condition:
        sys.exit(1)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="xor-drill-") as tmp:
        data_dir = Path(tmp) / "data"
        print(f"data dir: {data_dir}\n")

        print("phase 1: healthy writes")
        with Server(data_dir) as srv:
            for disk, fill in (("data0", 0x11), ("data1", 0x22)):
                status, _ = srv.request(
                    "POST", "/api/write", {"index": 0, "disk": disk, "fill": fill}
                )
                ok(status == 200, f"wrote stripe 0 of {disk}")
            status, verify = srv.get("/api/verify")
            ok(verify["consistent"], "array verifies")
            confirmed = srv.snapshot()
            ok(
                srv.get("/api/status")[1]["journal"]["state"] == "clear",
                "journal cleared after confirmed write",
            )

        print("\nphase 2: crash with data blocks landed, parity stale")
        with Server(data_dir) as srv:
            status, body = srv.request(
                "POST",
                "/api/write",
                {"index": 1, "disk": "data0", "fill": 0xC3, "power_loss": "data"},
            )
            ok(status == 503, "service returns 503 and stops")
            ok(
                body["error"]["landed"] == ["data0", "data1"],
                "data blocks landed, parity did not",
            )
            ok(
                srv.get("/api/disks/data1/blocks/1")[0] == 503,
                "crashed service refuses reads",
            )
        ok(
            (data_dir / "journal.img").stat().st_size == 512 + 2 * BLOCK,
            "journal is 512 B metadata + two 4 KiB blocks",
        )

        print("\nphase 3: restart with data1 offline (degraded, read-only)")
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
                status, body = srv.get("/api/disks/data1/blocks/1")
                ok(
                    body["source"] == "reconstructed",
                    f"restart {attempt}: offline disk read is reconstructed",
                )
                expected = body["sha256"]
                status, body = srv.request(
                    "POST", "/api/write", {"index": 2, "disk": "data0", "fill": 1}
                )
                ok(status == 409, f"restart {attempt}: write returns 409")
                view = srv.snapshot()
                if degraded_view is None:
                    degraded_view = view
                    ok(expected is not None, "captured logical view")
                else:
                    ok(view == degraded_view, f"restart {attempt}: byte-identical view")
        assert degraded_view is not None
        ok(
            degraded_view["data0/1"] == degraded_view["parity/1"],
            "interrupted stripe rolled forward: data0 matches parity view",
        )
        for key, value in confirmed.items():
            if not key.endswith("/1"):
                ok(degraded_view[key] == value, f"confirmed block unchanged: {key}")

        print("\nphase 4: restart healthy, finish recovery")
        with Server(data_dir) as srv:
            status = srv.get("/api/status")[1]
            ok(status["mode"] == "healthy", "healthy mode")
            ok(status["journal"]["state"] == "clear", "journal cleared")
            ok(status["last_recovery"]["recovered"], "recovery reported")
            ok(srv.get("/api/verify")[1]["consistent"], "array verifies")
            ok(srv.snapshot() == degraded_view, "final bytes match the degraded view")

    print("\nall drill phases passed")


if __name__ == "__main__":
    main()

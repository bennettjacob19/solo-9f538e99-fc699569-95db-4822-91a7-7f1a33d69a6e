"""Three-disk XOR array (A, B, parity) over persistent 4 KiB block files.

Layout: every disk is a flat file of ``stripes * 4096`` bytes.  Stripe *i*
occupies offset ``i * 4096`` on all three disks and satisfies::

    parity[i] = data0[i] XOR data1[i]

A stripe write is a whole-stripe operation guarded by the redo journal in
``journal.py``.  Because both new data blocks are journaled, recovery can
rebuild the entire stripe deterministically; the parity block never needs to
be journaled as it is a pure XOR of the two data blocks.
"""

from __future__ import annotations

import base64
import hashlib
import os
import threading
from pathlib import Path
from typing import Any

from .config import (
    BLOCK_SIZE,
    DATA_DISKS,
    DISK_NAMES,
    Settings,
    normalize_disk,
    normalize_power_loss,
)
from .errors import (
    BlockOutOfRange,
    DegradedReadOnly,
    InvalidBlockPayload,
    InvalidConfiguration,
    ServiceCrashed,
    SimulatedPowerLoss,
    VolumeCorrupt,
)
from .journal import Journal, JournalRecord, new_record

ZERO_BLOCK = bytes(BLOCK_SIZE)
POWER_LOSS_SCENARIOS = ("none", "data", "data0", "data1", "parity", "neither", "both")


def xor_blocks(left: bytes, right: bytes) -> bytes:
    return bytes(a ^ b for a, b in zip(left, right))


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class XorArray:
    """A tiny XOR array with a single-slot, crash-safe redo journal."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.offline = settings.offline_disk
        self.journal = Journal(settings.journal_path, fsync=settings.fsync)
        self._lock = threading.RLock()
        self.crashed = False
        self.last_recovery: dict[str, Any] | None = None
        self.rebuilt_disks: list[str] = []
        # Validate the journal *before* touching any disk image: a corrupt
        # intent must abort startup rather than trigger a rebuild/roll-forward.
        self._pending: JournalRecord | None = self.journal.load_pending()
        if self._pending is not None and not 0 <= self._pending.stripe < settings.stripes:
            raise VolumeCorrupt(
                "journal references a stripe outside the array",
                detail={
                    "stripe": self._pending.stripe,
                    "stripes": settings.stripes,
                },
            )
        self._ensure_disks()
        self._seq = 0
        self._recover()

    # ------------------------------------------------------------------ setup

    @property
    def path(self) -> Path:
        return Path(self.settings.data_dir)

    def _ensure_disks(self) -> None:
        """Create, validate or rebuild the backing files.

        A disk named on the command line as offline is never created or
        touched, so a genuinely absent disk can be simulated.  When the array
        starts healthy and exactly one image is missing, that image is rebuilt
        from the other two (``missing = other XOR parity``), which is the normal
        path after a replacement disk is inserted.
        """

        self.path.mkdir(parents=True, exist_ok=True)
        expected = self.settings.capacity_bytes
        paths = {name: self.settings.disk_path(name) for name in DISK_NAMES}
        existing = {name: path.exists() for name, path in paths.items()}

        if not any(existing.values()):
            for name in DISK_NAMES:
                if name == self.offline:
                    continue
                self._create_disk(name, expected)
            return

        for name, present in existing.items():
            # Never stat/validate the offline disk: it may be physically absent
            # or under repair, and the array must still come up degraded.
            if not present or name == self.offline:
                continue
            size = paths[name].stat().st_size
            if size != expected:
                raise VolumeCorrupt(
                    f"disk image {paths[name].name} has an unexpected size",
                    detail={
                        "disk": name,
                        "expected_bytes": expected,
                        "actual_bytes": size,
                    },
                )

        missing = [name for name, present in existing.items() if not present]
        if not missing:
            return
        if self.offline is not None:
            unexpected = [name for name in missing if name != self.offline]
            if unexpected:
                raise VolumeCorrupt(
                    "cannot start: disk images are missing while another disk is offline",
                    detail={"missing": missing, "offline_disk": self.offline},
                )
            return
        if len(missing) != 1:
            raise VolumeCorrupt(
                "cannot start: more than one disk image is missing",
                detail={"missing": missing, "data_dir": str(self.path)},
            )

        target = missing[0]
        if target == "parity":
            source_a, source_b = DATA_DISKS
        elif target == "data0":
            source_a, source_b = "data1", "parity"
        else:
            source_a, source_b = "data0", "parity"
        self._create_disk(target, expected)
        self._rebuild_disk(target, source_a, source_b)
        self.rebuilt_disks.append(target)

    def _create_disk(self, name: str, expected: int) -> None:
        target = self.settings.disk_path(name)
        fd = os.open(target, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            os.ftruncate(fd, expected)
            if self.settings.fsync:
                os.fsync(fd)
        finally:
            os.close(fd)
        if self.settings.fsync:
            dir_fd = os.open(self.path, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)

    def _rebuild_disk(self, target: str, source_a: str, source_b: str) -> None:
        for index in range(self.settings.stripes):
            rebuilt = xor_blocks(self._read_raw(source_a, index), self._read_raw(source_b, index))
            self._write_raw(target, index, rebuilt)

    def _recover(self) -> None:
        """Replay a pending journal record, tolerating a missing data disk.

        When a data disk is configured offline we still commit the stripe to
        every online disk (including parity).  The journal is intentionally
        *not* cleared in that case, so a later restart without the offline flag
        can finish the job.  Repeated restarts with the same offline disk are
        idempotent: they rewrite identical bytes and leave the journal as is.
        """

        header, _state = self.journal.read_header()
        self._seq = header.seq if header is not None else 0
        record = self._pending
        self._pending = None
        if record is None:
            return

        applied: list[str] = []
        if self.offline != "data0":
            self._write_raw("data0", record.stripe, record.data0)
            applied.append("data0")
        if self.offline != "data1":
            self._write_raw("data1", record.stripe, record.data1)
            applied.append("data1")
        self._write_raw("parity", record.stripe, record.parity)
        applied.append("parity")

        cleared = self.offline is None
        if cleared:
            self.journal.clear(record.seq)
        self._seq = record.seq
        self.last_recovery = {
            "recovered": True,
            "stripe": record.stripe,
            "sequence": record.seq,
            "applied_to": applied,
            "journal_cleared": cleared,
            "pending_offline_disk": self.offline,
            "data0_sha256": sha256_hex(record.data0),
            "data1_sha256": sha256_hex(record.data1),
            "parity_sha256": sha256_hex(record.parity),
        }

    # ----------------------------------------------------------------- state

    @property
    def mode(self) -> str:
        if self.crashed:
            return "crashed"
        return "degraded" if self.offline else "healthy"

    @property
    def writable(self) -> bool:
        return self.mode == "healthy"

    def _guard_operational(self) -> None:
        if self.crashed:
            raise ServiceCrashed(
                "the service simulated a power loss; restart it to recover",
                detail={"journal_stripe": self.journal.inspect().get("stripe")},
            )

    def _validate_index(self, index: int) -> int:
        try:
            value = int(index)
        except (TypeError, ValueError):
            raise BlockOutOfRange(f"block index {index!r} is not an integer")
        if not 0 <= value < self.settings.stripes:
            raise BlockOutOfRange(
                f"block index {value} is outside 0..{self.settings.stripes - 1}",
                detail={"index": value, "stripes": self.settings.stripes},
            )
        return value

    def _validate_disk(self, disk: str, *, allow_parity: bool = True) -> str:
        try:
            name = normalize_disk(disk, allow_parity=allow_parity)
        except ValueError as exc:
            raise InvalidConfiguration(str(exc)) from exc
        if name is None:
            raise InvalidConfiguration("a disk must be specified")
        return name

    def validate_disk_name(self, disk: str, *, allow_parity: bool = True) -> str:
        """Public wrapper used by the API for validation without I/O."""

        return self._validate_disk(disk, allow_parity=allow_parity)

    # --------------------------------------------------------------- raw I/O

    def _read_raw(self, name: str, index: int) -> bytes:
        target = self.settings.disk_path(name)
        try:
            fd = os.open(target, os.O_RDONLY)
        except OSError as exc:
            raise VolumeCorrupt(f"cannot open disk image {target.name}: {exc}") from exc
        try:
            data = os.pread(fd, BLOCK_SIZE, index * BLOCK_SIZE)
        finally:
            os.close(fd)
        if len(data) != BLOCK_SIZE:
            raise VolumeCorrupt(
                f"short read from {name} block {index}",
                detail={"disk": name, "index": index, "bytes": len(data)},
            )
        return data

    def _write_raw(self, name: str, index: int, data: bytes) -> None:
        target = self.settings.disk_path(name)
        try:
            fd = os.open(target, os.O_RDWR)
        except OSError as exc:
            raise VolumeCorrupt(f"cannot open disk image {target.name}: {exc}") from exc
        try:
            os.pwrite(fd, data, index * BLOCK_SIZE)
            if self.settings.fsync:
                os.fsync(fd)
        finally:
            os.close(fd)

    # --------------------------------------------------------------- commands

    def read_block(self, index: int, disk: str) -> dict[str, Any]:
        self._guard_operational()
        idx = self._validate_index(index)
        name = self._validate_disk(disk)

        with self._lock:
            if name == self.offline:
                other = "data1" if name == "data0" else "data0"
                other_data = self._read_raw(other, idx)
                parity = self._read_raw("parity", idx)
                data = xor_blocks(other_data, parity)
                source = "reconstructed"
            else:
                data = self._read_raw(name, idx)
                source = "native"

            return {
                "index": idx,
                "disk": name,
                "size": len(data),
                "data_b64": base64.b64encode(data).decode("ascii"),
                "sha256": sha256_hex(data),
                "source": source,
                "mode": self.mode,
                "offline_disk": self.offline,
                "consistent": None if self.offline else self._stripe_consistent(idx),
            }

    def write_block(
        self,
        index: int,
        disk: str,
        data: bytes,
        *,
        power_loss: str | None = None,
    ) -> dict[str, Any]:
        self._guard_operational()
        idx = self._validate_index(index)
        name = self._validate_disk(disk)
        if name == "parity":
            raise InvalidConfiguration(
                "the parity block is derived; write data0 or data1 instead"
            )
        if self.offline is not None:
            raise DegradedReadOnly(
                "the array is in read-only degraded mode because a data disk is offline",
                detail={"offline_disk": self.offline},
            )
        if len(data) != BLOCK_SIZE:
            raise InvalidBlockPayload(
                f"a block must be exactly {BLOCK_SIZE} bytes",
                detail={"received_bytes": len(data), "expected_bytes": BLOCK_SIZE},
            )
        try:
            scenario = normalize_power_loss(power_loss)
        except ValueError as exc:
            raise InvalidConfiguration(str(exc)) from exc

        with self._lock:
            current0 = self._read_raw("data0", idx)
            current1 = self._read_raw("data1", idx)
            new0 = data if name == "data0" else current0
            new1 = data if name == "data1" else current1
            seq = self._seq + 1

            record = new_record(stripe=idx, seq=seq, data0=new0, data1=new1)
            try:
                self.journal.write_prepared(record)
            except OSError as exc:
                # No disk block has been touched yet.  A torn header is treated
                # as "no intent" by recovery, so the striped data stays as-is.
                raise VolumeCorrupt(
                    f"could not persist the recovery journal: {exc}",
                    detail={"stripe": idx, "sequence": seq, "journal": "write_failed"},
                ) from exc
            self._seq = seq

            landed: list[str] = []
            try:
                if scenario in ("data", "data0", "data1"):
                    targets = ["data0", "data1"] if scenario == "data" else [scenario]
                    for disk_name in targets:
                        self._write_raw(disk_name, idx, new0 if disk_name == "data0" else new1)
                        landed.append(disk_name)
                if scenario in ("parity", "both"):
                    self._write_raw("parity", idx, record.parity)
                    landed.append("parity")
                if scenario == "none":
                    self._write_raw("data0", idx, new0)
                    self._write_raw("data1", idx, new1)
                    self._write_raw("parity", idx, record.parity)
                    landed.extend(("data0", "data1", "parity"))
                    self.journal.clear(seq)
                    return {
                        "index": idx,
                        "disk": name,
                        "size": len(data),
                        "data_b64": base64.b64encode(data).decode("ascii"),
                        "sha256": sha256_hex(data),
                        "stripe_sha256": {
                            "data0": sha256_hex(new0),
                            "data1": sha256_hex(new1),
                            "parity": sha256_hex(record.parity),
                        },
                        "sequence": seq,
                        "journal_cleared": True,
                        "power_loss": "none",
                        "mode": self.mode,
                    }
            except (OSError, VolumeCorrupt) as exc:
                self.crashed = True
                message = exc.message if isinstance(exc, VolumeCorrupt) else str(exc)
                raise VolumeCorrupt(
                    f"disk write failed and the array was stopped: {message}",
                    detail={
                        "stripe": idx,
                        "sequence": seq,
                        "landed": landed,
                        "journal": "prepared",
                        "restart_required": True,
                    },
                ) from exc

            self.crashed = True
            raise SimulatedPowerLoss(
                f"simulated power loss after {scenario} blocks landed; restart the "
                "service to recover",
                detail={
                    "stripe": idx,
                    "sequence": seq,
                    "scenario": scenario,
                    "landed": landed,
                    "journal": "prepared",
                    "restart_required": True,
                },
            )

    # ------------------------------------------------------------ inspection

    def _stripe_consistent(self, index: int) -> bool:
        d0 = self._read_raw("data0", index)
        d1 = self._read_raw("data1", index)
        p = self._read_raw("parity", index)
        return xor_blocks(xor_blocks(d0, d1), p) == ZERO_BLOCK

    def verify(self) -> dict[str, Any]:
        self._guard_operational()
        with self._lock:
            if self.offline is not None:
                return {
                    "mode": self.mode,
                    "checked": False,
                    "consistent": None,
                    "reason": "a data disk is offline; on-disk XOR cannot be verified",
                    "stripes": [],
                }
            stripes = []
            for idx in range(self.settings.stripes):
                ok = self._stripe_consistent(idx)
                stripes.append({"index": idx, "consistent": ok})
            return {
                "mode": self.mode,
                "checked": True,
                "consistent": all(item["consistent"] for item in stripes),
                "stripes": stripes,
            }

    def list_blocks(self, disk: str, *, include_data: bool = False) -> dict[str, Any]:
        """List one disk's blocks, reconstructing the offline disk if needed."""

        self._guard_operational()
        name = self._validate_disk(disk)
        with self._lock:
            blocks = []
            for idx in range(self.settings.stripes):
                if name == self.offline:
                    other = "data1" if name == "data0" else "data0"
                    data = xor_blocks(
                        self._read_raw(other, idx), self._read_raw("parity", idx)
                    )
                    source = "reconstructed"
                else:
                    data = self._read_raw(name, idx)
                    source = "native"
                entry: dict[str, Any] = {
                    "index": idx,
                    "sha256": sha256_hex(data),
                    "size": len(data),
                    "source": source,
                    "consistent": None if self.offline else self._stripe_consistent(idx),
                }
                if include_data:
                    entry["data_b64"] = base64.b64encode(data).decode("ascii")
                blocks.append(entry)
            return {
                "mode": self.mode,
                "disk": name,
                "blocks": blocks,
                "include_data": include_data,
            }

    def status(self) -> dict[str, Any]:
        with self._lock:
            journal_info = self.journal.inspect()
            return {
                "mode": self.mode,
                "writable": self.writable,
                "offline_disk": self.offline,
                "stripes": self.settings.stripes,
                "block_size": BLOCK_SIZE,
                "capacity_bytes": self.settings.capacity_bytes,
                "data_dir": str(self.settings.data_dir),
                "disks": list(DISK_NAMES),
                "sequence": self._seq,
                "journal": journal_info,
                "last_recovery": self.last_recovery,
                "rebuilt_disks": list(self.rebuilt_disks),
            }

    def journal_info(self) -> dict[str, Any]:
        with self._lock:
            return {
                "mode": self.mode,
                "sequence": self._seq,
                **self.journal.inspect(),
            }

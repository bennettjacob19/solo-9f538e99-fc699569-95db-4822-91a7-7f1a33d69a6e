"""Single-slot redo journal used to recover interrupted stripe writes.

The journal lives in its own persistent file (``journal.img``) next to the
three disk images and is deliberately small.  Two on-disk formats coexist:

* **v1 full records** (``XORJRN01``), the original format: a 512 byte header
  plus two whole 4 KiB payload blocks holding the new contents of both data
  blocks.  Old volumes ship such records and they must still replay, so the
  loader understands v1 forever; new writes no longer create them.
* **v2 records** (``XORJRN02``): the same 512 byte header plus a payload whose
  shape depends on the record kind:

  - ``full``  - two whole 4 KiB blocks, used when a write changes more than
    :data:`SMALL_WRITE_LIMIT` bytes of one data block;
  - ``delta`` - at most 64 entries describing the bytes that changed.  An
    entry is the triple
    ``(offset, new_target_byte, new_parity_byte)``: the target data block and
    the parity block are both updated with absolute byte replacements, which
    makes replay idempotent from every breakpoint and from repeated restarts.

Because the untouched data block never changes, the new parity byte of every
delta offset is already known when the intent is prepared
(``parity_new[off] = target_new[off] XOR other[off]``), so recovery never has
to guess whether parity already landed: it overwrites the journaled absolute
bytes regardless.  Replaying a delta record therefore reads at most two
member-disk blocks (the target block and the parity block; only parity when
the target disk is offline), and every other stripe keeps its last confirmed
contents untouched.

Journal write volume (the ``prepared`` intent; clearing the journal is counted
separately):

* delta:  ``512 B header + <= 64 * 4 B payload`` <= 768 B, always below the
  5 KiB budget for small writes;
* full:   ``512 B header + 2 * 4 KiB payload`` = 8704 B.

The file itself stays preallocated at 8704 B so both formats share one image
and old volumes need no migration.

Write ordering:

1. the payload is written and fsynced,
2. the header is written and fsynced -> the intent is now durable,
3. the caller writes the stripe's disk blocks,
4. the header is rewritten in the ``clear`` state and fsynced.

A crash at any point after step 2 leaves a ``prepared`` header behind.  A torn
header (step 2 never completed) means no disk block write could have started,
so it is safe to ignore.
"""

from __future__ import annotations

import hashlib
import os
import struct
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

from .config import BLOCK_SIZE
from .errors import VolumeCorrupt

# --- on-disk magic / versions ------------------------------------------------
MAGIC_V1 = b"XORJRN01"
MAGIC_V2 = b"XORJRN02"
MAGIC = MAGIC_V2  # current format for newly written headers
VERSION_V1 = 1
VERSION_V2 = 2
VERSION = VERSION_V2

HEADER_SIZE = 512
PAYLOAD_BLOCKS = 2
PAYLOAD_SIZE = PAYLOAD_BLOCKS * BLOCK_SIZE
TOTAL_SIZE = HEADER_SIZE + PAYLOAD_SIZE

# Small-write budget: a write touching at most this many bytes of one 4 KiB
# block is journaled as a delta record.  The prepared journal write must stay
# below this many bytes including the header.
SMALL_WRITE_LIMIT = 64
JOURNAL_PREPARE_BUDGET = 5 * 1024

# Delta payload packing: one entry is a 16-bit block offset plus the absolute
# new bytes of the target data block and of the parity block at that offset.
_ENTRY = struct.Struct("<HBB")
DELTA_ENTRY_SIZE = _ENTRY.size

STATE_EMPTY = 0
STATE_PREPARED = 1

KIND_FULL = 1
KIND_DELTA = 2

TARGET_DATA0 = 0
TARGET_DATA1 = 1
_TARGET_NAMES = {TARGET_DATA0: "data0", TARGET_DATA1: "data1"}
_TARGET_IDS = {"data0": TARGET_DATA0, "data1": TARGET_DATA1}

# v1 header (legacy, read-only): magic H H B B I I Q Q 32s 32s
_BODY_V1 = struct.Struct("<8sHHBBIIQQ32s32s")
# v2 header: magic H H B B B H I Q Q I 32s 32s 32s
_BODY_V2 = struct.Struct("<8sHHBBBHIQQI32s32s32s")
_CRC = struct.Struct("<I")
assert HEADER_SIZE >= _BODY_V1.size + _CRC.size
assert HEADER_SIZE >= _BODY_V2.size + _CRC.size


@dataclass(frozen=True)
class DeltaEntry:
    """One absolute byte replacement inside a 4 KiB stripe.

    ``value`` is the new byte of the target data block at ``offset`` and
    ``parity_value`` is the corresponding new parity byte.  Both are absolute
    replacements, so replaying an entry any number of times converges.
    """

    offset: int
    value: int
    parity_value: int

    def __post_init__(self) -> None:
        if not 0 <= self.offset < BLOCK_SIZE:
            raise ValueError("delta offset is outside the 4 KiB block")
        if not 0 <= self.value <= 0xFF:
            raise ValueError("delta value must be a byte")
        if not 0 <= self.parity_value <= 0xFF:
            raise ValueError("delta parity value must be a byte")


@dataclass(frozen=True)
class JournalRecord:
    """A complete, durable description of one in-flight stripe update.

    Full records carry both new data blocks; delta records carry a target disk
    and at most :data:`SMALL_WRITE_LIMIT` absolute byte replacements.
    """

    stripe: int
    seq: int
    created_ns: int
    kind: int = KIND_FULL
    target: int | None = None
    data0: bytes | None = None
    data1: bytes | None = None
    entries: tuple[DeltaEntry, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.kind == KIND_FULL:
            if len(self.data0 or b"") != BLOCK_SIZE or len(self.data1 or b"") != BLOCK_SIZE:
                raise ValueError("a full journal record needs two 4 KiB payload blocks")
        elif self.kind == KIND_DELTA:
            if self.target not in _TARGET_NAMES:
                raise ValueError("a delta record must target data0 or data1")
            if len(self.entries) > SMALL_WRITE_LIMIT:
                raise ValueError("a delta record holds at most 64 entries")
            offsets = [entry.offset for entry in self.entries]
            if len(set(offsets)) != len(offsets):
                raise ValueError("delta entries must not repeat an offset")
        else:
            raise ValueError(f"unknown journal record kind {self.kind}")

    # ------------------------------------------------------------- full view
    @property
    def is_delta(self) -> bool:
        return self.kind == KIND_DELTA

    @property
    def target_name(self) -> str | None:
        return _TARGET_NAMES.get(self.target) if self.target is not None else None

    @property
    def parity(self) -> bytes:
        """Full-stripe parity (full records only; delta derives it on replay)."""
        if self.is_delta:
            raise AttributeError("delta records derive parity during replay")
        return bytes(a ^ b for a, b in zip(self.data0, self.data1))

    @property
    def data0_sha256(self) -> str | None:
        return hashlib.sha256(self.data0).hexdigest() if self.data0 is not None else None

    @property
    def data1_sha256(self) -> str | None:
        return hashlib.sha256(self.data1).hexdigest() if self.data1 is not None else None

    @property
    def prepared_bytes(self) -> int:
        """Bytes written to the journal while preparing this intent."""
        if self.is_delta:
            return HEADER_SIZE + len(self.entries) * DELTA_ENTRY_SIZE
        return HEADER_SIZE + 2 * BLOCK_SIZE


@dataclass(frozen=True)
class Header:
    version: int
    state: int
    kind: int | None
    target: int | None
    flags: int
    entries: int
    stripe: int
    data_len: int
    seq: int
    created_ns: int
    payload_sha: bytes
    data0_sha: bytes
    data1_sha: bytes


def apply_target_entries(block: bytes, entries: tuple[DeltaEntry, ...]) -> bytes:
    """Apply absolute target-block byte replacements (idempotent)."""
    if len(block) != BLOCK_SIZE:
        raise ValueError("delta replay requires a full 4 KiB block")
    rebuilt = bytearray(block)
    for entry in entries:
        rebuilt[entry.offset] = entry.value
    return bytes(rebuilt)


def apply_parity_entries(block: bytes, entries: tuple[DeltaEntry, ...]) -> bytes:
    """Apply absolute parity-block byte replacements (idempotent)."""
    if len(block) != BLOCK_SIZE:
        raise ValueError("delta replay requires a full 4 KiB block")
    rebuilt = bytearray(block)
    for entry in entries:
        rebuilt[entry.offset] = entry.parity_value
    return bytes(rebuilt)


# Backwards-compatible alias for earlier call sites and tests.
def apply_entries(block: bytes, entries: tuple[DeltaEntry, ...]) -> bytes:
    return apply_target_entries(block, entries)


def pack_delta(entries: tuple[DeltaEntry, ...]) -> bytes:
    return b"".join(
        _ENTRY.pack(entry.offset, entry.value, entry.parity_value)
        for entry in entries
    )


def unpack_delta(blob: bytes) -> tuple[DeltaEntry, ...]:
    if len(blob) % DELTA_ENTRY_SIZE:
        raise VolumeCorrupt("delta payload has a truncated entry")
    entries = tuple(
        DeltaEntry(offset, value, parity_value)
        for offset, value, parity_value in _ENTRY.iter_unpack(blob)
    )
    if len(entries) > SMALL_WRITE_LIMIT:
        raise VolumeCorrupt(
            "delta payload exceeds the 64-entry small-write limit",
            detail={"entries": len(entries)},
        )
    offsets = [entry.offset for entry in entries]
    if len(set(offsets)) != len(offsets):
        raise VolumeCorrupt("delta payload repeats a block offset")
    return entries


# ----------------------------------------------------------------- headers ---
def _pack_header_v2(
    *,
    state: int,
    kind: int,
    target: int | None,
    entries: int,
    stripe: int,
    data_len: int,
    seq: int,
    created_ns: int,
    payload_sha: bytes,
    data0_sha: bytes,
    data1_sha: bytes,
) -> bytes:
    body = _BODY_V2.pack(
        MAGIC_V2,
        VERSION_V2,
        HEADER_SIZE,
        state,
        kind,
        0xFF if target is None else target,
        entries,
        stripe,
        seq,
        created_ns,
        data_len,
        payload_sha,
        data0_sha,
        data1_sha,
    )
    crc = zlib.crc32(body) & 0xFFFFFFFF
    return body + _CRC.pack(crc) + b"\x00" * (HEADER_SIZE - _BODY_V2.size - _CRC.size)


def _pack_empty_header(seq: int) -> bytes:
    zero = b"\x00" * 32
    return _pack_header_v2(
        state=STATE_EMPTY,
        kind=0,
        target=None,
        entries=0,
        stripe=0,
        data_len=0,
        seq=seq,
        created_ns=0,
        payload_sha=zero,
        data0_sha=zero,
        data1_sha=zero,
    )


def _pack_header(
    *,
    state: int,
    flags: int,
    stripe: int,
    data_len: int,
    seq: int,
    created_ns: int,
    data0_sha: bytes,
    data1_sha: bytes,
) -> bytes:
    """Legacy v1 header packer, kept for tests that forge torn/old headers."""
    body = _BODY_V1.pack(
        MAGIC_V1,
        VERSION_V1,
        HEADER_SIZE,
        state,
        flags,
        stripe,
        data_len,
        seq,
        created_ns,
        data0_sha,
        data1_sha,
    )
    crc = zlib.crc32(body) & 0xFFFFFFFF
    return body + _CRC.pack(crc) + b"\x00" * (HEADER_SIZE - _BODY_V1.size - _CRC.size)


def _parse_header(raw: bytes) -> Header | None:
    if len(raw) < HEADER_SIZE:
        return None
    magic = raw[:8]
    try:
        if magic == MAGIC_V1:
            unpacked = _BODY_V1.unpack_from(raw, 0)
            (crc,) = _CRC.unpack_from(raw, _BODY_V1.size)
            if crc != zlib.crc32(raw[: _BODY_V1.size]) & 0xFFFFFFFF:
                return None
            (_magic, version, header_size, state, flags, stripe,
             data_len, seq, created_ns, sha0, sha1) = unpacked
            if header_size != HEADER_SIZE:
                return None
            return Header(
                version=version,
                state=state,
                kind=KIND_FULL,
                target=None,
                flags=flags,
                entries=0,
                stripe=stripe,
                data_len=data_len,
                seq=seq,
                created_ns=created_ns,
                payload_sha=b"",
                data0_sha=sha0,
                data1_sha=sha1,
            )
        if magic == MAGIC_V2:
            unpacked = _BODY_V2.unpack_from(raw, 0)
            (crc,) = _CRC.unpack_from(raw, _BODY_V2.size)
            if crc != zlib.crc32(raw[: _BODY_V2.size]) & 0xFFFFFFFF:
                return None
            (_magic, version, header_size, state, kind, target, entries,
             stripe, seq, created_ns, data_len, payload_sha, sha0, sha1) = unpacked
            if header_size != HEADER_SIZE:
                return None
            return Header(
                version=version,
                state=state,
                kind=None if state == STATE_EMPTY else kind,
                target=None if target == 0xFF else target,
                flags=0,
                entries=entries,
                stripe=stripe,
                data_len=data_len,
                seq=seq,
                created_ns=created_ns,
                payload_sha=payload_sha,
                data0_sha=sha0,
                data1_sha=sha1,
            )
    except struct.error:
        return None
    return None


def _pwrite_all(fd: int, data: bytes, offset: int) -> None:
    view = memoryview(data)
    written = 0
    while written < len(view):
        n = os.pwrite(fd, view[written:], offset + written)
        if n <= 0:  # pragma: no cover - defensive, regular files do not do this
            raise OSError("short write to journal")
        written += n


class Journal:
    """Single-slot redo journal backed by ``journal.img``."""

    def __init__(self, path: Path, *, fsync: bool = True) -> None:
        self.path = Path(path)
        self.fsync = fsync
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_file()

    def _ensure_file(self) -> None:
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            size = os.fstat(fd).st_size
            if size < TOTAL_SIZE:
                os.ftruncate(fd, TOTAL_SIZE)
                if self.fsync:
                    os.fsync(fd)
                    dir_fd = os.open(self.path.parent, os.O_RDONLY)
                    try:
                        os.fsync(dir_fd)
                    finally:
                        os.close(dir_fd)
        finally:
            os.close(fd)

    def _open(self) -> int:
        return os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)

    def _sync(self, fd: int) -> None:
        if self.fsync:
            os.fsync(fd)

    def read_header(self) -> tuple[Header | None, str]:
        """Return the parsed header (or None) plus a diagnostic state string."""

        fd = self._open()
        try:
            raw = os.pread(fd, HEADER_SIZE, 0)
        finally:
            os.close(fd)
        if len(raw) < HEADER_SIZE or raw[:8] not in (MAGIC_V1, MAGIC_V2):
            return None, "empty"
        header = _parse_header(raw)
        if header is None:
            return None, "invalid_header"
        if header.state == STATE_EMPTY:
            return header, "clear"
        if header.state == STATE_PREPARED:
            return header, "prepared"
        return header, "unknown_state"

    # ------------------------------------------------------------- loading
    def load_pending(self) -> JournalRecord | None:
        """Load a prepared record, verifying every payload checksum.

        Supports both legacy v1 full records and v2 full/delta records.
        Raises :class:`VolumeCorrupt` when an intent's payload no longer
        matches its digest: the on-disk stripe state would be unknowable, so
        refusing to start is safer than presenting unverifiable data.
        """

        header, state = self.read_header()
        if header is None or state == "clear":
            return None
        if state != "prepared":
            raise VolumeCorrupt(
                f"journal header has an unknown state ({state})",
                detail={"state": header.state},
            )

        if header.version == VERSION_V1:
            return self._load_v1_full(header)
        if header.kind == KIND_FULL:
            return self._load_v2_full(header)
        if header.kind == KIND_DELTA:
            return self._load_v2_delta(header)
        raise VolumeCorrupt(
            "journal header declares an unknown record kind",
            detail={"kind": header.kind},
        )

    def _read_payload(self, length: int) -> bytes:
        fd = self._open()
        try:
            blob = os.pread(fd, length, HEADER_SIZE)
        finally:
            os.close(fd)
        if len(blob) != length:
            raise VolumeCorrupt("journal payload is truncated")
        return blob

    def _load_v1_full(self, header: Header) -> JournalRecord:
        if header.data_len != BLOCK_SIZE:
            raise VolumeCorrupt(
                "journal header declares an unexpected payload size",
                detail={"data_len": header.data_len},
            )
        fd = self._open()
        try:
            data0 = os.pread(fd, BLOCK_SIZE, HEADER_SIZE)
            data1 = os.pread(fd, BLOCK_SIZE, HEADER_SIZE + BLOCK_SIZE)
        finally:
            os.close(fd)
        if len(data0) != BLOCK_SIZE or len(data1) != BLOCK_SIZE:
            raise VolumeCorrupt("journal payload is truncated")
        if hashlib.sha256(data0).digest() != header.data0_sha:
            raise VolumeCorrupt("journal data0 payload failed its SHA-256 check")
        if hashlib.sha256(data1).digest() != header.data1_sha:
            raise VolumeCorrupt("journal data1 payload failed its SHA-256 check")
        return JournalRecord(
            stripe=header.stripe,
            seq=header.seq,
            created_ns=header.created_ns,
            kind=KIND_FULL,
            data0=data0,
            data1=data1,
        )

    def _load_v2_full(self, header: Header) -> JournalRecord:
        if header.data_len != BLOCK_SIZE:
            raise VolumeCorrupt(
                "journal header declares an unexpected payload size",
                detail={"data_len": header.data_len},
            )
        fd = self._open()
        try:
            data0 = os.pread(fd, BLOCK_SIZE, HEADER_SIZE)
            data1 = os.pread(fd, BLOCK_SIZE, HEADER_SIZE + BLOCK_SIZE)
        finally:
            os.close(fd)
        if len(data0) != BLOCK_SIZE or len(data1) != BLOCK_SIZE:
            raise VolumeCorrupt("journal payload is truncated")
        payload = data0 + data1
        if hashlib.sha256(payload).digest() != header.payload_sha:
            raise VolumeCorrupt("journal full payload failed its SHA-256 check")
        if hashlib.sha256(data0).digest() != header.data0_sha:
            raise VolumeCorrupt("journal data0 payload failed its SHA-256 check")
        if hashlib.sha256(data1).digest() != header.data1_sha:
            raise VolumeCorrupt("journal data1 payload failed its SHA-256 check")
        return JournalRecord(
            stripe=header.stripe,
            seq=header.seq,
            created_ns=header.created_ns,
            kind=KIND_FULL,
            data0=data0,
            data1=data1,
        )

    def _load_v2_delta(self, header: Header) -> JournalRecord:
        if header.target not in _TARGET_NAMES:
            raise VolumeCorrupt(
                "delta journal header has no valid target disk",
                detail={"target": header.target},
            )
        expected_len = header.entries * DELTA_ENTRY_SIZE
        if header.data_len != expected_len or not 0 <= expected_len <= (
            SMALL_WRITE_LIMIT * DELTA_ENTRY_SIZE
        ):
            raise VolumeCorrupt(
                "delta journal header declares an unexpected payload size",
                detail={"data_len": header.data_len, "entries": header.entries},
            )
        blob = self._read_payload(expected_len)
        if hashlib.sha256(blob).digest() != header.payload_sha:
            raise VolumeCorrupt("journal delta payload failed its SHA-256 check")
        entries = unpack_delta(blob)
        if len(entries) != header.entries:
            raise VolumeCorrupt("delta entry count does not match the header")
        return JournalRecord(
            stripe=header.stripe,
            seq=header.seq,
            created_ns=header.created_ns,
            kind=KIND_DELTA,
            target=header.target,
            entries=entries,
        )

    # ------------------------------------------------------------- writing
    def write_prepared(self, record: JournalRecord) -> None:
        if record.is_delta:
            self._write_prepared_delta(record)
        else:
            self._write_prepared_full(record)

    def _write_prepared_full(self, record: JournalRecord) -> None:
        assert record.data0 is not None and record.data1 is not None
        payload = record.data0 + record.data1
        header = _pack_header_v2(
            state=STATE_PREPARED,
            kind=KIND_FULL,
            target=None,
            entries=0,
            stripe=record.stripe,
            data_len=BLOCK_SIZE,
            seq=record.seq,
            created_ns=record.created_ns,
            payload_sha=hashlib.sha256(payload).digest(),
            data0_sha=hashlib.sha256(record.data0).digest(),
            data1_sha=hashlib.sha256(record.data1).digest(),
        )
        fd = self._open()
        try:
            _pwrite_all(fd, payload, HEADER_SIZE)
            self._sync(fd)
            _pwrite_all(fd, header, 0)
            self._sync(fd)
        finally:
            os.close(fd)

    def _write_prepared_delta(self, record: JournalRecord) -> None:
        blob = pack_delta(record.entries)
        header = _pack_header_v2(
            state=STATE_PREPARED,
            kind=KIND_DELTA,
            target=record.target,
            entries=len(record.entries),
            stripe=record.stripe,
            data_len=len(blob),
            seq=record.seq,
            created_ns=record.created_ns,
            payload_sha=hashlib.sha256(blob).digest(),
            data0_sha=b"\x00" * 32,
            data1_sha=b"\x00" * 32,
        )
        fd = self._open()
        try:
            _pwrite_all(fd, blob, HEADER_SIZE)
            self._sync(fd)
            _pwrite_all(fd, header, 0)
            self._sync(fd)
        finally:
            os.close(fd)

    def clear(self, seq: int) -> None:
        fd = self._open()
        try:
            _pwrite_all(fd, _pack_empty_header(seq), 0)
            self._sync(fd)
        finally:
            os.close(fd)

    # ------------------------------------------------------------- inspect
    def inspect(self) -> dict:
        """Metadata-only view used by ``GET /api/journal`` (never raises)."""

        base = {
            "header_bytes": HEADER_SIZE,
            "payload_blocks": PAYLOAD_BLOCKS,
            "payload_bytes": PAYLOAD_SIZE,
            "max_bytes": TOTAL_SIZE,
            "small_write_limit": SMALL_WRITE_LIMIT,
            "prepare_budget_bytes": JOURNAL_PREPARE_BUDGET,
        }
        try:
            header, state = self.read_header()
        except OSError as exc:
            return {
                "state": "unavailable",
                "message": str(exc),
                "file_bytes": None,
                "stripe": None,
                "seq": None,
                "created_ns": None,
                "record_kind": None,
                "target_disk": None,
                "delta_entries": None,
                "prepared_bytes": None,
                **base,
            }
        try:
            size = self.path.stat().st_size
        except OSError:
            size = None
        info = {
            "state": state,
            "file_bytes": size,
            "stripe": None,
            "seq": None,
            "created_ns": None,
            "record_kind": None,
            "target_disk": None,
            "delta_entries": None,
            "prepared_bytes": 0,
            **base,
        }
        if header is not None:
            info["seq"] = header.seq
            info["stripe"] = header.stripe
            info["created_ns"] = header.created_ns
            if state == "prepared":
                info["record_kind"] = "delta" if header.kind == KIND_DELTA else "full"
                info["target_disk"] = _TARGET_NAMES.get(header.target)
                info["delta_entries"] = header.entries if header.kind == KIND_DELTA else None
                info["prepared_bytes"] = (
                    HEADER_SIZE + header.entries * DELTA_ENTRY_SIZE
                    if header.kind == KIND_DELTA
                    else TOTAL_SIZE
                )
        if state == "prepared":
            try:
                self.load_pending()
            except VolumeCorrupt as exc:
                info["state"] = "corrupt"
                info["message"] = exc.message
        return info


def new_record(*, stripe: int, seq: int, data0: bytes, data1: bytes) -> JournalRecord:
    """Build a full-stripe record (legacy constructor, still used by tests)."""
    return JournalRecord(
        stripe=stripe,
        seq=seq,
        created_ns=time.time_ns(),
        kind=KIND_FULL,
        data0=data0,
        data1=data1,
    )


def new_full_record(
    *, stripe: int, seq: int, data0: bytes, data1: bytes
) -> JournalRecord:
    return new_record(stripe=stripe, seq=seq, data0=data0, data1=data1)


def new_delta_record(
    *, stripe: int, seq: int, target: str, entries: tuple[DeltaEntry, ...]
) -> JournalRecord:
    if target not in _TARGET_IDS:
        raise ValueError("delta record target must be data0 or data1")
    return JournalRecord(
        stripe=stripe,
        seq=seq,
        created_ns=time.time_ns(),
        kind=KIND_DELTA,
        target=_TARGET_IDS[target],
        entries=tuple(entries),
    )

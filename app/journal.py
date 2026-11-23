"""Single-slot redo journal used to recover interrupted stripe writes.

The journal is kept in its own persistent file (``journal.img``) next to the
three disk images.  It is deliberately small:

* one 512 byte header holding the intent metadata (state, sequence, stripe,
  payload hashes and timestamp),
* two 4 KiB payload blocks holding the *new* contents of the two data blocks.

The parity block never needs to be journaled because it is a pure function of
the two data blocks (``parity = data0 XOR data1``), so two blocks are enough to
describe a full stripe update.  That matches the drill requirement of "at most
two blocks plus a little metadata".

Write ordering:

1. payload blocks are written and fsynced,
2. the header is written and fsynced -> the intent is now durable,
3. the caller writes the stripe's disk blocks,
4. the header is rewritten in the ``clear`` state and fsynced.

A crash at any point after step 2 leaves a ``prepared`` header behind.  The
recovery path therefore always has a complete new picture of the stripe and
can deterministically commit it.  A torn header (step 2 never completed)
means no disk block write could have started, so it is safe to ignore.
"""

from __future__ import annotations

import hashlib
import os
import struct
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

from .config import BLOCK_SIZE
from .errors import VolumeCorrupt

MAGIC = b"XORJRN01"
VERSION = 1
HEADER_SIZE = 512
PAYLOAD_BLOCKS = 2
PAYLOAD_SIZE = PAYLOAD_BLOCKS * BLOCK_SIZE
TOTAL_SIZE = HEADER_SIZE + PAYLOAD_SIZE

STATE_EMPTY = 0
STATE_PREPARED = 1

_BODY = struct.Struct("<8sHHBBIIQQ32s32s")
_CRC = struct.Struct("<I")
assert HEADER_SIZE >= _BODY.size + _CRC.size


@dataclass(frozen=True)
class JournalRecord:
    """A complete, durable description of one in-flight stripe update."""

    stripe: int
    seq: int
    data0: bytes
    data1: bytes
    created_ns: int
    flags: int = 0

    def __post_init__(self) -> None:
        if len(self.data0) != BLOCK_SIZE or len(self.data1) != BLOCK_SIZE:
            raise ValueError("journal payload blocks must be exactly 4 KiB")

    @property
    def parity(self) -> bytes:
        return bytes(a ^ b for a, b in zip(self.data0, self.data1))

    @property
    def data0_sha256(self) -> str:
        return hashlib.sha256(self.data0).hexdigest()

    @property
    def data1_sha256(self) -> str:
        return hashlib.sha256(self.data1).hexdigest()


@dataclass(frozen=True)
class Header:
    state: int
    flags: int
    stripe: int
    data_len: int
    seq: int
    created_ns: int
    data0_sha: bytes
    data1_sha: bytes


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
    body = _BODY.pack(
        MAGIC,
        VERSION,
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
    return body + _CRC.pack(crc) + b"\x00" * (HEADER_SIZE - _BODY.size - _CRC.size)


def _parse_header(raw: bytes) -> Header | None:
    if len(raw) < _BODY.size + _CRC.size:
        return None
    try:
        unpacked = _BODY.unpack_from(raw, 0)
        (crc,) = _CRC.unpack_from(raw, _BODY.size)
    except struct.error:
        return None
    magic, version, header_size, state, flags, stripe, data_len, seq, created_ns, sha0, sha1 = unpacked
    if magic != MAGIC or version != VERSION or header_size != HEADER_SIZE:
        return None
    expected = zlib.crc32(raw[: _BODY.size]) & 0xFFFFFFFF
    if crc != expected:
        return None
    return Header(
        state=state,
        flags=flags,
        stripe=stripe,
        data_len=data_len,
        seq=seq,
        created_ns=created_ns,
        data0_sha=sha0,
        data1_sha=sha1,
    )


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
        if len(raw) < HEADER_SIZE or raw[: len(MAGIC)] != MAGIC:
            return None, "empty"
        header = _parse_header(raw)
        if header is None:
            return None, "invalid_header"
        if header.state == STATE_EMPTY:
            return header, "clear"
        if header.state == STATE_PREPARED:
            return header, "prepared"
        return header, "unknown_state"

    def load_pending(self) -> JournalRecord | None:
        """Load a prepared record, verifying both payload checksums.

        Raises :class:`VolumeCorrupt` when the header claims a prepared intent
        but the payload no longer matches its digest.  In that situation the
        on-disk stripe state is unknowable, so refusing to start is safer than
        silently presenting unverifiable data.
        """

        header, state = self.read_header()
        if header is None:
            return None
        if state == "clear":
            return None
        if state != "prepared":
            # A CRC-valid header with an unknown state byte cannot be produced
            # by this version; refuse to guess rather than ignore an intent.
            raise VolumeCorrupt(
                f"journal header has an unknown state ({state})",
                detail={"state": header.state},
            )
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
            data0=data0,
            data1=data1,
            created_ns=header.created_ns,
            flags=header.flags,
        )

    def write_prepared(self, record: JournalRecord) -> None:
        header = _pack_header(
            state=STATE_PREPARED,
            flags=record.flags,
            stripe=record.stripe,
            data_len=BLOCK_SIZE,
            seq=record.seq,
            created_ns=record.created_ns,
            data0_sha=hashlib.sha256(record.data0).digest(),
            data1_sha=hashlib.sha256(record.data1).digest(),
        )
        fd = self._open()
        try:
            _pwrite_all(fd, record.data0, HEADER_SIZE)
            _pwrite_all(fd, record.data1, HEADER_SIZE + BLOCK_SIZE)
            self._sync(fd)
            _pwrite_all(fd, header, 0)
            self._sync(fd)
        finally:
            os.close(fd)

    def clear(self, seq: int) -> None:
        header = _pack_header(
            state=STATE_EMPTY,
            flags=0,
            stripe=0,
            data_len=0,
            seq=seq,
            created_ns=0,
            data0_sha=b"\x00" * 32,
            data1_sha=b"\x00" * 32,
        )
        fd = self._open()
        try:
            _pwrite_all(fd, header, 0)
            self._sync(fd)
        finally:
            os.close(fd)

    def inspect(self) -> dict:
        """Metadata-only view used by ``GET /api/journal`` (never raises)."""

        try:
            header, state = self.read_header()
        except OSError as exc:
            return {
                "state": "unavailable",
                "message": str(exc),
                "header_bytes": HEADER_SIZE,
                "payload_blocks": PAYLOAD_BLOCKS,
                "payload_bytes": PAYLOAD_SIZE,
                "file_bytes": TOTAL_SIZE,
            }
        try:
            size = self.path.stat().st_size
        except OSError:
            size = None
        info = {
            "state": state,
            "file_bytes": size,
            "header_bytes": HEADER_SIZE,
            "payload_blocks": PAYLOAD_BLOCKS,
            "payload_bytes": PAYLOAD_SIZE,
            "max_bytes": TOTAL_SIZE,
            "stripe": None,
            "seq": None,
            "created_ns": None,
            "data0_sha256": None,
            "data1_sha256": None,
        }
        if header is not None:
            info["seq"] = header.seq
            info["stripe"] = header.stripe
            info["created_ns"] = header.created_ns
            info["data0_sha256"] = header.data0_sha.hex()
            info["data1_sha256"] = header.data1_sha.hex()
        if state == "prepared":
            try:
                self.load_pending()
            except VolumeCorrupt as exc:
                info["state"] = "corrupt"
                info["message"] = exc.message
        return info


def new_record(*, stripe: int, seq: int, data0: bytes, data1: bytes) -> JournalRecord:
    return JournalRecord(
        stripe=stripe,
        seq=seq,
        data0=data0,
        data1=data1,
        created_ns=time.time_ns(),
    )

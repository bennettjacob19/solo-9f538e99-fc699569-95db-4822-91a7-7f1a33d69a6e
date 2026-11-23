"""Runtime configuration for the three-disk XOR array drill."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BLOCK_SIZE = 4096
DISK_NAMES: tuple[str, str, str] = ("data0", "data1", "parity")
DATA_DISKS: tuple[str, str] = ("data0", "data1")
JOURNAL_NAME = "journal.img"

_DISK_ALIASES = {
    "0": "data0",
    "data0": "data0",
    "d0": "data0",
    "1": "data1",
    "data1": "data1",
    "d1": "data1",
    "parity": "parity",
    "p": "parity",
    "": None,
    "none": None,
    "off": None,
    "null": None,
}


def normalize_disk(value: str | None, *, allow_parity: bool = True) -> str | None:
    """Normalize a user supplied disk identifier.

    ``None``/empty/``none`` mean "no disk".  Integer aliases make the drill
    friendlier to use from curl, while the canonical names are used in every
    response.
    """

    if value is None:
        return None
    key = value.strip().lower()
    if key not in _DISK_ALIASES:
        raise ValueError(f"unknown disk {value!r}; expected data0, data1 or parity")
    name = _DISK_ALIASES[key]
    if name == "parity" and not allow_parity:
        raise ValueError("the parity disk cannot be taken offline in this drill")
    return name


def normalize_power_loss(value: str | None) -> str:
    """Normalize the power-loss simulation selector used by writes."""

    if value is None or value.strip() == "":
        return "none"
    key = value.strip().lower().replace("-", "_")
    aliases = {
        "none": "none",
        "off": "none",
        "no": "none",
        "data": "data",
        "data_only": "data",
        "data_landed": "data",
        "parity": "parity",
        "parity_only": "parity",
        "parity_landed": "parity",
        "neither": "neither",
        "none_landed": "neither",
        "journal_only": "neither",
        "both": "both",
        "both_landed": "both",
        "data0": "data0",
        "d0": "data0",
        "data0_only": "data0",
        "data1": "data1",
        "d1": "data1",
        "data1_only": "data1",
    }
    if key not in aliases:
        raise ValueError(
            f"unknown power-loss scenario {value!r}; expected none, data, data0, "
            "data1, parity, neither or both"
        )
    return aliases[key]


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    stripes: int = 16
    offline_disk: str | None = None
    fsync: bool = True

    def __post_init__(self) -> None:
        if self.stripes < 1:
            raise ValueError("STRIPES must be a positive integer")
        offline = normalize_disk(self.offline_disk, allow_parity=False)
        object.__setattr__(self, "offline_disk", offline)

    @property
    def block_size(self) -> int:
        return BLOCK_SIZE

    @property
    def capacity_bytes(self) -> int:
        return self.stripes * BLOCK_SIZE

    def disk_path(self, name: str) -> Path:
        return Path(self.data_dir) / f"{name}.img"

    @property
    def journal_path(self) -> Path:
        return Path(self.data_dir) / JOURNAL_NAME

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = Path(os.environ.get("DATA_DIR", "./data")).resolve()
        stripes = int(os.environ.get("STRIPES", "16"))
        offline = os.environ.get("OFFLINE_DISK", "none")
        fsync = os.environ.get("FSYNC", "1").lower() not in {"0", "false", "no", "off"}
        return cls(
            data_dir=data_dir,
            stripes=stripes,
            offline_disk=normalize_disk(offline, allow_parity=False),
            fsync=fsync,
        )

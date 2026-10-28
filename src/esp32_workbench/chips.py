"""Chip database: per-target GPIO, ADC and toolchain facts.

Every chip file in ``data/chips/`` cites the Espressif sources it was built from.
Fields that could not be confirmed from a primary source are ``null`` and are
treated as "unknown" (no rule fires) rather than guessed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import cache
from importlib import resources
from pathlib import Path

DEFAULT_TARGET = "esp32"

TARGET_PATTERN = re.compile(r'^CONFIG_IDF_TARGET="?(\w+)"?\s*$', re.MULTILINE)


@dataclass(frozen=True)
class Chip:
    """GPIO and toolchain facts for one ESP-IDF target."""

    target: str
    name: str
    arch: str
    toolchain_prefix: str
    gpios: frozenset[int]
    input_only: frozenset[int] = frozenset()
    no_internal_pull: frozenset[int] = frozenset()
    strapping: dict[int, str] = field(default_factory=dict)
    flash_spi: frozenset[int] = frozenset()
    psram_pins: frozenset[int] = frozenset()
    psram_condition: str = ""
    usb_jtag: frozenset[int] = frozenset()
    jtag: frozenset[int] = frozenset()
    uart0_tx: int | None = None
    uart0_rx: int | None = None
    adc1: dict[int, int] = field(default_factory=dict)
    adc2: dict[int, int] = field(default_factory=dict)
    conditional: dict[int, str] = field(default_factory=dict)
    adc2_wifi_conflict: str | None = None
    has_wifi: bool = False
    notes: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()

    @property
    def adc2_wifi_policy(self) -> str | None:
        """How ADC2 interacts with Wi-Fi, derived from the cited ESP-IDF wording.

        ``forbidden``: docs say ADC2 pins cannot be used while Wi-Fi is used (classic ESP32).
        ``arbitrated``: the driver arbitrates; reads can fail while Wi-Fi holds ADC2.
        ``None``: no ADC2, or no documented interaction.
        """
        text = (self.adc2_wifi_conflict or "").lower()
        if not self.adc2 or not text:
            return None
        return "forbidden" if "cannot be used" in text else "arbitrated"

    @property
    def adc2_channel_to_gpio(self) -> dict[int, int]:
        """Map ADC2 channel number to GPIO number."""
        return {ch: gpio for gpio, ch in self.adc2.items()}


def _ints(values: list | None) -> frozenset[int]:
    return frozenset(int(v) for v in values or [])


def _int_map(mapping: dict | None) -> dict[int, int]:
    return {int(k): int(v) for k, v in (mapping or {}).items() if v is not None}


def _from_json(data: dict) -> Chip:
    psram = data.get("psram") or {}
    usb = data.get("usb_jtag") or {}
    uart0 = data.get("uart0_default") or {}
    return Chip(
        target=data["target"],
        name=data["name"],
        arch=data["arch"],
        toolchain_prefix=data["toolchain_prefix"],
        gpios=_ints(data.get("gpios")),
        input_only=_ints(data.get("input_only")),
        no_internal_pull=_ints(data.get("no_internal_pull")),
        strapping={int(k): v for k, v in (data.get("strapping") or {}).items()},
        flash_spi=_ints(data.get("flash_spi")),
        psram_pins=_ints(psram.get("pins")),
        psram_condition=psram.get("condition") or "",
        usb_jtag=_ints(usb.get("pins")),
        jtag=_ints(data.get("jtag")),
        uart0_tx=uart0.get("tx"),
        uart0_rx=uart0.get("rx"),
        adc1=_int_map(data.get("adc1")),
        adc2=_int_map(data.get("adc2")),
        conditional={int(k): v for k, v in (data.get("conditional") or {}).items()},
        adc2_wifi_conflict=data.get("adc2_wifi_conflict"),
        has_wifi=bool(data.get("has_wifi")),
        notes=tuple(data.get("notes") or ()),
        sources=tuple(data.get("sources") or ()),
    )


def _chip_dir():
    return resources.files("esp32_workbench") / "data" / "chips"


def list_targets() -> list[str]:
    """Return all targets that have a chip file, sorted."""
    return sorted(p.name[:-5] for p in _chip_dir().iterdir() if p.name.endswith(".json"))


@cache
def load_chip(target: str = DEFAULT_TARGET) -> Chip:
    """Load the chip definition for an ESP-IDF target name (e.g. ``esp32s3``).

    Raises:
        ValueError: If the target is unknown.
    """
    normalized = target.lower().replace("-", "").replace("_", "")
    path = _chip_dir() / f"{normalized}.json"
    if not path.is_file():
        raise ValueError(f"Unknown target '{target}'. Known targets: {', '.join(list_targets())}")
    return _from_json(json.loads(path.read_text(encoding="utf-8")))


def detect_target(project_dir: Path) -> str | None:
    """Read ``CONFIG_IDF_TARGET`` from a project's sdkconfig files.

    Checks ``sdkconfig`` first (the active build), then ``sdkconfig.defaults``.
    Walks up from ``project_dir`` so a ``main/`` subfolder also works.
    """
    start = project_dir if project_dir.is_dir() else project_dir.parent
    for directory in (start, *start.resolve().parents):
        for name in ("sdkconfig", "sdkconfig.defaults"):
            candidate = directory / name
            if candidate.is_file():
                match = TARGET_PATTERN.search(
                    candidate.read_text(encoding="utf-8", errors="ignore")
                )
                if match:
                    return match.group(1).lower()
        if (directory / ".git").exists():
            break
    return None


def resolve_chip(target: str | None, project_dir: Path | None = None) -> Chip:
    """Pick a chip from an explicit target, the project's sdkconfig, or the default."""
    if target:
        return load_chip(target)
    if project_dir is not None:
        detected = detect_target(project_dir)
        if detected:
            return load_chip(detected)
    return load_chip(DEFAULT_TARGET)

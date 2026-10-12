"""Single-core SRAM capacity; addresses are always 32-bit words."""
from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareConfig:
    per_core_usable_bytes: int = 131072

    def __post_init__(self):
        if type(self.per_core_usable_bytes) is not int or self.per_core_usable_bytes <= 0:
            raise ValueError('per_core_usable_bytes must be a positive integer')


DEFAULT_HARDWARE = HardwareConfig()

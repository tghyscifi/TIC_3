"""Only a complete sequence on a single core is currently supported."""
from dataclasses import dataclass
from .address_allocator import allocate_addresses, allocate_addresses_for_stage
from .hardware_config import DEFAULT_HARDWARE
from .diagnostics import CompilerError


@dataclass(frozen=True)
class Stage:
    block_indices: tuple
    size_x: int = 1
    size_y: int = 1
    core_offset: tuple = (0, 0)

    def __post_init__(self):
        object.__setattr__(self, 'block_indices', tuple(self.block_indices))
        object.__setattr__(self, 'core_offset', tuple(self.core_offset))


def plan_stages(seq, hw=DEFAULT_HARDWARE):
    assignments, _, _ = allocate_addresses(seq)
    # Count occupied allocations including alignment gaps, but not the bank gap.
    mem0_end = max([a.output_addr+a.output_words for a in assignments] +
                   [a.scratch_addr+a.scratch_words for a in assignments] +
                   [a.bias_addr+a.bias_words for a in assignments if a.weight_addr < 0x4000])
    mem1_end = max([0x4000] + [a.bias_addr+a.bias_words for a in assignments if a.weight_addr >= 0x4000])
    if 4*(mem0_end+mem1_end-0x4000) > hw.per_core_usable_bytes:
        raise CompilerError('Sequence exceeds configured single-core capacity', code='MEMORY')
    return [Stage(tuple(a.block_index for a in assignments))]

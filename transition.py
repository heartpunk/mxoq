"""Edge schema + mnemonic normalization for the LTS extractor.

A `Transition` is the unit of LTS output: each basic block produces
one `Transition` per outgoing control-flow edge. The extractor
serializes the list as JSON.

`VexTranslationError` is the base exception class for translation
failures (subclassed e.g. by `AngrBackendNotImplemented`).

`normalize_mnemonic` canonicalizes capstone-emitted instruction
mnemonics (Intel syntax + AT&T width-suffixed forms) against a
single set of internal names used by the manifest gating and the
flag-effect tables.

`MemWriteOp` captures provenance for one in-block memory write
operation (sophie + codex 2026-04-27 Phase B design): the angr
mem_write event width is authoritative for "single n-byte machine
write," so a byte-write loop produces N entries with `nbytes=1`
each — never confusable with one N-byte write at the same address.
This metadata drives compact `(store_le_N M A V)` serialization
without falling back to syntactic AST pattern matching, which is
unsound for that distinction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import z3

import x86_flags


class VexTranslationError(Exception):
    """Raised by either backend when a block's shape isn't translatable.

    Subclasses include `AngrBackendNotImplemented` (raised by
    `angr_backend.py`) and `UnmodeledFlagWriter` (raised by
    `vex_to_smt.py`).
    """


@dataclass(frozen=True)
class MemWriteOp:
    """Provenance for one in-block memory write event.

    Recorded at extraction time by `angr_backend._mem_write_timeline_hook`
    (the angr `mem_write` BP_AFTER hook). The `nbytes` field is the
    angr-reported instruction-width — authoritative for "this came
    from a single n-byte machine write." A byte-write loop produces
    a chronological sequence of `MemWriteOp` records each with
    `nbytes=1`, distinguishing them from a single n-byte write at
    the same base address (which would appear as one record with
    `nbytes=N`).

    Fields:
      - `kind`: discriminator for future extensibility. Initially the
        only value is "store_le_n" (single n-byte little-endian write
        from one machine instruction). Future kinds may include
        "memcpy_n", etc.
      - `nbytes`: width of the write in bytes (1, 2, 4, 8 for amd64).
      - `addr_z3`: the write address as a z3 BV expression. Lives in
        `smt_emit.LTS_EXTRACTION_CTX` (post `_to_z3` translation).
      - `val_z3`: the write value as a z3 BV expression of width
        `8 * nbytes`. Same ctx as `addr_z3`.
      - `prev_mem_id`: `z3.ExprRef.get_id()` of the in-block
        `_qfabv_mem_expr` BEFORE this write was applied. Used by the
        block-end ID-chain verifier to confirm the timeline links
        correctly across writes.
      - `result_mem_id`: `get_id()` AFTER the write was applied.
      - `pc`: instruction-pointer value (block-local PC) for this
        write. Useful for provenance linking from compact LTS rows
        back to source instructions.

    Frozen so dataclass instances are hashable / safe to share across
    `state.globals` copies that angr's symbolic execution makes.
    """
    kind: str
    nbytes: int
    addr_z3: z3.BitVecRef
    val_z3: z3.BitVecRef
    prev_mem_id: int
    result_mem_id: int
    pc: int


@dataclass
class Transition:
    """One outgoing control-flow edge from a basic block.

    Fields:
      - `kind`: one of "exit" | "default" | "call" | "primitive".
        In practice the angr backend emits only "exit" and
        "default"; "call" and "primitive" are reserved for the
        multi-function extractor's call-graph closure.
      - `guard`: 1-bit z3.BoolRef. The condition under which this
        edge fires.
      - `sub`: parallel state-update map (var name → SMT-LIB
        z3 expression). Keys include canonical GPRs ("rax", "rip"),
        per-flag 1-bit names ("rflags_CF", ...), and "mem" for
        the byte-array memory.
      - `tgt`: concrete successor address, or None for symbolic
        targets (ret edges).
      - `tgt_smt`: SMT-LIB expression for symbolic targets — set
        when `tgt is None` and the block ends in a ret or other
        symbolic indirect control transfer.
      - `jumpkind`: raw VEX jumpkind ("Ijk_Boring", "Ijk_Call",
        "Ijk_Ret", ...).
      - `callee_sym`: callee symbol name for Ijk_Call edges.
      - `flag_reads`: list of (insn_addr, mnem, flags_read_set)
        tuples. Vestigial — neither backend populates it on
        emitted edges, no downstream consumer reads it.
      - `exit_flag_status`: per-flag DEFINED/UNDEFINED/UNCHANGED
        status at block exit. Computed via a flow-sensitive
        capstone-driven dataflow pass; the Lean loader uses this
        to reject downstream blocks that read undefined flags.
      - `mem_timeline`: chronological list of `MemWriteOp` for every
        in-block memory write. Populated by the angr backend's
        `_mem_write_timeline_hook`. Empty for the vex backend,
        which doesn't track per-write provenance — vex-backed edges
        fall back to legacy verbose-chain serialization for the
        `mem` field.
    """
    kind: str
    guard: z3.BoolRef
    sub: dict[str, z3.ExprRef]
    tgt: int | None
    tgt_smt: z3.BitVecRef | None = None
    jumpkind: str = "Ijk_Boring"
    callee_sym: str | None = None
    flag_reads: list[tuple[int, str, set[str]]] = field(default_factory=list)
    exit_flag_status: dict[str, str] = field(default_factory=dict)
    mem_timeline: list[MemWriteOp] = field(default_factory=list)


# ---------------------------------------------------------------------
# Mnemonic normalization
# ---------------------------------------------------------------------

# Stripping 'b'/'w'/'l'/'q' from the end of an AT&T mnemonic typically
# yields the canonical mnemonic. We accept either syntax (Intel /
# capstone default OR AT&T); internal lookups use the canonical form.
_WIDTH_SUFFIXES = ("b", "w", "l", "q")

# Canonical mnemonics the manifests and flag tables speak.
CANONICAL_MNEMONICS: set[str] = {
    # data movement
    "mov", "movzx", "movsx", "movzb", "movzw", "movsb", "movsw", "movq", "movl",
    "push", "pop", "lea",
    # control flow
    "call", "ret", "jmp", "nop",
    # conditional jumps (sourced from x86_flags.JCC_READS, which is the
    # authoritative set of Jcc mnemonics with their flag-read sets)
    *x86_flags.JCC_READS.keys(),
    # arithmetic / logic
    "add", "sub", "mul", "imul", "div", "idiv", "neg",
    "and", "or", "xor", "not",
    "shl", "shr", "sar", "rol", "ror",
    "cmp", "test",
    "inc", "dec",
    "adc", "sbb",
    # byte-level variants that capstone sometimes emits separately
    "movzbl", "movzwl", "movsbl", "movswl", "movsbq", "movsxd",
}


def normalize_mnemonic(raw: str) -> str:
    """Map a capstone-emitted mnemonic to our canonical internal name.

    Rules:
      - If `raw` is already canonical, use it as-is.
      - Else, if stripping an AT&T width suffix yields a canonical
        name, use the stripped form.
      - Else return `raw` unchanged (caller will fail the manifest
        check, which is the desired behavior — unknown mnemonics
        should not be silently accepted).
    """
    if raw in CANONICAL_MNEMONICS:
        return raw
    if len(raw) >= 2 and raw[-1] in _WIDTH_SUFFIXES:
        stem = raw[:-1]
        if stem in CANONICAL_MNEMONICS:
            return stem
    return raw

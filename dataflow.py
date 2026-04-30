"""
Function-level live-flag dataflow pass.

Computes, for every basic block in a function, the entry flag status
(per-flag DEFINED/UNDEFINED/UNCHANGED). This is used to initialize each
block's LiveFlagTracker in the translator so cross-block undefined-flag
reads are detected.

Standard worklist fixed-point:
  entry(b) = meet over preds(b) of exit(pred)
  exit(b)  = apply(local_effect(b), entry(b))

local_effect(b) is computed ONCE per block via a light mnemonic pass
(no VEX translation, just capstone).
"""
from __future__ import annotations

from collections import defaultdict

import x86_flags as xf
from transition import normalize_mnemonic


def _immediate_count_for_shift(ins) -> int | None:
    """If ins is a shift/rotate with an immediate count operand, return the
    count's integer value. Otherwise (cl-register variant or operand missing)
    return None so callers fall back to the worst-case effect."""
    if ins is None or not getattr(ins, "operands", None) or len(ins.operands) < 2:
        return None
    op = ins.operands[1]
    # capstone x86 operand type for immediate is X86_OP_IMM = 2. Dodge the
    # import by checking via attribute (ins.operands[1].type enumeration).
    try:
        import capstone
        if op.type != capstone.x86.X86_OP_IMM:
            return None
    except ImportError:
        # If capstone somehow isn't available, fall back to worst case.
        return None
    return op.imm


def _x86_count_mask(count: int, width: int) -> int:
    """Same hardware mask as the translator: 5 bits for 8/16/32-bit ops,
    6 bits for 64-bit."""
    return count & (0x3F if width == 64 else 0x1F)


def _effective_rotate_count(count: int, width: int) -> int:
    """Full effective rotate count: mask, then reduce mod width."""
    return _x86_count_mask(count, width) % width


def _width_bits_for_insn(ins) -> int:
    """Get operand width in bits from capstone. Fallback 32 if unknown."""
    if ins is not None and getattr(ins, "operands", None):
        return ins.operands[0].size * 8
    return 32


def _effect_for_insn(mnem: str, ins) -> dict[str, str]:
    """Return {flag: DEFINED|UNDEFINED|UNCHANGED} for a single instruction."""
    # Non-flag-writing mnemonic (mov, load, push, jmp, etc.)
    if not xf.is_known_flag_writer(mnem):
        return {f: xf.UNCHANGED for f in xf.FLAGS}

    # Table-driven: directly from FLAG_TABLE.
    if mnem in xf.FLAG_TABLE:
        return {f: rule.effect for f, rule in xf.FLAG_TABLE[mnem].items()}

    # shl/rol: effect depends on the masked effective count. If the count is
    # an immediate operand, we can compute the exact effect. If it's in `cl`
    # (a variable), we fall back to the worst case.
    if mnem in ("shl", "rol"):
        raw_count = _immediate_count_for_shift(ins)
        if raw_count is None:
            # Variable count (`cl`) — worst case.
            if mnem == "shl":
                return {"CF": xf.UNDEFINED, "ZF": xf.UNDEFINED, "SF": xf.UNDEFINED,
                        "OF": xf.UNDEFINED, "PF": xf.UNDEFINED, "AF": xf.UNDEFINED}
            return {"CF": xf.UNDEFINED, "ZF": xf.UNCHANGED, "SF": xf.UNCHANGED,
                    "OF": xf.UNDEFINED, "PF": xf.UNCHANGED, "AF": xf.UNCHANGED}
        width = _width_bits_for_insn(ins)
        if mnem == "rol":
            effective = _effective_rotate_count(raw_count, width)
        else:
            effective = _x86_count_mask(raw_count, width)
        rules = xf.rules_for_shift_rotate(mnem, effective)
        if rules is None:
            # effective count = 0: flags UNCHANGED.
            return {f: xf.UNCHANGED for f in xf.FLAGS}
        return {f: rule.effect for f, rule in rules.items()}

    # Known flag-writer but unmodeled; by design we fail-fast in the
    # translator, so here we mark everything undefined to be safe if the
    # dataflow runs ahead of the translator check.
    return {f: xf.UNDEFINED for f in xf.FLAGS}


def _compose_effects(acc: dict[str, str], nxt: dict[str, str]) -> dict[str, str]:
    """Compose two effects left-to-right. `nxt` executes after `acc`."""
    result = {}
    for f in xf.FLAGS:
        if nxt[f] == xf.UNCHANGED:
            result[f] = acc[f]
        else:
            result[f] = nxt[f]  # DEFINED or UNDEFINED overrides prior
    return result


def block_effect(project, bb_addr: int) -> dict[str, str]:
    """Compute the composed flag effect for a whole block."""
    block = project.factory.block(bb_addr)
    insns = block.capstone.insns
    acc = {f: xf.UNCHANGED for f in xf.FLAGS}
    for ins in insns:
        mnem = normalize_mnemonic(ins.mnemonic)
        acc = _compose_effects(acc, _effect_for_insn(mnem, ins))
    return acc


def _apply_effect(entry: dict[str, str], effect: dict[str, str]) -> dict[str, str]:
    """Apply a block's effect to an entry state, producing exit state."""
    return _compose_effects(entry, effect)


def dataflow_entry_statuses(
    project,
    blocks: list[int],
    successors: dict[int, list[int]],
    entry_block: int,
    initial_entry: dict[str, str] | None = None,
) -> dict[int, dict[str, str]]:
    """Run worklist until convergence. Return per-block entry flag status.

    Args:
        project: angr Project
        blocks: all block entry addresses in the function
        successors: for each block addr, list of successor block addrs
          (only the ones WITHIN this function; external calls/syscalls are
          not successors for dataflow purposes here)
        entry_block: the function's entry block addr
        initial_entry: flag status at function entry (default: all UNCHANGED
          = assumed-defined-by-caller)

    Returns:
        {block_addr: entry_status_dict}
    """
    if initial_entry is None:
        initial_entry = {f: xf.UNCHANGED for f in xf.FLAGS}

    # Precompute each block's local effect.
    effects: dict[int, dict[str, str]] = {b: block_effect(project, b) for b in blocks}

    # Invert successors → predecessors.
    predecessors: dict[int, list[int]] = defaultdict(list)
    for b, succs in successors.items():
        for t in succs:
            predecessors[t].append(b)

    # Initialize.
    entry: dict[int, dict[str, str]] = {}
    exit_: dict[int, dict[str, str]] = {}
    for b in blocks:
        entry[b] = dict(initial_entry) if b == entry_block else {f: xf.UNDEFINED for f in xf.FLAGS}
        exit_[b] = _apply_effect(entry[b], effects[b])

    # Worklist fixed-point.
    worklist = list(blocks)
    iter_count = 0
    MAX_ITERS = len(blocks) * 20  # bound — CFG is finite and lattice height is 3
    while worklist:
        iter_count += 1
        if iter_count > MAX_ITERS:
            raise RuntimeError(
                f"dataflow did not converge after {iter_count} iterations — "
                "possible infinite loop in lattice?"
            )
        b = worklist.pop(0)
        preds = predecessors.get(b, [])
        if b == entry_block and not preds:
            new_entry = dict(initial_entry)
        elif not preds:
            # Unreachable block (no preds and not entry) — leave as all-UNDEFINED.
            new_entry = {f: xf.UNDEFINED for f in xf.FLAGS}
        else:
            new_entry = xf.meet(*(exit_[p] for p in preds))
            if b == entry_block:
                # Entry block also gets the initial-entry as an input.
                new_entry = xf.meet(new_entry, initial_entry)

        if new_entry != entry[b]:
            entry[b] = new_entry
            new_exit = _apply_effect(new_entry, effects[b])
            if new_exit != exit_[b]:
                exit_[b] = new_exit
                for t in successors.get(b, []):
                    if t not in worklist:
                        worklist.append(t)

    return entry

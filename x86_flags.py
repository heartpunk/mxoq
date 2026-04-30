"""
x86-64 flag semantics for the 30-instruction manifest.

Each flag (CF, ZF, SF, OF, PF, AF) for each flag-affecting instruction can
be:
  DEFINED    — we emit an explicit SMT formula for the post-value
  UNDEFINED  — per Intel SDM. Live-flag analysis fails if later read.
  UNCHANGED  — no flag-write emitted; the flag keeps its prior value.

All formulas operate on and return `z3.ExprRef`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import z3

import smt_emit as s

# Ctx-scoped constants (sophie option C, 2026-04-27): every minting helper
# requires explicit ctx; this module operates entirely in the canonical
# LTS extraction context.
_CTX = s.LTS_EXTRACTION_CTX


# ---------------------------------------------------------------------
# Flag-effect enums
# ---------------------------------------------------------------------

DEFINED = "defined"
UNDEFINED = "undefined"
UNCHANGED = "unchanged"

FLAGS = ["CF", "ZF", "SF", "OF", "PF", "AF"]


@dataclass
class FlagRule:
    """One flag's effect under an instruction.

    effect:  DEFINED | UNDEFINED | UNCHANGED
    formula: if DEFINED, a callable that takes
             (operands: dict[str, z3.BitVecRef], result: z3.BitVecRef, width: int)
             and returns a 1-bit z3.BitVecRef (the new flag value).
    """
    effect: str
    formula: Optional[Callable[[dict, z3.BitVecRef, int], z3.BitVecRef]] = None


def _zero_bit() -> z3.BitVecRef:
    return s.bvlit(1, 0, _CTX)


def _one_bit() -> z3.BitVecRef:
    return s.bvlit(1, 1, _CTX)


def _parity_flag(_operands: dict, result: z3.BitVecRef, _width: int) -> z3.BitVecRef:
    """PF = 1 iff the low 8 bits of result have even number of 1-bits."""
    lo8 = s.extract(7, 0, result)
    # XOR-reduce the 8 bits; result is 0 if even parity, 1 if odd.
    bits = [s.extract(i, i, lo8) for i in range(8)]
    xor_all = bits[0]
    for b in bits[1:]:
        xor_all = s.bvxor(xor_all, b)
    # PF = 1 if xor_all == 0 (even), else 0.
    return s.ite(s.bveq(xor_all, _zero_bit()), _one_bit(), _zero_bit())


def _zf(_operands: dict, result: z3.BitVecRef, width: int) -> z3.BitVecRef:
    return s.ite(s.bveq(result, s.bvlit(width, 0, _CTX)), _one_bit(), _zero_bit())


def _sf(_operands: dict, result: z3.BitVecRef, width: int) -> z3.BitVecRef:
    return s.extract(width - 1, width - 1, result)


def _cf_sub(operands: dict, _result: z3.BitVecRef, _width: int) -> z3.BitVecRef:
    """CF for sub/cmp = unsigned(lhs) < unsigned(rhs)."""
    return s.ite(s.bvult(operands["lhs"], operands["rhs"]), _one_bit(), _zero_bit())


def _cf_add(operands: dict, result: z3.BitVecRef, _width: int) -> z3.BitVecRef:
    """CF for add = (result unsigned-less-than lhs)."""
    return s.ite(s.bvult(result, operands["lhs"]), _one_bit(), _zero_bit())


def _of_sub(operands: dict, result: z3.BitVecRef, width: int) -> z3.BitVecRef:
    """OF for sub: sign(lhs) != sign(rhs) AND sign(lhs) != sign(result)."""
    lhs_sign = s.extract(width - 1, width - 1, operands["lhs"])
    rhs_sign = s.extract(width - 1, width - 1, operands["rhs"])
    res_sign = s.extract(width - 1, width - 1, result)
    cond = s.and_(
        s.not_(s.bveq(lhs_sign, rhs_sign)),
        s.not_(s.bveq(lhs_sign, res_sign)),
    )
    return s.ite(cond, _one_bit(), _zero_bit())


def _of_add(operands: dict, result: z3.BitVecRef, width: int) -> z3.BitVecRef:
    """OF for add: sign(lhs) == sign(rhs) AND sign(result) != sign(lhs)."""
    lhs_sign = s.extract(width - 1, width - 1, operands["lhs"])
    rhs_sign = s.extract(width - 1, width - 1, operands["rhs"])
    res_sign = s.extract(width - 1, width - 1, result)
    cond = s.and_(
        s.bveq(lhs_sign, rhs_sign),
        s.not_(s.bveq(res_sign, lhs_sign)),
    )
    return s.ite(cond, _one_bit(), _zero_bit())


def _const_zero(_operands: dict, _result: z3.BitVecRef, _width: int) -> z3.BitVecRef:
    return _zero_bit()


# ---------------------------------------------------------------------
# Shift-specific flag formulas (count must be known at emission time)
# ---------------------------------------------------------------------

def _shl_cf_const(operands: dict, _result: z3.BitVecRef, width: int) -> z3.BitVecRef:
    """CF after shll by a constant count: the LAST bit shifted out of the high end.
    For count >= 1, CF = bit(lhs, width - count). For count == 0 this rule
    shouldn't be invoked (flags UNCHANGED per SDM)."""
    count = operands["count"]
    if count == 0:
        raise ValueError("_shl_cf_const called with count=0")
    if count >= width:
        return _zero_bit()  # all bits shifted out; CF = 0
    bit_idx = width - count
    return s.extract(bit_idx, bit_idx, operands["lhs"])


def _shl_of_const(operands: dict, result: z3.BitVecRef, width: int) -> z3.BitVecRef:
    """OF after shll by count == 1: XOR of the top two bits of the RESULT.
    For count > 1 OF is undefined; our dispatch should not call this then."""
    count = operands["count"]
    if count != 1:
        raise ValueError("_shl_of_const is only defined for count==1")
    top = s.extract(width - 1, width - 1, result)
    next_top = s.extract(width - 2, width - 2, result)
    return s.bvxor(top, next_top)


def _shr_cf_const(operands: dict, _result: z3.BitVecRef, width: int) -> z3.BitVecRef:
    """CF after shrl by a constant count >= 1: the last bit shifted out the
    LOW end, which is bit(lhs, count - 1). For count == 0 unchanged; for
    count >= width, CF = 0 (all bits shifted out)."""
    count = operands["count"]
    if count == 0:
        raise ValueError("_shr_cf_const called with count=0")
    if count > width:
        return _zero_bit()
    bit_idx = count - 1
    return s.extract(bit_idx, bit_idx, operands["lhs"])


def _shr_of_const(operands: dict, _result: z3.BitVecRef, width: int) -> z3.BitVecRef:
    """OF after shrl by count == 1: the MSB of the ORIGINAL operand (what
    used to be the sign bit before the shift). For count > 1 OF is
    undefined; our dispatch should not call this then."""
    count = operands["count"]
    if count != 1:
        raise ValueError("_shr_of_const is only defined for count==1")
    return s.extract(width - 1, width - 1, operands["lhs"])


def _sar_cf_const(operands: dict, _result: z3.BitVecRef, width: int) -> z3.BitVecRef:
    """CF after sarl by a constant count >= 1: same bit as shr — the
    last bit shifted out the low end = bit(lhs, count - 1). For
    count >= width, CF = the sign bit (which is all that's left to
    'shift out' — arithmetic right fill is sign-preserving). For
    count == 0 unchanged (caller skips)."""
    count = operands["count"]
    if count == 0:
        raise ValueError("_sar_cf_const called with count=0")
    if count >= width:
        # All original bits shifted out except sign; CF = sign bit.
        return s.extract(width - 1, width - 1, operands["lhs"])
    bit_idx = count - 1
    return s.extract(bit_idx, bit_idx, operands["lhs"])


def _sar_of_const(operands: dict, _result: z3.BitVecRef, _width: int) -> z3.BitVecRef:
    """OF after sarl by count == 1: always 0 (signed shift-right can't
    change the sign bit, so no overflow). For count > 1 OF is
    undefined; our dispatch should not call this then."""
    count = operands["count"]
    if count != 1:
        raise ValueError("_sar_of_const is only defined for count==1")
    return _zero_bit()


def _rol_cf_const(operands: dict, result: z3.BitVecRef, _width: int) -> z3.BitVecRef:
    """CF after roll by a constant count >= 1: equal to bit 0 of the result
    (the bit rotated from high end back to low end)."""
    count = operands["count"]
    if count == 0:
        raise ValueError("_rol_cf_const called with count=0")
    return s.extract(0, 0, result)


def _rol_of_const(operands: dict, result: z3.BitVecRef, width: int) -> z3.BitVecRef:
    """OF after roll by count == 1: XOR of CF (new bit 0) and the top bit of
    the result."""
    count = operands["count"]
    if count != 1:
        raise ValueError("_rol_of_const is only defined for count==1")
    top = s.extract(width - 1, width - 1, result)
    bit0 = s.extract(0, 0, result)
    return s.bvxor(top, bit0)


# ---------------------------------------------------------------------
# Flag-effect tables
# ---------------------------------------------------------------------

# cmp: subtract + discard. CF/ZF/SF/OF/PF defined, AF undefined.
_cmp_rules = {
    "CF": FlagRule(DEFINED, _cf_sub),
    "ZF": FlagRule(DEFINED, _zf),
    "SF": FlagRule(DEFINED, _sf),
    "OF": FlagRule(DEFINED, _of_sub),
    "PF": FlagRule(DEFINED, _parity_flag),
    "AF": FlagRule(UNDEFINED),
}

_sub_rules = dict(_cmp_rules)

_add_rules = {
    "CF": FlagRule(DEFINED, _cf_add),
    "ZF": FlagRule(DEFINED, _zf),
    "SF": FlagRule(DEFINED, _sf),
    "OF": FlagRule(DEFINED, _of_add),
    "PF": FlagRule(DEFINED, _parity_flag),
    "AF": FlagRule(UNDEFINED),
}

# test / and / or / xor: CF=0, OF=0, ZF/SF/PF defined, AF undefined.
_bitop_rules = {
    "CF": FlagRule(DEFINED, _const_zero),
    "ZF": FlagRule(DEFINED, _zf),
    "SF": FlagRule(DEFINED, _sf),
    "OF": FlagRule(DEFINED, _const_zero),
    "PF": FlagRule(DEFINED, _parity_flag),
    "AF": FlagRule(UNDEFINED),
}

# shl (and synonyms): count-dependent. Our table applies to count >= 1 case.
# For count == 0, caller must detect and skip (flags UNCHANGED). For count == 1,
# all defined. For count > 1, OF is UNDEFINED (caller uses _shl_rules_multi).
_shl_rules_count_eq_1 = {
    "CF": FlagRule(DEFINED, _shl_cf_const),
    "ZF": FlagRule(DEFINED, _zf),
    "SF": FlagRule(DEFINED, _sf),
    "OF": FlagRule(DEFINED, _shl_of_const),
    "PF": FlagRule(DEFINED, _parity_flag),
    "AF": FlagRule(UNDEFINED),
}

_shl_rules_count_gt_1 = {
    "CF": FlagRule(DEFINED, _shl_cf_const),
    "ZF": FlagRule(DEFINED, _zf),
    "SF": FlagRule(DEFINED, _sf),
    "OF": FlagRule(UNDEFINED),
    "PF": FlagRule(DEFINED, _parity_flag),
    "AF": FlagRule(UNDEFINED),
}

# shr (logical shift right): count-dependent, same shape as shl with
# different CF/OF formulas. For count == 0 flags UNCHANGED (caller skips).
# For count == 1, all flags defined. For count > 1, OF is UNDEFINED per
# SDM (and AF is UNDEFINED regardless).
_shr_rules_count_eq_1 = {
    "CF": FlagRule(DEFINED, _shr_cf_const),
    "ZF": FlagRule(DEFINED, _zf),
    "SF": FlagRule(DEFINED, _sf),
    "OF": FlagRule(DEFINED, _shr_of_const),
    "PF": FlagRule(DEFINED, _parity_flag),
    "AF": FlagRule(UNDEFINED),
}

_shr_rules_count_gt_1 = {
    "CF": FlagRule(DEFINED, _shr_cf_const),
    "ZF": FlagRule(DEFINED, _zf),
    "SF": FlagRule(DEFINED, _sf),
    "OF": FlagRule(UNDEFINED),
    "PF": FlagRule(DEFINED, _parity_flag),
    "AF": FlagRule(UNDEFINED),
}


# sar (arithmetic shift right): same flag structure as shr with
# different CF/OF formulas. CF matches shr (bit shifted out the low
# end). OF differs — for count==1, sar always yields OF=0 (sign bit
# is preserved by arithmetic shift, so no overflow).
_sar_rules_count_eq_1 = {
    "CF": FlagRule(DEFINED, _sar_cf_const),
    "ZF": FlagRule(DEFINED, _zf),
    "SF": FlagRule(DEFINED, _sf),
    "OF": FlagRule(DEFINED, _sar_of_const),
    "PF": FlagRule(DEFINED, _parity_flag),
    "AF": FlagRule(UNDEFINED),
}

_sar_rules_count_gt_1 = {
    "CF": FlagRule(DEFINED, _sar_cf_const),
    "ZF": FlagRule(DEFINED, _zf),
    "SF": FlagRule(DEFINED, _sf),
    "OF": FlagRule(UNDEFINED),
    "PF": FlagRule(DEFINED, _parity_flag),
    "AF": FlagRule(UNDEFINED),
}


# rol: CF and OF defined for count == 1; OF undefined for count > 1.
# ZF/SF/PF are UNCHANGED by rotates per SDM.
_rol_rules_count_eq_1 = {
    "CF": FlagRule(DEFINED, _rol_cf_const),
    "ZF": FlagRule(UNCHANGED),
    "SF": FlagRule(UNCHANGED),
    "OF": FlagRule(DEFINED, _rol_of_const),
    "PF": FlagRule(UNCHANGED),
    "AF": FlagRule(UNCHANGED),
}

_rol_rules_count_gt_1 = {
    "CF": FlagRule(DEFINED, _rol_cf_const),
    "ZF": FlagRule(UNCHANGED),
    "SF": FlagRule(UNCHANGED),
    "OF": FlagRule(UNDEFINED),
    "PF": FlagRule(UNCHANGED),
    "AF": FlagRule(UNCHANGED),
}


_div_rules = {
    # Per Intel SDM: all of CF, OF, SF, ZF, AF, PF are undefined after
    # div / idiv. Downstream code must not read them; the extractor
    # fails fast if any consumer depends on a flag this instruction
    # nominally left undefined.
    "CF": FlagRule(UNDEFINED),
    "ZF": FlagRule(UNDEFINED),
    "SF": FlagRule(UNDEFINED),
    "OF": FlagRule(UNDEFINED),
    "PF": FlagRule(UNDEFINED),
    "AF": FlagRule(UNDEFINED),
}


_imul_rules = {
    # Per Intel SDM: CF and OF ARE defined — set when the signed
    # result differs in size from the destination operand (i.e., the
    # full product doesn't fit in the destination). SF/ZF/AF/PF are
    # undefined.
    #
    # TODO: model CF/OF precisely as "upper half ≠ sign-extend of
    # lower half" (one-operand form) or "result differs from dst-width
    # truncated form" (two/three-operand forms). For now, UNDEFINED
    # all — works for code that doesn't branch on imul overflow, which
    # is the common case. If the extractor fail-fasts on a real
    # CF/OF read after imul, that's the signal to add the precise
    # model.
    "CF": FlagRule(UNDEFINED),
    "ZF": FlagRule(UNDEFINED),
    "SF": FlagRule(UNDEFINED),
    "OF": FlagRule(UNDEFINED),
    "PF": FlagRule(UNDEFINED),
    "AF": FlagRule(UNDEFINED),
}


FLAG_TABLE: dict[str, dict[str, FlagRule]] = {
    "cmp": _cmp_rules,
    "sub": _sub_rules,
    "add": _add_rules,
    "test": _bitop_rules,
    "and": _bitop_rules,
    "or": _bitop_rules,
    "xor": _bitop_rules,
    "div": _div_rules,
    "imul": _imul_rules,
    # shl/shr/rol are looked up via rules_for_shift_rotate() because the
    # rule set depends on the count value (which must be a compile-time
    # constant; we fail extraction otherwise).
}


def rules_for_shift_rotate(mnem: str, count: int) -> Optional[dict[str, FlagRule]]:
    """Return the per-flag rules for shll / roll at the given concrete count.
    Returns None if count == 0 (flags UNCHANGED across the instruction)."""
    if count == 0:
        return None
    if mnem == "shl":
        return _shl_rules_count_eq_1 if count == 1 else _shl_rules_count_gt_1
    if mnem == "shr":
        return _shr_rules_count_eq_1 if count == 1 else _shr_rules_count_gt_1
    if mnem == "sar":
        return _sar_rules_count_eq_1 if count == 1 else _sar_rules_count_gt_1
    if mnem == "rol":
        return _rol_rules_count_eq_1 if count == 1 else _rol_rules_count_gt_1
    raise ValueError(f"rules_for_shift_rotate: unexpected mnemonic {mnem!r}")


# ---------------------------------------------------------------------
# Modeled-mnemonic check (fail-fast on unmodeled flag-setters)
# ---------------------------------------------------------------------

# Every x86 mnemonic that WRITES any flag. If a mnemonic is here but not
# modeled (via FLAG_TABLE or rules_for_shift_rotate), the extractor must
# fail-fast — silent flag preservation would be semantically wrong.
KNOWN_FLAG_WRITERS: set[str] = {
    "cmp", "sub", "add", "test", "and", "or", "xor",
    "shl", "shr", "sar", "rol", "ror",
    "inc", "dec", "neg",
    "adc", "sbb",
    "mul", "imul", "div", "idiv",
    "cmp", "cmpxchg", "xadd",
    "bt", "bts", "btr", "btc",
}


def is_modeled_flag_writer(mnem: str) -> bool:
    """True iff this mnemonic's flag semantics are fully modeled by our
    FLAG_TABLE or shift/rotate path. Used by the extractor to decide
    whether to decompose flags OR fail the block."""
    if mnem in FLAG_TABLE:
        return True
    if mnem in ("shl", "shr", "sar", "rol"):
        return True  # modeled via rules_for_shift_rotate
    return False


def is_known_flag_writer(mnem: str) -> bool:
    """True iff this mnemonic WRITES flags in real hardware. If this is true
    but is_modeled_flag_writer is false, the extractor must fail-fast."""
    return mnem in KNOWN_FLAG_WRITERS


# ---------------------------------------------------------------------
# Jcc flag reads
# ---------------------------------------------------------------------

JCC_READS: dict[str, set[str]] = {
    "je": {"ZF"}, "jz": {"ZF"},
    "jne": {"ZF"}, "jnz": {"ZF"},
    "js": {"SF"}, "jns": {"SF"},
    "jc": {"CF"}, "jnc": {"CF"},
    "jo": {"OF"}, "jno": {"OF"},
    "jp": {"PF"}, "jpe": {"PF"}, "jnp": {"PF"}, "jpo": {"PF"},
    "ja": {"CF", "ZF"}, "jnbe": {"CF", "ZF"},
    "jae": {"CF"}, "jnb": {"CF"},
    "jb": {"CF"}, "jnae": {"CF"},
    "jbe": {"CF", "ZF"}, "jna": {"CF", "ZF"},
    "jg": {"ZF", "SF", "OF"}, "jnle": {"ZF", "SF", "OF"},
    "jge": {"SF", "OF"}, "jnl": {"SF", "OF"},
    "jl": {"SF", "OF"}, "jnge": {"SF", "OF"},
    "jle": {"ZF", "SF", "OF"}, "jng": {"ZF", "SF", "OF"},
}


def jcc_reads(mnem: str) -> set[str]:
    if mnem not in JCC_READS:
        raise ValueError(f"unknown Jcc mnemonic: {mnem!r}")
    return JCC_READS[mnem]


def jcc_guard_smt(mnem: str, flag_exprs: dict[str, z3.BitVecRef]) -> z3.BoolRef:
    """Build the SMT bool guard for a conditional jump. `flag_exprs` maps
    flag name → 1-bit z3 expression for the current value."""
    one = s.bvlit(1, 1, _CTX)

    def is_set(f: str) -> z3.BoolRef:
        return s.bveq(flag_exprs[f], one)

    if mnem in ("je", "jz"):
        return is_set("ZF")
    if mnem in ("jne", "jnz"):
        return s.not_(is_set("ZF"))
    if mnem == "js":
        return is_set("SF")
    if mnem == "jns":
        return s.not_(is_set("SF"))
    if mnem == "jc":
        return is_set("CF")
    if mnem == "jnc":
        return s.not_(is_set("CF"))
    if mnem == "jo":
        return is_set("OF")
    if mnem == "jno":
        return s.not_(is_set("OF"))
    if mnem in ("jp", "jpe"):
        return is_set("PF")
    if mnem in ("jnp", "jpo"):
        return s.not_(is_set("PF"))
    if mnem in ("ja", "jnbe"):
        return s.and_(s.not_(is_set("CF")), s.not_(is_set("ZF")))
    if mnem in ("jae", "jnb"):
        return s.not_(is_set("CF"))
    if mnem in ("jb", "jnae"):
        return is_set("CF")
    if mnem in ("jbe", "jna"):
        return s.or_(is_set("CF"), is_set("ZF"))
    if mnem in ("jg", "jnle"):
        return s.and_(s.not_(is_set("ZF")), s.bveq(flag_exprs["SF"], flag_exprs["OF"]))
    if mnem in ("jge", "jnl"):
        return s.bveq(flag_exprs["SF"], flag_exprs["OF"])
    if mnem in ("jl", "jnge"):
        return s.not_(s.bveq(flag_exprs["SF"], flag_exprs["OF"]))
    if mnem in ("jle", "jng"):
        return s.or_(is_set("ZF"), s.not_(s.bveq(flag_exprs["SF"], flag_exprs["OF"])))
    raise ValueError(f"no guard translation for Jcc {mnem!r}")


# ---------------------------------------------------------------------
# Block-local live-flag tracker
# ---------------------------------------------------------------------

class LiveFlagError(Exception):
    pass


class LiveFlagTracker:
    """Tracks per-flag defined/undefined/unchanged-since-entry status through
    a single block. The `entry` state at construction reflects the function-
    level live-flag pass's meet over predecessor blocks — if any predecessor
    could leave a flag undefined, that flag starts as UNDEFINED here."""

    def __init__(self, entry_status: Optional[dict[str, str]] = None):
        if entry_status is None:
            self.status: dict[str, str] = {f: UNCHANGED for f in FLAGS}
        else:
            self.status = dict(entry_status)

    def snapshot(self) -> dict[str, str]:
        return dict(self.status)

    def apply_write(self, flag: str, effect: str) -> None:
        if flag not in FLAGS:
            raise ValueError(f"unknown flag {flag!r}")
        if effect not in (DEFINED, UNDEFINED, UNCHANGED):
            raise ValueError(f"bad effect {effect!r}")
        if effect == UNCHANGED:
            return
        self.status[flag] = effect

    def check_read(self, flag: str, addr: int, mnem: str) -> None:
        if flag not in FLAGS:
            raise LiveFlagError(f"unknown flag read {flag!r}")
        if self.status[flag] == UNDEFINED:
            raise LiveFlagError(
                f"instruction 0x{addr:x} ({mnem}) reads "
                f"flag {flag} which is undefined at this point"
            )

    def check_reads(self, flags: set[str], addr: int, mnem: str) -> None:
        for f in flags:
            self.check_read(f, addr, mnem)


def meet(*statuses: dict[str, str]) -> dict[str, str]:
    """Dataflow meet over flag-status dicts.

    Order: UNDEFINED ⊑ UNCHANGED ⊑ DEFINED (where meet goes DOWN toward
    UNDEFINED — pessimistic). So:
      - any input UNDEFINED → result UNDEFINED
      - all inputs DEFINED → DEFINED
      - mixed DEFINED / UNCHANGED → UNCHANGED (since "unchanged since entry"
        means the flag has its pre-entry value, which we consider presumed-
        defined for the purposes of reads within the function; the dataflow
        pass must separately verify that the caller provides a defined flag)
    """
    if not statuses:
        return {f: UNCHANGED for f in FLAGS}
    result = {}
    for f in FLAGS:
        vals = {st[f] for st in statuses}
        if UNDEFINED in vals:
            result[f] = UNDEFINED
        elif vals == {DEFINED}:
            result[f] = DEFINED
        else:
            result[f] = UNCHANGED
    return result

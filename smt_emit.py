"""
SMT-LIB S-expression emission, backed by z3.

All builders return `z3.ExprRef`. Serialization to an SMT-LIB string happens
only at the top-level boundary (when writing LTS JSON) via `to_smtlib(expr)`.

Rationale for z3-backing (vs. hand-rolled strings/AST):
- concat / extract arity + width correctness is checked by z3 at build time.
- `z3.substitute` handles the parent-register-reference substitution needed
  by partial-register compose without regex gymnastics.
- `z3.simplify` can be applied opportunistically to shorten formulas.
- Output via `expr.sexpr()` is guaranteed to be valid SMT-LIB 2.

Naming conventions (v4.2 design §7):
- registers: rax rcx rdx rbx rsp rbp rsi rdi r8..r15 rip   (BV 64)
- flags:     rflags_CF rflags_ZF rflags_SF rflags_OF rflags_PF rflags_AF  (BV 1)
- memory:    mem : (Array (BV 64) (BV 8))

Callers get these canonical z3 variables via `reg64(name, ctx)`,
`flag(f, ctx)`, `mem(ctx)`. Every primitive-minting helper takes the
target z3.Context as a required positional argument with no default —
see `LTS_EXTRACTION_CTX` below.
"""
from __future__ import annotations

from functools import lru_cache

import z3


# Configure z3's pretty printer to introduce `let` bindings for shared
# subterms with AST size >= 4 (default is 10, which is too high to catch
# common VEX flag-formula sharing patterns like the
# `(bvadd #xfffffffb ((_ extract 31 0) rax))` subterm reused across
# rflags_PF bit extractions). At threshold 4 the pretty printer emits
# compact let-bound output equivalent in shape to what the cvc5 post-pass
# (removed earlier in this commit chain) used to produce.
#
# This is a global z3 process-wide setting; we own the only `.sexpr()`
# caller in the parent codebase (`smt_emit.to_smtlib`), so the side
# effect is contained.
z3.set_param('pp.min_alias_size', 4)


# ---------------------------------------------------------------------
# Dedicated z3 context for LTS extraction
# ---------------------------------------------------------------------
#
# All z3 expressions that smt_emit constructs live in this context.
# claripy / angr default to `z3.main_ctx()`; isolating our work in a
# separate context prevents declaration leakage, model-extraction
# crosstalk, and accidental ctx mixing.
#
# At the angr→us boundary, callers route claripy ASTs through this
# context via `_BZ3.convert(claripy_ast).translate(LTS_EXTRACTION_CTX)`
# (see angr_backend `_to_z3`).
#
# All primitive-minting helpers in this module REQUIRE an explicit
# `ctx` argument with no default. This is by design (sophie 2026-04-27,
# option C: "force ctx, no default"): a default would silently route
# constructions in the wrong context if anyone added a second context
# in the future. z3 raises `Z3Exception: context mismatch` on cross-ctx
# operations, so any drift surfaces loudly. The strict-arg discipline
# ensures the drift is detected at construction time, not at first op.
#
# Composite helpers that take z3 inputs (bvadd, concat, store_le, etc.)
# derive ctx from their inputs and don't take a `ctx` argument.
LTS_EXTRACTION_CTX = z3.Context()


# ---------------------------------------------------------------------
# Canonical variables
# ---------------------------------------------------------------------

@lru_cache(maxsize=None)
def reg64(name: str, ctx: z3.Context) -> z3.BitVecRef:
    """Return the canonical 64-bit register variable in `ctx`."""
    return z3.BitVec(name, 64, ctx=ctx)


@lru_cache(maxsize=None)
def flag(name: str, ctx: z3.Context) -> z3.BitVecRef:
    """Return the canonical 1-bit flag variable in `ctx`. Name should be 'CF' / 'ZF' etc."""
    return z3.BitVec(f"rflags_{name}", 1, ctx=ctx)


@lru_cache(maxsize=None)
def mem(ctx: z3.Context) -> z3.ArrayRef:
    """Return the canonical byte-addressable memory array in `ctx`."""
    addr_sort = z3.BitVecSort(64, ctx=ctx)
    byte_sort = z3.BitVecSort(8, ctx=ctx)
    return z3.Array("mem", addr_sort, byte_sort)


@lru_cache(maxsize=None)
def tmp(idx: int, width: int, ctx: z3.Context) -> z3.BitVecRef:
    """Block-local fresh tmp variable in `ctx`. Not used in LTS output
    (tmps are inlined), but available for constructing intermediate
    terms during translation."""
    return z3.BitVec(f"tmp_{idx}", width, ctx=ctx)


# ---------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------

def bvlit(width: int, value: int, ctx: z3.Context) -> z3.BitVecNumRef:
    """A bitvector literal of `width` bits with value `value` in `ctx`.
    z3 normalizes to unsigned two's complement internally."""
    return z3.BitVecVal(value, width, ctx=ctx)


def bvadd(a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BitVecRef:
    return a + b


def bvsub(a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BitVecRef:
    return a - b


def bvmul(a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BitVecRef:
    return a * b


def bvand(a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BitVecRef:
    return a & b


def bvor(a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BitVecRef:
    return a | b


def bvxor(a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BitVecRef:
    return a ^ b


def bvnot(a: z3.BitVecRef) -> z3.BitVecRef:
    return ~a


def bvneg(a: z3.BitVecRef) -> z3.BitVecRef:
    return -a


def bvshl(a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BitVecRef:
    return a << b


def bvlshr(a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BitVecRef:
    return z3.LShR(a, b)


def bvashr(a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BitVecRef:
    return a >> b


def bvudiv(a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BitVecRef:
    return z3.UDiv(a, b)


def bvsdiv(a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BitVecRef:
    # z3 exposes signed division via `a / b` on BitVecRef (matches
    # SMT-LIB bvsdiv semantics); URem / SRem are separate named fns.
    return a / b


def bvurem(a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BitVecRef:
    return z3.URem(a, b)


def bvsrem(a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BitVecRef:
    return z3.SRem(a, b)


def bvrol_const(width: int, a: z3.BitVecRef, count: int) -> z3.BitVecRef:
    count = count % width
    if count == 0:
        return a
    return z3.RotateLeft(a, count)


def bvrol_var(width: int, a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BitVecRef:
    """Variable-count rotate-left implemented via shift+or.
    Assumes `b` is reduced to [0, width) before calling.
    The width-literal is constructed in `a.ctx` (derived from input)."""
    return (a << b) | z3.LShR(a, bvlit(width, width, a.ctx) - b)


def bveq(a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BoolRef:
    return a == b


def bvult(a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BoolRef:
    return z3.ULT(a, b)


def bvule(a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BoolRef:
    return z3.ULE(a, b)


def bvugt(a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BoolRef:
    return z3.UGT(a, b)


def bvuge(a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BoolRef:
    return z3.UGE(a, b)


def bvslt(a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BoolRef:
    return a < b


def bvsle(a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BoolRef:
    return a <= b


def bvsgt(a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BoolRef:
    return a > b


def bvsge(a: z3.BitVecRef, b: z3.BitVecRef) -> z3.BoolRef:
    return a >= b


def extract(hi: int, lo: int, a: z3.BitVecRef) -> z3.BitVecRef:
    return z3.Extract(hi, lo, a)


def zero_extend(amt: int, a: z3.BitVecRef) -> z3.BitVecRef:
    if amt == 0:
        return a
    return z3.ZeroExt(amt, a)


def sign_extend(amt: int, a: z3.BitVecRef) -> z3.BitVecRef:
    if amt == 0:
        return a
    return z3.SignExt(amt, a)


def concat(*args: z3.BitVecRef) -> z3.BitVecRef:
    """Concatenate 2+ bitvectors. z3.Concat accepts variadic args and produces
    a left-associative binary tree internally, which serializes to standard
    binary-nested SMT-LIB `(concat (concat a b) c)` or equivalent."""
    if len(args) < 2:
        raise ValueError("concat requires at least 2 arguments")
    return z3.Concat(*args)


def ite(cond: z3.BoolRef, t: z3.ExprRef, f: z3.ExprRef) -> z3.ExprRef:
    return z3.If(cond, t, f)


def and_(*args: z3.BoolRef) -> z3.BoolRef:
    """N-ary And. Requires len(args) >= 1 — for an explicit True
    constant, use `true_(ctx)`. The empty-args case was dropped to
    avoid silent-default-ctx construction (sophie option C, 2026-04-27).
    """
    if len(args) == 0:
        raise ValueError("and_() requires at least one argument; use true_(ctx) for an explicit True")
    if len(args) == 1:
        return args[0]
    return z3.And(*args)


def or_(*args: z3.BoolRef) -> z3.BoolRef:
    """N-ary Or. Requires len(args) >= 1 — for an explicit False
    constant, use `false_(ctx)`. The empty-args case was dropped to
    avoid silent-default-ctx construction (sophie option C, 2026-04-27).
    """
    if len(args) == 0:
        raise ValueError("or_() requires at least one argument; use false_(ctx) for an explicit False")
    if len(args) == 1:
        return args[0]
    return z3.Or(*args)


def not_(a: z3.BoolRef) -> z3.BoolRef:
    return z3.Not(a)


def select(arr: z3.ArrayRef, idx: z3.BitVecRef) -> z3.BitVecRef:
    return z3.Select(arr, idx)


def store(arr: z3.ArrayRef, idx: z3.BitVecRef, val: z3.BitVecRef) -> z3.ArrayRef:
    return z3.Store(arr, idx, val)


def true_(ctx: z3.Context) -> z3.BoolRef:
    """The Bool constant True in `ctx`. Replaces the prior module-level
    `TRUE` constant (which was bound to z3.main_ctx by default). With
    explicit ctx, callers can't accidentally route True into the wrong
    context (sophie option C, 2026-04-27)."""
    return z3.BoolVal(True, ctx=ctx)


def false_(ctx: z3.Context) -> z3.BoolRef:
    """The Bool constant False in `ctx`. See `true_` for rationale."""
    return z3.BoolVal(False, ctx=ctx)


# ---------------------------------------------------------------------
# Multi-byte little-endian load/store
# ---------------------------------------------------------------------
#
# `store_le(mem, addr, val, N)` and `load_le(mem, addr, N)` return
# fully expanded z3 ASTs (byte-by-byte store chain / concat-of-N-selects).
# The expanded form is what claripy's solver and z3.simplify reason
# about uniformly — naming the chain via `z3.Function` instead would
# break claripy's model-extraction path, which assumes
# uninterpreted-function constants are arity-0.
#
# `smt_defs_for_widths` and `discover_load_store_widths_in_text` (below) are
# helpers for a future serialization-time compaction pass that
# detects byte-by-byte store chains in emitted z3 ASTs and rewrites
# them to compact `(store_le_N ...)` / `(load_le_N ...)` calls in
# the SMT-LIB output, keeping the construction-time z3 ASTs
# fully-expanded (claripy-compatible) while compressing only the
# wire format. That compaction pass is queued; this commit lays the
# helper groundwork.


def load_le(mem_expr: z3.ArrayRef, addr: z3.BitVecRef, nbytes: int) -> z3.BitVecRef:
    """Read `nbytes` bytes from `mem_expr` at `addr`, little-endian.

    The returned bitvector's low byte is the byte at `addr`, next byte is at
    `addr+1`, etc. Implemented as repeated selects composed via binary concat.

    Ctx is derived from `mem_expr.ctx` for any new literals constructed
    along the way (offset bvlits); inputs `mem_expr` and `addr` must
    already share that ctx — z3 raises on cross-ctx ops if not.
    """
    if nbytes < 1:
        raise ValueError("load_le: nbytes must be >= 1")
    if nbytes == 1:
        return select(mem_expr, addr)
    # Build high-to-low so concat gives MSB-first.
    parts = []
    for i in range(nbytes - 1, -1, -1):
        parts.append(select(mem_expr, addr + bvlit(64, i, mem_expr.ctx)))
    return concat(*parts)


def store_le(mem_expr: z3.ArrayRef, addr: z3.BitVecRef, value: z3.BitVecRef, nbytes: int) -> z3.ArrayRef:
    """Store low `nbytes` of `value` into `mem_expr` at `addr..addr+nbytes-1`,
    little-endian. Returns the new memory.

    Ctx is derived from `mem_expr.ctx` for any new literals constructed
    along the way; inputs must share that ctx (z3 raises on cross-ctx
    ops if not).
    """
    if nbytes < 1:
        raise ValueError("store_le: nbytes must be >= 1")
    cur = mem_expr
    for i in range(nbytes):
        byte = extract(8 * i + 7, 8 * i, value)
        cur = store(cur, addr + bvlit(64, i, mem_expr.ctx), byte)
    return cur


def smt_defs_for_widths(widths: set[int]) -> list[str]:
    """Return SMT-LIB `(define-fun ...)` strings for every width
    in `widths` (one `store_le_N` and one `load_le_N` per N), in
    deterministic ascending-width order.

    The body for `store_le_N(M, A, V)` is the byte-by-byte
    `(store ... (store M A v[7:0]) (bvadd A 1) v[15:8] ...)` chain
    formerly inlined by the function above. Same for `load_le_N`.

    Consumers prepend these to their SMT-LIB declarations preamble
    before parsing LTS formulas; z3.parse_smt2_string inlines each
    `define-fun` at parse time, recovering the expanded form for
    solver-side simplification.
    """
    out: list[str] = []
    for n in sorted(widths):
        if n < 1:
            raise ValueError(f"smt_defs_for_widths: invalid width {n}")
        bits = 8 * n
        # store_le_N: nested (store ... ) building from the bottom up.
        # Innermost write is the byte at addr+0, outermost is addr+(N-1).
        store_body = "M"
        for i in range(n):
            offset_hex = f"#x{i:016x}"
            byte_extract = f"((_ extract {8*i + 7} {8*i}) V)"
            store_body = (
                f"(store {store_body} (bvadd A {offset_hex}) {byte_extract})"
            )
        out.append(
            f"(define-fun store_le_{n} "
            f"((M (Array (_ BitVec 64) (_ BitVec 8))) "
            f"(A (_ BitVec 64)) "
            f"(V (_ BitVec {bits}))) "
            f"(Array (_ BitVec 64) (_ BitVec 8)) "
            f"{store_body})"
        )
        # load_le_N: concat of N selects, MSB-first (high byte first
        # in concat so the result's LSB is the byte at addr+0).
        if n == 1:
            load_body = "(select M A)"
        else:
            sel_parts = []
            for i in range(n - 1, -1, -1):
                offset_hex = f"#x{i:016x}"
                sel_parts.append(f"(select M (bvadd A {offset_hex}))")
            load_body = "(concat " + " ".join(sel_parts) + ")"
        out.append(
            f"(define-fun load_le_{n} "
            f"((M (Array (_ BitVec 64) (_ BitVec 8))) "
            f"(A (_ BitVec 64))) "
            f"(_ BitVec {bits}) "
            f"{load_body})"
        )
    return out


def discover_load_store_widths_in_text(text: str) -> set[int]:
    """Scan an SMT-LIB string for `store_le_N` / `load_le_N` references
    and return the set of byte-widths N. Used by the extractor (when a
    future serialization-time compaction pass produces compact text)
    to populate the LTS's `smt_defs` field with only the widths an
    artifact references.

    Text-based rather than AST-based because the compact names only
    exist in the serialized output — the construction-time z3 ASTs
    keep the byte-by-byte expanded form (claripy compatibility).
    """
    import re as _re
    widths: set[int] = set()
    for m in _re.finditer(r"\b(?:store_le|load_le)_(\d+)\b", text):
        widths.add(int(m.group(1)))
    return widths


# ---------------------------------------------------------------------
# Substitution (correct-by-construction via z3)
# ---------------------------------------------------------------------

def subst(expr: z3.ExprRef, pairs: list[tuple[z3.ExprRef, z3.ExprRef]]) -> z3.ExprRef:
    """Substitute variables with expressions in `expr`. Each pair is
    (variable_to_replace, replacement_expression). Uses z3.substitute, which
    is correct by construction — no string gymnastics."""
    return z3.substitute(expr, pairs)


# ---------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------

def to_smtlib(expr: z3.ExprRef) -> str:
    """Emit the expression as an SMT-LIB S-expression string via z3's
    pretty printer, after running `z3.simplify` for canonicalization
    (collapses VEX widening sandwiches like
    `((_ extract 31 0) ((_ zero_extend 32) ((_ extract 31 0) X)))`,
    and rewrites `(= (ite (bvsgt ... ) 1 0) 1)` style cmp-coercions
    to `(not (bvsle ...))`). Semantics-preserving; the simplifier is
    z3's built-in. Output is standard SMT-LIB 2."""
    return z3.simplify(expr).sexpr()


# ---------------------------------------------------------------------
# Simplification helpers
# ---------------------------------------------------------------------

def simplify(expr: z3.ExprRef) -> z3.ExprRef:
    """Apply z3's simplifier. Useful for shortening generated formulas at
    LTS emission time; optional for correctness."""
    return z3.simplify(expr)

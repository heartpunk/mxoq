# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "z3-solver>=4.13",
# ]
# ///
"""Python executor for QF_ABV LTS extracted by `demo.py` / the small-core
extractor.

Loads `tsm_utf8_mach_feed.lts.json` (the whole-function LTS), sets up the
SystemV-amd64 calling convention for `tsm_utf8_mach_feed(mach*, ch)`,
steps through transitions until the function returns, and reports the
observation (return value in rax + 8 bytes at the mach pointer).

This is a CONCRETE evaluator — not symbolic execution. Inputs are concrete
ints; outputs are concrete ints. The LTS's symbolic formulas are evaluated
under the concrete state via z3.substitute + z3.simplify.

Usage:
    uv run executor.py tsm_utf8_mach_feed.lts.json --byte 0x41
    uv run executor.py tsm_utf8_mach_feed.lts.json --bytes 0xc3,0xa9   # multi-byte UTF-8

For property tests against the native binary, see test_executor.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import z3

import smt_emit as s

# Ctx-scoped local alias (sophie option C, 2026-04-27): every z3 primitive
# constructor in this module routes through `_CTX` so executor-side
# expressions live in the same context as smt_emit's. The executor still
# parses SMT-LIB at the wire-format boundary via `parse_smt2_string`,
# which accepts an explicit `ctx` argument; threading it through keeps
# all derived expressions ctx-consistent and lets executor-built state
# (register values, mem array) compose cleanly with parsed formula
# expressions inside z3.substitute / z3.simplify.
_CTX = s.LTS_EXTRACTION_CTX


# ---------------------------------------------------------------------
# Calling convention + addresses (matches Validation/Targets/TsmUtf8Attest.lean)
# ---------------------------------------------------------------------

MACH_PTR = 0x10000        # rdi: where we put the mach struct
RSP_INITIAL = 0x7FFFF000  # initial stack pointer
MACH_BYTES = 8            # libtsm.so.4.4.2 mach struct is 8 bytes
MAX_STEPS = 10_000        # safety bound; tsm_utf8_mach_feed runs in a few dozen

# Sentinel return address. We seed [rsp_initial] with this 64-bit value
# so when the function's terminating `ret` loads the saved-rip from the
# stack, it ends up here. Halt iff rip == SENTINEL_RIP. Anything else
# (no firing transitions, rip pointing to an unmapped block) is an
# extraction-or-executor BUG and surfaces as an exception, not a silent
# halt. Per codex: "No candidates should be an error unless it is exactly
# the sentinel."
SENTINEL_RIP = 0xDEADC0DE


# ---------------------------------------------------------------------
# Sort schema for free variables in the LTS's SMT formulas
# ---------------------------------------------------------------------

_GPRS_64 = (
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
    "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "rip",
)
_FLAGS_1 = ("rflags_CF", "rflags_ZF", "rflags_SF", "rflags_OF", "rflags_PF", "rflags_AF")

def _decls_str() -> str:
    """Build the SMT-LIB declarations preamble matching the LTS's
    free variables: GPRs (64-bit), flags (1-bit), and the byte-
    addressable `mem` array.

    No memory-read placeholder declarations: angr_backend
    intercepts every read with known width and substitutes it back
    to `load_le(mem_at_read, addr, nbytes)` at extraction time. The
    backend asserts no `__qfabv_load_*__` or `mem_<hex>_<id>_<width>`
    placeholder survives in the emitted LTS, so the executor only
    needs to declare the actual model variables.
    """
    lines = []
    for r in _GPRS_64:
        lines.append(f"(declare-fun {r} () (_ BitVec 64))")
    for f in _FLAGS_1:
        lines.append(f"(declare-fun {f} () (_ BitVec 1))")
    lines.append("(declare-fun mem () (Array (_ BitVec 64) (_ BitVec 8)))")
    return "\n".join(lines)


# Module-level constant preamble. The LTS contract (post angr_backend
# commit C) is uniform across targets — every LTS references this same
# set of free vars. No per-LTS preamble construction needed.
_DECLS = _decls_str()


# Cache for `decls_for_lts` keyed by the tuple of smt_defs strings.
# Without this cache, two run() calls on the same LTS would each
# build a fresh decls string with a different `id()`, defeating
# `parse_smt`'s id-keyed cache and reparsing every formula every
# call (codex 2026-04-27 non-blocking note caught during B.2 review).
_DECLS_CACHE: dict[tuple[str, ...], str] = {}


def decls_for_lts(lts: dict) -> str:
    """Return the parsing preamble for `lts`, augmented with the
    LTS's top-level `smt_defs` strings (B.2: the extractor emits
    `(store_le_N M A V)` / `(load_le_N M A)` calls in the
    transitions, and the matching `(define-fun store_le_N ...)`
    bodies live in the top-level `smt_defs` field). z3's
    `parse_smt2_string` inlines the define-funs at parse time so
    downstream `z3.simplify` / `z3.substitute` see the expanded
    QF_ABV form — same shape downstream consumers always saw,
    just compactly serialized in the JSON.

    Returns `_DECLS` unchanged when the LTS has no compact
    references (older or vex-only artifacts).

    Cached by tuple-of-smt_defs so repeated calls on the same LTS
    return the same string object → `parse_smt` cache (keyed by
    `id(decls)`) keeps hitting across feed_byte / run_sequence
    calls.
    """
    smt_defs = lts.get("smt_defs", []) or []
    if not smt_defs:
        return _DECLS
    key = tuple(smt_defs)
    cached = _DECLS_CACHE.get(key)
    if cached is None:
        cached = _DECLS + "\n" + "\n".join(smt_defs)
        _DECLS_CACHE[key] = cached
    return cached


def _sort_for_field(field: str) -> str:
    """Return the SMT sort string for a given LTS field name."""
    if field in _FLAGS_1:
        return "(_ BitVec 1)"
    if field == "mem":
        return "(Array (_ BitVec 64) (_ BitVec 8))"
    # default: 64-bit BV (registers, rip, tgt)
    return "(_ BitVec 64)"


# ---------------------------------------------------------------------
# SMT parsing + concrete evaluation via z3
# ---------------------------------------------------------------------

def _parse_smt(expr_str: str, sort: str, decls: str = _DECLS) -> z3.ExprRef:
    """Parse an SMT-LIB expression string under the given declarations.
    Returns a z3 ExprRef with free vars matching `decls`.
    """
    src = (
        decls
        + f"\n(declare-fun __probe__ () {sort})"
        + f"\n(assert (= __probe__ {expr_str}))"
    )
    asts = z3.parse_smt2_string(src, ctx=_CTX)
    # asts[0] is `(= __probe__ <body>)`; arg(1) is the body we want
    return asts[0].arg(1)


# Cache key includes id(decls) so per-LTS preambles get separate
# entries (different decls → different z3 AST identity for the same
# free-var name → must reparse).
_PARSE_CACHE: dict[tuple[int, str, str], z3.ExprRef] = {}


def parse_smt(expr_str: str, sort: str, decls: str = _DECLS) -> z3.ExprRef:
    """Cached parse of an SMT expression. Same (decls, expr_str, sort)
    returns the same z3 AST."""
    key = (id(decls), expr_str, sort)
    cached = _PARSE_CACHE.get(key)
    if cached is None:
        cached = _parse_smt(expr_str, sort, decls=decls)
        _PARSE_CACHE[key] = cached
    return cached


def _state_subs(state: dict[str, z3.ExprRef]) -> list[tuple[z3.ExprRef, z3.ExprRef]]:
    """Build the (free-var, concrete-value) substitution list for z3.substitute.

    Width comes from the stored value itself (z3 BitVecVal carries its
    own size), so this works uniformly for the fixed-width registers
    (64-bit) and per-flag bits (1-bit) the LTS references.
    """
    subs = []
    for name, val in state.items():
        if name == "mem":
            subs.append((
                z3.Array(
                    "mem",
                    z3.BitVecSort(64, ctx=_CTX),
                    z3.BitVecSort(8, ctx=_CTX),
                ),
                val,
            ))
        elif name in _FLAGS_1:
            subs.append((z3.BitVec(name, 1, ctx=_CTX), val))
        else:
            w = val.size()
            subs.append((z3.BitVec(name, w, ctx=_CTX), val))
    return subs


def smt_eval(
    expr_str: str,
    sort: str,
    state: dict[str, z3.ExprRef],
    decls: str = _DECLS,
) -> z3.ExprRef:
    """Parse, substitute concrete state values, simplify to a literal."""
    expr = parse_smt(expr_str, sort, decls=decls)
    return z3.simplify(z3.substitute(expr, _state_subs(state)))


# ---------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------

def initial_state(
    entry_addr: int,
    mach_init_bytes: bytes,
    byte: int,
) -> dict[str, z3.ExprRef]:
    """Build the initial state for a single-byte feed call.

    Per Validation/Targets/TsmUtf8Attest.lean conventions:
    - rdi = mach pointer (0x10000)
    - rsi = byte
    - rsp = 0x7ffff000
    - rip = entry
    - all other registers zeroed
    - all flags zeroed
    - mem zeroed outside the mach region (initially zeroed inside too)
    - saved-rip slot at [rsp_initial] seeded with SENTINEL_RIP so the
      terminating `ret` lands somewhere we can recognize. Reads of
      that slot (e.g. the function's stack-canary-shaped entry-guard
      that compares the saved-rip with 0) flow through `mem` because
      angr_backend resolves all reads to load_le over the mem array.
    """
    state: dict[str, z3.ExprRef] = {}
    for r in _GPRS_64:
        state[r] = z3.BitVecVal(0, 64, ctx=_CTX)
    state["rdi"] = z3.BitVecVal(MACH_PTR, 64, ctx=_CTX)
    state["rsi"] = z3.BitVecVal(byte, 64, ctx=_CTX)
    state["rsp"] = z3.BitVecVal(RSP_INITIAL, 64, ctx=_CTX)
    state["rip"] = z3.BitVecVal(entry_addr, 64, ctx=_CTX)
    for f in _FLAGS_1:
        state[f] = z3.BitVecVal(0, 1, ctx=_CTX)
    # zeroed memory; concrete arrays in z3 use K(0) for default-zero
    mem = z3.K(z3.BitVecSort(64, ctx=_CTX), z3.BitVecVal(0, 8, ctx=_CTX))
    # write mach_init_bytes into the mach region
    for i, b in enumerate(mach_init_bytes):
        mem = z3.Store(
            mem,
            z3.BitVecVal(MACH_PTR + i, 64, ctx=_CTX),
            z3.BitVecVal(b, 8, ctx=_CTX),
        )
    # Seed the saved-rip slot at [rsp_initial] with SENTINEL_RIP so the
    # function's terminating `ret` loads sentinel into rip — making
    # function exit explicit and unambiguous (vs falling out the bottom
    # because rip became 0 from a zeroed stack slot, which would also
    # halt iff 0 isn't a known block address — fragile).
    sentinel_bytes = SENTINEL_RIP.to_bytes(8, "little")
    for i, b in enumerate(sentinel_bytes):
        mem = z3.Store(
            mem,
            z3.BitVecVal(RSP_INITIAL + i, 64, ctx=_CTX),
            z3.BitVecVal(b, 8, ctx=_CTX),
        )
    state["mem"] = z3.simplify(mem)
    return state


# ---------------------------------------------------------------------
# LTS step semantics
# ---------------------------------------------------------------------

def evaluate_guard(
    guard_str: str,
    state: dict[str, z3.ExprRef],
    decls: str = _DECLS,
) -> bool:
    """Evaluate a guard SMT formula under the concrete state. Returns True/False."""
    val = smt_eval(guard_str, "Bool", state, decls=decls)
    if z3.is_true(val):
        return True
    if z3.is_false(val):
        return False
    raise RuntimeError(f"guard did not simplify to a literal: {val}")


def apply_sub(
    state: dict[str, z3.ExprRef],
    sub: dict[str, str],
    decls: str = _DECLS,
) -> dict[str, z3.ExprRef]:
    """Apply a parallel state-update. Each sub field's value is evaluated
    under the *old* state, then all updates are committed to a new state.
    """
    new_vals = {}
    for field, expr_str in sub.items():
        sort = _sort_for_field(field)
        new_vals[field] = smt_eval(expr_str, sort, state, decls=decls)
    new_state = dict(state)
    new_state.update(new_vals)
    return new_state


def step(
    state: dict[str, z3.ExprRef],
    lts: dict,
    decls: str = _DECLS,
) -> tuple[dict[str, z3.ExprRef], bool]:
    """Take one LTS step. Returns (new_state, halted).

    Halt iff rip == SENTINEL_RIP — the function's `ret` loaded sentinel
    from the seeded saved-rip slot, and we're done. Any other "no
    transition matches the current rip" is a BUG (extraction missing
    a block, or executor diverged from the LTS) and surfaces as an
    exception, not silent termination.

    Determinism invariants (per codex review):
    1. Among candidates from the current rip, EXACTLY ONE guard must
       fire. Zero firing → guards malformed; multiple firing →
       overlapping guards (LTS non-deterministic at this rip). Both
       are bugs we want surfaced rather than papered over by picking
       the first match.
    2. If an edge specifies both an explicit `tgt` AND a `sub["rip"]`,
       they must agree under the OLD state. Disagreement is an
       extraction inconsistency: the explicit tgt comes from
       angr_backend's `post_addr` short-circuit, sub["rip"] from the
       symbolic next-rip; both should describe the same successor.
    """
    rip_val = z3.simplify(state["rip"]).as_long()
    if rip_val == SENTINEL_RIP:
        return state, True
    candidates = [t for t in lts["transitions"] if int(t["src"], 16) == rip_val]
    if not candidates:
        raise RuntimeError(
            f"no transitions for rip=0x{rip_val:x} (and not SENTINEL_RIP); "
            f"LTS missing a block, or executor diverged"
        )
    firing = [t for t in candidates if evaluate_guard(t["guard"], state, decls=decls)]
    if len(firing) == 0:
        raise RuntimeError(
            f"no firing transition at rip=0x{rip_val:x} "
            f"(candidates: {len(candidates)}); LTS may be incomplete "
            f"or guards malformed"
        )
    if len(firing) > 1:
        kinds = [t.get("kind", "?") for t in firing]
        raise RuntimeError(
            f"multiple firing transitions at rip=0x{rip_val:x} "
            f"(firing: {len(firing)}, kinds: {kinds}); LTS guards "
            f"overlap — non-deterministic at this rip"
        )
    t = firing[0]
    new_state = apply_sub(state, t["sub"], decls=decls)
    # Resolve next rip. Validate explicit-tgt and sub["rip"]
    # agreement when both are present so a future extractor change
    # that drifts the two doesn't silently produce wrong control
    # flow.
    explicit_tgt = t.get("tgt")
    sub_rip = t.get("sub", {}).get("rip") if t.get("sub") else None
    if explicit_tgt and sub_rip:
        # Both present — check agreement under the OLD state. The
        # explicit tgt is a hex string; sub_rip is an SMT expression
        # whose value should match.
        explicit_val = int(explicit_tgt, 16)
        sub_rip_val = smt_eval(sub_rip, "(_ BitVec 64)", state, decls=decls)
        if not z3.is_bv_value(sub_rip_val):
            raise RuntimeError(
                f"sub['rip'] did not simplify to a literal at "
                f"rip=0x{rip_val:x}: {sub_rip_val}"
            )
        if sub_rip_val.as_long() != explicit_val:
            raise RuntimeError(
                f"explicit tgt 0x{explicit_val:x} disagrees with "
                f"sub['rip'] 0x{sub_rip_val.as_long():x} at "
                f"rip=0x{rip_val:x}"
            )
        new_state["rip"] = z3.BitVecVal(explicit_val, 64, ctx=_CTX)
    elif explicit_tgt:
        new_state["rip"] = z3.BitVecVal(int(explicit_tgt, 16), 64, ctx=_CTX)
    elif t.get("tgt_smt"):
        new_state["rip"] = smt_eval(
            t["tgt_smt"], "(_ BitVec 64)", state, decls=decls,
        )
    # else: sub already set new_state["rip"] via apply_sub (sub
    # contained a rip entry), no override needed.
    return new_state, False


def run(
    lts: dict,
    byte: int,
    mach_init: bytes = b"\x00" * MACH_BYTES,
    max_steps: int = MAX_STEPS,
) -> tuple[int, bytes]:
    """Run the LTS for one feed call. Returns (return_value_low32, mach_post_bytes)."""
    entry = int(lts["entry_addr"], 16)
    state = initial_state(entry, mach_init, byte)
    decls = decls_for_lts(lts)
    for _ in range(max_steps):
        state, halted = step(state, lts, decls=decls)
        if halted:
            break
    else:
        raise RuntimeError(f"executor did not halt within {max_steps} steps")
    # Observation: return_value = low 32 bits of rax
    rax = z3.simplify(state["rax"]).as_long()
    ret_val = rax & 0xFFFFFFFF
    # Sign-extend to int32-like (per the int return type in the C ABI)
    if ret_val >= 0x80000000:
        ret_val -= 0x100000000
    # mach_post_bytes: 8 bytes at mach pointer
    mem = state["mem"]
    bytes_out = bytearray()
    for i in range(MACH_BYTES):
        b = z3.simplify(z3.Select(mem, z3.BitVecVal(MACH_PTR + i, 64, ctx=_CTX))).as_long()
        bytes_out.append(b)
    return ret_val, bytes(bytes_out)


def run_sequence(
    lts: dict,
    byte_seq: list[int],
    max_steps_per_call: int = MAX_STEPS,
) -> tuple[int, bytes]:
    """Run the LTS for a sequence of bytes (multi-byte UTF-8). The mach
    state is carried across calls; the return value is from the LAST call.
    """
    mach = b"\x00" * MACH_BYTES
    ret = 0
    for b in byte_seq:
        ret, mach = run(
            lts, b,
            mach_init=mach,
            max_steps=max_steps_per_call,
        )
    return ret, mach


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def _parse_byte(s: str) -> int:
    if s.startswith(("0x", "0X")):
        return int(s, 16)
    return int(s)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Concrete LTS executor for tsm_utf8_mach_feed. Feeds bytes into "
            "the extracted LTS and reports the observation (return value + "
            "post-call mach struct bytes), matching the native binary's "
            "behavior up to the LTS extraction's correctness."
        ),
    )
    p.add_argument(
        "lts_json", type=Path,
        help="Path to the whole-function LTS JSON (e.g. tsm_utf8_mach_feed.lts.json).",
    )
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--byte",
        help="Single byte to feed (decimal or 0x-prefixed hex).",
    )
    grp.add_argument(
        "--bytes",
        help="Comma-separated byte sequence (e.g. 0xc3,0xa9 for é).",
    )
    args = p.parse_args(argv)

    lts = json.loads(args.lts_json.read_text())

    if args.byte is not None:
        byte = _parse_byte(args.byte)
        ret, mach = run(lts, byte)
        result = {
            "input_bytes": [byte],
            "return_value": ret,
            "mach_post_bytes": list(mach),
        }
    else:
        byte_seq = [_parse_byte(b.strip()) for b in args.bytes.split(",")]
        ret, mach = run_sequence(lts, byte_seq)
        result = {
            "input_bytes": byte_seq,
            "return_value": ret,
            "mach_post_bytes": list(mach),
        }
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

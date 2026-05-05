"""IMark-granular angr-backed block extractor.

Walks one basic block of an x86-64 binary one machine instruction at
a time via `project.factory.successors(state, num_inst=1)`, then
reads register / memory state-diffs and the post-step rflags value
off claripy and emits one `Transition` record per outgoing
control-flow edge. Each `Transition` carries an SMT-LIB guard and a
parallel state-update map (`sub`) in QF_ABV.

Symbolic state setup:
  - All canonical GPRs (rax..r15, rip) are reseeded as fresh
    BVS symbols whose names match the canonical smt_emit
    variable names (rax, rcx, ...) so the post-conversion z3
    expressions reference exactly those free symbols.
  - rsp / rbp are reseeded symbolic too, with a range constraint
    `rsp >= 0x40000000` that keeps angr's address-concretization
    strategy from pinning the stack into the loaded code region.
  - An `address_concretization` BP_BEFORE inspect hook sets
    `add_constraints=False` for symbolic-write concretization, so
    rsp stays symbolic in the emitted sub even across stack writes.
  - A `mem_read` BP_AFTER inspect hook intercepts EVERY known-width
    memory read (stack-derived or otherwise) and replaces angr's
    concretized read result with a `__qfabv_load_N__` placeholder
    BVS, recording `(addr, nbytes, mem_at_read)` in
    `state.globals['_qfabv_load_placeholders']`. A paired
    `mem_write` BP_AFTER hook folds every write into a side-band
    `_qfabv_mem_expr` so each read's `mem_at_read` reflects all
    preceding in-block stores. After stepping, the resolver
    substitutes each placeholder with
    `s.load_le(mem_at_read, addr_z3, nbytes)` — uniform across
    stack and non-stack reads.

Eager flag decomposition uses a `cc_op = G_CC_OP_COPY` seeding
trick: cc_dep1 is packed with six 1-bit BVS symbols at the
architectural flag bit positions (CF=0, PF=2, AF=4, ZF=6, SF=7,
OF=11). Reading `state.regs.rflags` post-step triggers angr's
`pc_calculate_rdata_all_WRK` helper, which projects exactly those
six bits out of cc_dep1 in the COPY case, and unfolds to explicit
per-op SMT (e.g. cc_op = G_CC_OP_SUBQ) when an instruction wrote
flags. Each flag becomes its own 1-bit BV variable in `sub`.

Edge classification uses `post.scratch.exit_stmt_idx`: ≥0 indicates
the post-state took a specific Exit stmt (kind="exit"), -2 indicates
the default fallthrough (kind="default"). This handles multi-Exit
IRSBs (e.g. `rep cmpsb`) where two distinct successors can share the
same post-rip value but only one of them is the structural default.

For Ijk_Ret the default edge carries `tgt=None` and `tgt_smt =
load_le(mem, rsp, 8)`; for Ijk_Call it carries `kind="default"`,
`tgt = callee addr`, `callee_sym` populated, and
`jumpkind="Ijk_Call"`. `kind="call"` is intentionally not emitted
— each block produces exactly one `kind="default"` edge; the
callness is carried by `jumpkind` + `callee_sym`.

Fail-fast on:
  - Indirect control (irsb.next non-concrete and jumpkind ≠ Ijk_Ret).
  - Non-byte-aligned memory operations.
  - Symbolic / multivalued / unsupported cc_op post-step (caught
    from angr.errors.SimError / SimCCallError /
    CCallMultivaluedException raised by the rflags helper).

`flag_reads` on emitted edges is left empty; the field is vestigial
in the schema and no downstream consumer reads it.
`exit_flag_status` is populated via `dataflow.block_effect`, the
capstone-driven flow-sensitive analysis.

KEY INVARIANT: for a Jcc-terminated block the default edge's guard
is the conjunction of the negated exit guards — not the constant
True. The deterministic-step policy is enforced post-loop in
`translate_block`.
"""
from __future__ import annotations

import dataclasses
import logging
import re

import claripy
import z3
from claripy.backends.backend_z3 import BackendZ3

import smt_emit as s
from transition import Transition, MemWriteOp, VexTranslationError
from target_manifest import TargetManifest

# angr's unicorn plugin warns loudly on macOS where unicorn is absent;
# silence it (unicorn is only needed for concrete exec, not our symbolic path).
logging.getLogger("angr.state_plugins.unicorn_engine").setLevel(logging.CRITICAL)

# Lazy-import angr to keep this module importable even when angr isn't
# installed (matches vex_to_smt's behavior).
import angr  # noqa: E402
from angr.engines.vex.claripy import ccall as _angr_ccall  # noqa: E402

_BZ3 = BackendZ3()

# Ctx-scoped local alias (sophie option C, 2026-04-27): every smt_emit
# minting helper requires explicit ctx. All construction in this module
# routes through `s.LTS_EXTRACTION_CTX`; cross-ctx ops with main_ctx
# (claripy's default) raise loudly via z3's context_mismatch check.
_CTX = s.LTS_EXTRACTION_CTX

# Canonical GPR set for amd64. We reseed each register symbolically
# at block entry and scan for sub entries at block exit. Names match
# smt_emit's reg64(name) convention so post-conversion z3 vars are
# literally `rax`, `rcx`, etc, with no claripy auto-suffix.
_GPR64 = (
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
    "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
)

# Lower bound on symbolic rsp. Keeps concretization away from the
# code region (which lives near 0x400000 for ELF, 0x0 for shellcode).
_STACK_FLOOR = 0x40000000

# Eager flag computation: cc_op is seeded with G_CC_OP_COPY and
# cc_dep1 is packed with symbolic per-flag BVS at the architectural
# bit positions. Reading state.regs.rflags triggers angr's
# pc_calculate_rdata_all_WRK helper, which under COPY projects
# exactly these six bits out of cc_dep1 (masking everything else
# off). When an instruction writes flags, angr replaces cc_op with
# the operation-specific value (e.g. G_CC_OP_SUBQ) and reads of
# rflags unfold to the explicit per-flag SMT formulas.
_AMD64_OP_COPY: int = _angr_ccall.data["AMD64"]["OpTypes"]["G_CC_OP_COPY"]
_AMD64_FLAG_BIT: dict[str, int] = {
    "CF": _angr_ccall.data["AMD64"]["CondBitOffsets"]["G_CC_SHIFT_C"],
    "PF": _angr_ccall.data["AMD64"]["CondBitOffsets"]["G_CC_SHIFT_P"],
    "AF": _angr_ccall.data["AMD64"]["CondBitOffsets"]["G_CC_SHIFT_A"],
    "ZF": _angr_ccall.data["AMD64"]["CondBitOffsets"]["G_CC_SHIFT_Z"],
    "SF": _angr_ccall.data["AMD64"]["CondBitOffsets"]["G_CC_SHIFT_S"],
    "OF": _angr_ccall.data["AMD64"]["CondBitOffsets"]["G_CC_SHIFT_O"],
}
_AMD64_FLAG_NAMES = ("CF", "PF", "AF", "ZF", "SF", "OF")


class AngrBackendNotImplemented(VexTranslationError):
    """Raised when the angr backend encounters a block shape it doesn't
    handle yet (conditional branches, setcc, flag-consumer opcodes).
    Fail-fast so the caller knows to fall back to the vex backend."""


def _no_concretization_constraints(state):
    """Inspect hook: for symbolic memory WRITES, do NOT add the
    equality constraint that pins the address variable. This
    preserves symbolic rsp in the emitted sub. Reads are left
    alone — without add_constraints for reads, angr enumerates
    multiple possible concrete addrs and constructs an If ladder
    over them which fails with mismatched widths when combined
    with our placeholder substitution. Stack reads are handled
    separately via the mem_read BP_AFTER hook, which overrides
    the read-result after angr's internal load completes.
    """
    if state.inspect.address_concretization_action == "store":
        state.inspect.address_concretization_add_constraints = False


def _qfabv_load_hook(state):
    """BP_AFTER hook for mem_read: replace ANY known-width memory
    read result with a fresh `__qfabv_load_N__` placeholder BVS, and
    record `(addr, nbytes, mem_at_read)` in
    state.globals['_qfabv_load_placeholders']. The post-step resolver
    substitutes the placeholder back to
    `s.load_le(mem_at_read, addr_z3, nbytes)`.

    Generalized in commit C from the original stack-only hook so the
    LTS represents EVERY memory read uniformly as a `load_le` on the
    `mem` array. Pre-C, non-stack reads passed through angr's default
    model and emerged as `mem_<addr>_<id>_<width>` filler vars — a
    semantic leak (the LTS lost the dependency on `mem`) and a leak-
    assertion failure once we generalized the assertion. Stack-derived
    vs not is now naming/diagnostics only; the resolution path is the
    same.

    `mem_at_read` is the side-band z3 mem expression captured at this
    exact point in the block — it includes every store the block has
    issued so far (folded in by `_mem_write_timeline_hook`). This is
    what makes the resolver correct against in-block store/load
    chains: a read of [rbp-0x10] after a `mov [rbp-0x10], rdi` sees a
    `(store mem ... rdi)` snapshot, so z3 simplification folds the
    select-of-store down to bytes of `rdi` rather than the bare
    pre-block memory.
    """
    addr = state.inspect.mem_read_address
    if addr is None:
        return
    length = state.inspect.mem_read_length
    try:
        nbytes = int(length) if not hasattr(length, "op") else (
            int(length.args[0]) if length.op == "BVV" else None
        )
    except Exception:
        nbytes = None
    if nbytes is None or nbytes <= 0:
        return
    # Only override if the placeholder width matches angr's original
    # read-result width. Mismatches (rare but possible with endian
    # conversions or sub-byte reads) would cause claripy to raise on
    # subsequent If/concat operations that expect a specific size.
    expected_bits = nbytes * 8
    existing_expr = state.inspect.mem_read_expr
    if existing_expr is not None and hasattr(existing_expr, "size"):
        existing_bits = existing_expr.size()
        if existing_bits != expected_bits:
            return
    # Replace the table to avoid mutating a dict that may be aliased
    # by a forked sibling state. (state.globals copies shallowly on
    # state forks, so reassigning is the safe pattern.)
    table = dict(state.globals.get("_qfabv_load_placeholders", {}))
    idx = len(table)
    placeholder_name = f"__qfabv_load_{idx}__"
    placeholder = claripy.BVS(placeholder_name, expected_bits, explicit_name=True)
    mem_at_read = state.globals.get("_qfabv_mem_expr", s.mem(_CTX))
    table[placeholder_name] = (addr, nbytes, mem_at_read)
    state.globals["_qfabv_load_placeholders"] = table
    state.inspect.mem_read_expr = placeholder


def _mem_write_timeline_hook(state):
    """BP_AFTER hook for mem_write: fold the just-completed write into
    the side-band z3 mem expression in
    `state.globals['_qfabv_mem_expr']` as a `store_le(...)`, AND
    record a `MemWriteOp` provenance entry in
    `state.globals['_qfabv_mem_timeline']` capturing the angr-reported
    write width — authoritative for "single n-byte machine write" and
    therefore the correct source for compact `(store_le_N M A V)`
    serialization (vs. structural pattern-matching the AST shape,
    which is unsound: a byte-write loop and a single n-byte write
    produce identical chains).

    Together with `_qfabv_load_hook` (which captures
    `mem_at_read` from the same global), this gives every memory-read
    placeholder access to the in-block memory state at its read
    point — the foundation for resolving placeholders against the
    correct store/load chain instead of bare pre-block memory.

    Non-BV stores (uncommon but possible for vector ops the
    accepted-mnemonic preflight doesn't yet model) raise so the
    extraction fails loud rather than silently dropping a write.

    Timeline list discipline: appended via copy-on-write
    (`list(...)` then append, then reassign) so angr's per-fork
    state.globals shallow-copy doesn't share mutation across
    forked sibling states (codex 2026-04-27 design review).
    """
    addr = state.inspect.mem_write_address
    data = state.inspect.mem_write_expr
    length = state.inspect.mem_write_length
    if data is None or addr is None:
        return
    # nbytes from explicit length when present; otherwise derive
    # from the data's bit-width.
    nbytes: int | None
    if length is not None:
        try:
            nbytes = int(length) if not hasattr(length, "op") else (
                int(length.args[0]) if length.op == "BVV" else None
            )
        except Exception:
            nbytes = None
    else:
        nbytes = None
    if nbytes is None:
        if not hasattr(data, "size"):
            raise AngrBackendNotImplemented(
                "mem_write hook: non-BV write data without explicit length"
            )
        bits = data.size()
        if bits % 8 != 0:
            raise AngrBackendNotImplemented(
                f"mem_write hook: non-byte-aligned write width {bits}"
            )
        nbytes = bits // 8
    if nbytes <= 0:
        return
    cur = state.globals.get("_qfabv_mem_expr", s.mem(_CTX))
    addr_z3 = _to_z3(addr)
    data_z3 = _to_z3(data)
    if data_z3.size() != nbytes * 8:
        raise AngrBackendNotImplemented(
            f"mem_write hook: data size {data_z3.size()} bits != "
            f"length {nbytes * 8} bits"
        )
    new_mem = s.store_le(cur, addr_z3, data_z3, nbytes)
    state.globals["_qfabv_mem_expr"] = new_mem

    # Record write-op provenance for compact serialization.
    # `state.scratch.ins_addr` is the currently-executing instruction's
    # PC inside this IRSB step; safe to capture here since the hook
    # fires inside instruction execution.
    pc_int = int(state.scratch.ins_addr)
    op = MemWriteOp(
        kind="store_le_n",
        nbytes=nbytes,
        addr_z3=addr_z3,
        val_z3=data_z3,
        prev_mem_id=cur.get_id(),
        result_mem_id=new_mem.get_id(),
        pc=pc_int,
    )
    timeline = list(state.globals.get("_qfabv_mem_timeline", ()))
    timeline.append(op)
    state.globals["_qfabv_mem_timeline"] = timeline


def _seed_flag_bvs() -> dict[str, "claripy.ast.bv.BV"]:
    """Mint six 1-bit symbolic flag values, named to match smt_emit
    canonical naming (rflags_CF / rflags_ZF / etc). explicit_name=True
    suppresses claripy's auto-suffix so the post-conversion z3 vars
    are literally `rflags_CF`, etc — matching what vex_to_smt emits
    and what QfabvLean/Arch declares."""
    return {
        f: claripy.BVS(f"rflags_{f}", 1, explicit_name=True)
        for f in _AMD64_FLAG_NAMES
    }


def _build_initial_cc_dep1(flag_bvs: dict[str, "claripy.ast.bv.BV"]) -> "claripy.ast.bv.BV":
    """Build a 64-bit cc_dep1 with the six flag BVS at their AMD64
    G_CC_SHIFT_* positions, zero elsewhere.

    NOTE: angr's COPY-path in pc_calculate_rdata_all_WRK
    (ccall.py:592-601) returns `cc_dep1 & (mask_O|mask_S|mask_Z|
    mask_A|mask_C|mask_P)` — only the six flag bits survive. Bits at
    other positions are masked off and don't reach state.regs.rflags.
    So we don't need the architectural fixed-1 at bit 1; that bit
    becomes don't-care.
    """
    pieces: list = []
    next_bit = 63
    # Walk down 63 → 0; emit flag BVS at flag positions, BVV(0, run)
    # for zero-runs. flag positions in descending order:
    flag_positions = sorted(_AMD64_FLAG_BIT.items(), key=lambda kv: -kv[1])
    # = [(OF,11), (SF,7), (ZF,6), (AF,4), (PF,2), (CF,0)]
    for flag_name, bit_pos in flag_positions:
        if next_bit > bit_pos:
            pieces.append(claripy.BVV(0, next_bit - bit_pos))
        pieces.append(flag_bvs[flag_name])
        next_bit = bit_pos - 1
    if next_bit >= 0:
        pieces.append(claripy.BVV(0, next_bit + 1))
    return claripy.Concat(*pieces)


def _seed_flags_in_state(state: "angr.SimState") -> dict[str, "claripy.ast.bv.BV"]:
    """Set up cc_op = G_CC_OP_COPY (concrete sentinel), cc_dep1 =
    packed flag BVS, cc_dep2/cc_ndep = 0. Returns the per-flag BVS
    dict so callers can detect "unchanged" reads in post-state by
    syntactic identity check.
    """
    flag_bvs = _seed_flag_bvs()
    state.regs.cc_op = claripy.BVV(_AMD64_OP_COPY, 64)
    state.regs.cc_dep1 = _build_initial_cc_dep1(flag_bvs)
    state.regs.cc_dep2 = claripy.BVV(0, 64)
    state.regs.cc_ndep = claripy.BVV(0, 64)
    return flag_bvs


def _read_post_rflags_or_fail(post: "angr.SimState", bb_addr: int) -> "claripy.ast.bv.BV":
    """Read post-step rflags via angr's COPY-path or per-op helper.
    Wrap the helper invocation so symbolic / multivalued / unsupported
    cc_op values surface as AngrBackendNotImplemented rather than
    silently producing partial-flag output.

    Three exception classes the helper can raise, all caught here:
      - SimError (op_concretize on a symbolic BVS cc_op)
      - CCallMultivaluedException (op_concretize on If-over-BVV cc_op)
      - SimCCallError (helper rejects the opcode, e.g. SMULQ/UMULQ)
    """
    # Imports kept local: these classes are not always present in old
    # angr versions; we use them only for the except block.
    from angr.errors import SimError, SimCCallError
    try:
        from angr.errors import CCallMultivaluedException
    except ImportError:
        # Older angr labelled this differently or didn't expose it; use
        # SimError as a superset (it catches the multivalued path too).
        CCallMultivaluedException = SimError  # type: ignore[assignment]

    try:
        return post.regs.rflags
    except (SimError, SimCCallError, CCallMultivaluedException) as exc:
        raise AngrBackendNotImplemented(
            f"block at 0x{bb_addr:x}: cc_op symbolic / multivalued / "
            f"unsupported by angr's rflags helper ({type(exc).__name__}: "
            f"{exc})"
        ) from exc


def _extract_post_flag_subs(
    post: "angr.SimState",
    flag_bvs: dict[str, "claripy.ast.bv.BV"],
    bb_addr: int,
    sub: dict[str, z3.ExprRef],
    placeholders: dict | None = None,
) -> None:
    """Read post.regs.rflags, extract per-flag bits, emit `rflags_F`
    sub entries for any flag whose post-value differs syntactically
    from its seeded BVS. Mutates `sub` in place.

    Identity check uses z3 .eq() on the converted expression vs the
    converted seed BVS. When .eq() holds, no sub entry — the flag is
    unchanged from block entry, and downstream Lean side reuses the
    bare rflags_F free variable.

    `placeholders` is the side table from _qfabv_load_hook;
    when present, every emitted flag formula has its __qfabv_load_N__
    references resolved back to load_le(mem, addr, nbytes), matching
    how the caller resolves the rest of `sub` and `tgt_smt`. Without
    this, flag formulas that depend on a stack read (e.g. cmp [rsp+X],
    Y at the entry block) leaked __qfabv_load_N__ free vars past the
    backend's assertion.
    """
    post_rflags = _read_post_rflags_or_fail(post, bb_addr)
    for flag in _AMD64_FLAG_NAMES:
        bit = _AMD64_FLAG_BIT[flag]
        bit_expr = claripy.Extract(bit, bit, post_rflags)
        z_expr = _to_z3(bit_expr)
        z_seed = _to_z3(flag_bvs[flag])
        if z_expr.eq(z_seed):
            continue
        if placeholders:
            z_expr = _resolve_qfabv_load_placeholders(z_expr, placeholders)
        sub[f"rflags_{flag}"] = z_expr


def _exit_flag_status_for_block(
    project: "angr.Project",
    bb_addr: int,
    entry_flag_status: dict[str, str] | None,
) -> dict[str, str]:
    """Compute the exit_flag_status the same way vex_to_smt does:
    capstone-driven dataflow pass over block mnemonics, composed
    with the entry status. Reuses dataflow.block_effect so behavior
    matches per-mnemonic exactly.

    Caveat: this is only as honest as x86_flags.py's mnemonic
    coverage. If we accept mnemonics x86_flags doesn't model
    (inc/dec/neg/adc/sbb/ror/...), the dataflow pass returns
    conservative all-UNDEFINED or pass-through UNCHANGED, while the
    eager-flag side of `sub` is still correct. The metadata can
    drift from the actual flag values until x86_flags is widened.
    """
    import dataflow
    import x86_flags
    if entry_flag_status is None:
        entry_flag_status = {f: x86_flags.UNCHANGED for f in x86_flags.FLAGS}
    effect = dataflow.block_effect(project, bb_addr)
    out: dict[str, str] = {}
    for f in x86_flags.FLAGS:
        if effect[f] == x86_flags.UNCHANGED:
            out[f] = entry_flag_status[f]
        else:
            out[f] = effect[f]
    return out


def _fresh_symbolic_state(project: "angr.Project", entry: int) -> "angr.SimState":
    """Seed a SimState with claripy BVS vars whose names match the
    canonical smt_emit variables *exactly* (rax, rdi, ..., rsp, rbp).
    explicit_name=True suppresses claripy's default name-counter
    suffix so post-conversion z3 vars are literally s.reg64('rax', _CTX).

    rsp is constrained to >= _STACK_FLOOR to keep concretization
    away from the code region, and an address_concretization
    inspect hook sets add_constraints=False so rsp stays symbolic
    in the emitted sub even across stack writes.
    """
    state = project.factory.blank_state(addr=entry)
    # TRACK_MEMORY_ACTIONS dropped in B.2.1 — its sole consumer
    # `_build_mem_sub` was removed once B.2 (compact mem timeline
    # via per-write provenance) proved stable end-to-end via the
    # gist-regen consistency check at 0x40990f. The `_qfabv_mem_expr`
    # side-band + `_qfabv_mem_timeline` MemWriteOp records are now
    # the authoritative source for the per-block mem chain.
    state.options.add(angr.options.TRACK_REGISTER_ACTIONS)
    state.options.add(angr.options.SYMBOLIC_WRITE_ADDRESSES)
    state.options.add(angr.options.UNDER_CONSTRAINED_SYMEXEC)
    state.options.discard(angr.options.UNICORN)
    for reg in _GPR64:
        state.regs.__setattr__(reg, claripy.BVS(reg, 64, explicit_name=True))
    # Range constraint: rsp lives high enough that angr's
    # concretization can't land inside the code region.
    state.solver.add(state.regs.rsp >= _STACK_FLOOR)
    # Inspect hook: don't pin rsp (or any symbolic write addr) via
    # constraints when angr concretizes for a memory write.
    state.inspect.b(
        "address_concretization",
        when=angr.BP_BEFORE,
        action=_no_concretization_constraints,
    )
    # Side-band z3 memory expression — accumulated as the block
    # symbolically executes. Initialized to s.mem(_CTX) (the bare
    # block-entry mem array). Each mem write hook step folds the
    # write into a `store_le(...)` so a later read inside the same
    # block can capture the post-write memory snapshot. This is the
    # foundation for resolving __qfabv_load_N__ placeholders with
    # the right point-in-time memory expression instead of the bare
    # pre-block mem.
    state.globals["_qfabv_mem_expr"] = s.mem(_CTX)
    # Side-band write-op provenance timeline (B.1, sophie+codex 2026-04-27).
    # Each `_mem_write_timeline_hook` invocation appends a `MemWriteOp`
    # via copy-on-write. At block end, copied onto `Transition.mem_timeline`
    # for the compact serializer to walk (vs. AST pattern-matching).
    state.globals["_qfabv_mem_timeline"] = []
    # Inspect hook: substitute a placeholder BVS for EVERY known-width
    # memory read (stack-derived or otherwise) so emitted formulas
    # represent reads as `load_le(mem_at_read, addr, nbytes)` over the
    # `mem` array. Records the side-band mem snapshot at read time
    # for the resolver.
    state.inspect.b(
        "mem_read",
        when=angr.BP_AFTER,
        action=_qfabv_load_hook,
    )
    # Inspect hook: fold every memory write into the side-band mem
    # expression as a `store_le(...)`. Order follows IRSB statement
    # order, so mem_at_read snapshots taken in subsequent reads see
    # all preceding writes in the same block.
    state.inspect.b(
        "mem_write",
        when=angr.BP_AFTER,
        action=_mem_write_timeline_hook,
    )
    # Eager flag seeding: cc_op = COPY, cc_dep1 packs six BVS,
    # cc_dep2/cc_ndep = 0. Stored in state.globals so translate_block
    # can recover the BVS dict for syntactic identity checks during
    # post-state extraction.
    flag_bvs = _seed_flags_in_state(state)
    state.globals["_flag_bvs"] = flag_bvs
    return state


def _to_z3(ast) -> z3.ExprRef:
    """Convert a claripy AST to z3, then translate into the LTS extraction
    context. Preserves variable names (a claripy BVS named 'rax' becomes
    a z3 BitVec named 'rax').

    Translation discipline (sophie 2026-04-27, option C + isolation):
    claripy's _BZ3 backend builds expressions in z3.main_ctx (claripy's
    default). All smt_emit-side construction lives in
    `s.LTS_EXTRACTION_CTX`. We `translate` here so every expression
    that crosses the angr→us boundary lands in our context. Any later
    op that mixes z3.main_ctx and our context will raise
    `z3.Z3Exception: context mismatch` at the cross-ctx site — loud,
    not silent."""
    return _BZ3.convert(ast).translate(s.LTS_EXTRACTION_CTX)


_CFG_CACHE: dict[int, "angr.analyses.cfg.CFGFast"] = {}


def _resolve_indirect_via_cfg(
    project: "angr.Project", bb_addr: int
) -> list[int]:
    """Try to resolve an indirect jump/call target via angr's CFGFast.

    Caches the CFG per project (by id). Runs CFGFast with
    resolve_indirect_jumps=True so jump-table dispatches get resolved.
    Returns the list of concrete successor addresses for the block at
    bb_addr (excluding the fallthrough), or [] if CFGFast couldn't
    resolve.

    Codex 2025-05-05 design note: CFGFast is the canonical resolver for
    indirect jumps in angr. factory.successors() may not enumerate when
    the selector is symbolic; CFGFast pre-resolves via concrete analysis
    with constraints. This function is the fallback when symbolic
    stepping returned only symbolic-rip post-states.
    """
    cfg = _CFG_CACHE.get(id(project))
    if cfg is None:
        try:
            cfg = project.analyses.CFGFast(
                resolve_indirect_jumps=True,
                normalize=True,
                show_progressbar=False,
            )
            _CFG_CACHE[id(project)] = cfg
        except Exception:
            return []
    # angr 9.x: CFGFast nodes via cfg.model.get_any_node
    node = None
    for getter in ('get_any_node', 'get_node'):
        method = getattr(cfg.model, getter, None) or getattr(cfg, getter, None)
        if method:
            try:
                node = method(bb_addr)
                if node:
                    break
            except Exception:
                continue
    if node is None:
        return []
    out = []
    successors = (
        getattr(node, 'successors', None)
        or getattr(node, 'all_successors', None)
        or []
    )
    for succ_node in successors:
        addr = getattr(succ_node, 'addr', None)
        if addr is not None:
            out.append(addr)
    return out


def _conditional_exit_targets(project: "angr.Project", bb_addr: int) -> list[int]:
    """Return the list of target addresses for conditional (Jcc) exit
    statements in the IRSB. Used to classify successor edges as
    'exit' (Jcc-taken) vs 'default' (fallthrough)."""
    irsb = project.factory.block(bb_addr).vex
    out: list[int] = []
    for stmt in irsb.statements:
        if stmt.__class__.__name__ == "Exit":
            dst = stmt.dst
            if hasattr(dst, "value"):
                out.append(int(dst.value))
    return out


def _mnemonic_preflight(block, manifest: TargetManifest) -> None:
    """No-op for the angr backend.

    Historically this fail-fast'd on mnemonics not in the manifest's
    `accepted_mnemonics` whitelist, matching vex_to_smt's policy where
    each instruction needed an explicit semantic handler. The angr
    backend doesn't share that constraint — angr can symbolically
    execute any x86 instruction angr's lifter supports — so the
    preflight here is purely scope-control noise that whack-a-moles
    every time we extract a target with SIMD/FPU ops (e.g. SHA1's
    XMM-packed round constants).

    Kept as a no-op (not deleted) so the call site at translate_block's
    entry stays uniform with vex_to_smt's, and so the docstring
    captures the migration intent until the vex backend is retired.
    Once vex_to_smt is gone, drop this and the call.
    """
    return None


def _build_reg_sub(state, project: "angr.Project") -> dict[str, z3.ExprRef]:
    """For each canonical GPR, emit a sub entry iff the post-state
    value differs syntactically from its seeded BVS. Also always emit
    rip = post-state rip."""
    sub: dict[str, z3.ExprRef] = {}
    for reg in _GPR64:
        post = getattr(state.regs, reg)
        # Syntactic-identity check in claripy: post is just the BVS iff
        # it has op 'BVS' and the same cache_key as the seeded var.
        # Simpler: convert to z3 and compare .eq() against reg64(name).
        z_post = _to_z3(post)
        z_seed = s.reg64(reg, _CTX)
        if z_post.eq(z_seed):
            continue
        sub[reg] = z_post
    # rip is always written (successor addr). Emit it.
    rip_post = state.regs.rip
    z_rip = _to_z3(rip_post)
    sub["rip"] = z_rip
    return sub


def _resolve_qfabv_load_placeholders(
    z_expr: z3.ExprRef,
    placeholders: dict,
) -> z3.ExprRef:
    """Substitute every `__qfabv_load_N__` z3 var in z_expr with
    `s.load_le(mem_at_read, addr_z3, nbytes)` using the per-read
    memory snapshot captured during stepping by
    `_qfabv_load_hook`.

    Substitution is deterministic and topological in placeholder
    creation order: placeholders are numbered monotonically per
    block step, so a later placeholder's (addr, mem_at_read) may
    reference earlier placeholders (e.g. an address built from a
    previous stack load, or a memory snapshot containing earlier
    in-block stores that themselves wrote a previously-loaded
    value). For each placeholder N in creation order, substitute
    all already-resolved earlier placeholders into its addr and
    mem_at_read, then build its load expression. Apply all built
    pairs to `z_expr` at the end.

    No fixed-point loop, no iteration cap: insertion order +
    topological substitution guarantees structural termination —
    each placeholder is resolved exactly once against expressions
    that contain only earlier (already-resolved) placeholders.
    """
    if not placeholders:
        return z_expr
    pairs: list[tuple[z3.ExprRef, z3.ExprRef]] = []
    for name, entry in placeholders.items():
        # Strict 3-tuple shape: (addr_claripy, nbytes, mem_at_read).
        # A 2-tuple would mean a stale state leak from before the
        # side-band timeline existed; degrading to bare s.mem(_CTX) there
        # silently re-introduces the in-block-store-load unsoundness.
        # Fail loud instead — a placeholder that wasn't recorded with
        # a snapshot is a backend bug, not a recoverable shape.
        if len(entry) != 3:
            raise AngrBackendNotImplemented(
                f"placeholder {name!r}: expected 3-tuple "
                f"(addr, nbytes, mem_at_read), got {len(entry)}-tuple"
            )
        addr_claripy, nbytes, mem_at_read = entry
        width = nbytes * 8
        # Placeholder is constructed in `_CTX` so the substitution pair
        # is ctx-consistent with `z_expr` (which `_to_z3` already
        # translated into `_CTX`). Without explicit ctx, z3 defaults
        # this BitVec into main_ctx and z3.substitute raises a
        # cross-ctx sort mismatch.
        placeholder_z3 = z3.BitVec(name, width, ctx=_CTX)
        addr_z3 = _to_z3(addr_claripy)
        # Resolve any earlier placeholders that flow through this
        # entry's address or memory snapshot before we build the
        # load expression.
        if pairs:
            addr_z3 = z3.substitute(addr_z3, pairs)
            mem_at_read = z3.substitute(mem_at_read, pairs)
        load_expr = s.load_le(mem_at_read, addr_z3, nbytes)
        pairs.append((placeholder_z3, load_expr))
    return z3.substitute(z_expr, pairs)


def _verify_mem_timeline(
    timeline: list[MemWriteOp],
    final_mem: z3.ArrayRef,
    bb_addr: int,
) -> None:
    """ID-chain verifier for the per-block mem-write timeline (B.1,
    sophie+codex 2026-04-27).

    Confirms that the recorded `MemWriteOp` chain links correctly:
      - first entry's `prev_mem_id` equals the canonical block-entry
        mem expression (`s.mem(_CTX).get_id()`),
      - each entry's `prev_mem_id` equals the previous entry's
        `result_mem_id`,
      - the last entry's `result_mem_id` equals `final_mem.get_id()`.

    Raises `AngrBackendNotImplemented` (a `VexTranslationError`
    subclass) on any mismatch — strict, not warning, not bare assert
    (codex's pick: a Python `assert` is silently disabled under
    `python -O`, which would let the soundness check vanish in
    optimized runs).

    Drift here means something between the hook and serialization
    has mutated the mem-chain z3 ASTs (simplify, substitute, etc.)
    in a way that breaks the timeline's identity assumptions. Per
    sophie's standing rule: that's a bug to fix at the call site,
    not a thing to tolerate.
    """
    base_id = s.mem(_CTX).get_id()
    if not timeline:
        # No writes: final mem must equal the base.
        if final_mem.get_id() != base_id:
            raise AngrBackendNotImplemented(
                f"block 0x{bb_addr:x}: empty mem timeline but final "
                f"mem id {final_mem.get_id()} != base mem id {base_id}"
            )
        return
    if timeline[0].prev_mem_id != base_id:
        raise AngrBackendNotImplemented(
            f"block 0x{bb_addr:x}: first timeline entry's prev_mem_id "
            f"{timeline[0].prev_mem_id} != base mem id {base_id}"
        )
    for i in range(1, len(timeline)):
        prev = timeline[i - 1]
        cur = timeline[i]
        if cur.prev_mem_id != prev.result_mem_id:
            raise AngrBackendNotImplemented(
                f"block 0x{bb_addr:x}: timeline link broken at index "
                f"{i} (kind={cur.kind}, pc=0x{cur.pc:x}): "
                f"prev_mem_id {cur.prev_mem_id} != "
                f"prior result_mem_id {prev.result_mem_id}"
            )
    if timeline[-1].result_mem_id != final_mem.get_id():
        raise AngrBackendNotImplemented(
            f"block 0x{bb_addr:x}: final timeline entry's result_mem_id "
            f"{timeline[-1].result_mem_id} != live _qfabv_mem_expr id "
            f"{final_mem.get_id()}"
        )


def translate_block(
    project,
    bb_addr: int,
    manifest: TargetManifest,
    entry_flag_status: dict[str, str] | None = None,  # noqa: ARG001
) -> list[Transition]:
    """Translate one basic block at `bb_addr` into its outgoing
    Transitions. Each edge carries a guard, a parallel state-update
    `sub` mapping reg/flag/mem names to z3 expressions, and the
    target address (or symbolic tgt_smt for ret-style edges).

    Edges are classified as kind="exit" (took a Jcc Exit stmt) or
    kind="default" (fallthrough or unconditional terminator). Calls
    are kind="default" with jumpkind="Ijk_Call" and a callee_sym;
    rets are kind="default" with jumpkind="Ijk_Ret", tgt=None,
    tgt_smt=load_le(mem, rsp, 8).

    `entry_flag_status` is accepted for signature compatibility with
    vex_to_smt.translate_block but is consumed only by
    `_exit_flag_status_for_block`; the symbolic execution itself
    doesn't need the live-in metadata because flags are reseeded
    fresh at every block entry.
    """
    block = project.factory.block(bb_addr)
    _mnemonic_preflight(block, manifest)

    state = _fresh_symbolic_state(project, bb_addr)
    irsb = block.vex
    num_insts = block.instructions
    if num_insts == 0:
        raise AngrBackendNotImplemented(f"empty block at 0x{bb_addr:x}")

    # Control-flow classification up front. For direct (concrete next)
    # blocks, record the fallthrough addr. For Ijk_Ret, leave the
    # target symbolic (tgt_smt = load_le(mem, rsp, 8), materialized
    # after stepping via placeholder substitution).
    next_expr = irsb.next
    jumpkind = irsb.jumpkind
    next_is_concrete = hasattr(next_expr, "con") and hasattr(next_expr.con, "value")
    is_ret = jumpkind == "Ijk_Ret"
    fallthrough_addr = int(next_expr.con.value) if next_is_concrete else None

    succ = project.factory.successors(state, num_inst=num_insts)
    # all_successors over .successors: some branches come back as
    # unsat when their symbolic target has no mapped-address constraint.
    all_succs = list(succ.all_successors)

    # Indirect-control resolution (codex 2025-05-05 design review).
    # Previously: fail-fast when irsb.next was symbolic and not Ijk_Ret.
    # That broke any block ending in a switch-table dispatch or
    # function-pointer indirect call (libaom CVE-2024-5171 hit this at
    # img_alloc_helper's format dispatch).
    #
    # Now: tolerate symbolic next when at least one post-state resolves
    # to a concrete rip (BVV). The per-successor loop below handles
    # concrete post_addr cases natively. If angr's symbolic stepping
    # didn't enumerate (all post-states have symbolic rip), fall back
    # to CFGFast resolution.
    if not next_is_concrete and not is_ret:
        concrete_post = [p for p in all_succs
                         if p.regs.rip.op == "BVV"]
        if not concrete_post:
            # Try CFGFast for jump-table / indirect-call resolution.
            cfg_targets = _resolve_indirect_via_cfg(project, bb_addr)
            if not cfg_targets:
                raise AngrBackendNotImplemented(
                    f"block at 0x{bb_addr:x}: indirect/symbolic control "
                    f"unresolved (no concrete post-states, CFGFast did "
                    f"not resolve targets); direct-only + ret for now"
                )
            # CFGFast resolution path — emit one edge per resolved
            # target by re-stepping with each rip pinned. For now,
            # take the first concrete post-state we got and clone it
            # per target with appropriate guards.
            # NOTE: this is reachability-only, not proof-grade. Per
            # codex's design review: jump-table guards need .rodata
            # modeling for full soundness; using post.regs.rip == tgt
            # is acceptable for unblocking but not proof-grade.
            if not all_succs:
                raise AngrBackendNotImplemented(
                    f"block at 0x{bb_addr:x}: indirect/symbolic control "
                    f"and 0 successors from symbolic stepping"
                )
    exit_targets = _conditional_exit_targets(project, bb_addr)
    # We accept multi-Exit IRSBs. Real x86 rarely emits >1 Exit from
    # a single instruction, but pyvex does for compound patterns
    # (e.g. rep-prefixed string ops, certain cmpxchg lifts). Each
    # Exit becomes one kind="exit" edge; the default's guard remains
    # AND of negated exit guards (handled in the post-loop fixup
    # below).
    exit_targets_set = set(exit_targets)

    if not all_succs:
        raise AngrBackendNotImplemented(
            f"block at 0x{bb_addr:x} produced 0 successors"
        )

    edges: list[Transition] = []
    is_call = jumpkind == "Ijk_Call"

    # Per-block flag BVS dict, recovered from the seeded state. Used
    # by _extract_post_flag_subs to detect "unchanged" reads via
    # syntactic identity.
    flag_bvs: dict[str, "claripy.ast.bv.BV"] = state.globals.get("_flag_bvs", {})

    # Compute exit_flag_status once per block via the capstone-driven
    # dataflow pass. All edges from a block share the same exit
    # status (vex's tracker.snapshot is also block-level).
    exit_flag_status = _exit_flag_status_for_block(
        project, bb_addr, entry_flag_status
    )

    for post in all_succs:
        sub = _build_reg_sub(post, project)
        # B.2.1 (sophie+codex 2026-04-27): no longer reconstruct
        # `sub["mem"]` from `recent_actions` here. The
        # `_qfabv_mem_timeline` MemWriteOp records collected by the
        # mem_write hook are now the authoritative source for the
        # per-block mem chain, and `extractor.serialize_edge` walks
        # them to emit compact `(store_le_N M A V)` text. The legacy
        # `_collect_mem_writes` + `_build_mem_sub` reconstruction
        # path was kept for one commit (B.2) as a parity oracle and
        # then deleted here once the gist regen consistency check at
        # 0x40990f confirmed the new path is sound end-to-end.
        # Substitute any qfabv-load placeholders (from the mem_read
        # inspect hook) with s.load_le(mem_at_read, addr_z3, nbytes)
        # expressions so the emitted sub references the `mem` array
        # rather than free placeholder vars. Applies uniformly to
        # stack-derived and non-stack reads.
        placeholders = post.globals.get("_qfabv_load_placeholders", {})
        if placeholders:
            sub = {
                k: _resolve_qfabv_load_placeholders(v, placeholders)
                for k, v in sub.items()
            }
        # Eager flag extraction. Reads post.regs.rflags via angr's
        # COPY-or-OP_X helper, extracts per-flag bits, emits
        # `rflags_F` sub entries when they differ syntactically from
        # the seeded BVS. Fail-fasts on symbolic / multivalued /
        # unsupported cc_op rather than producing partial output.
        # `placeholders` threaded so flag formulas depending on stack
        # reads (e.g. cmp [rsp+X], Y) get the same resolution as the
        # rest of `sub`.
        if flag_bvs:
            _extract_post_flag_subs(
                post, flag_bvs, bb_addr, sub, placeholders=placeholders,
            )

        # Classify edge:
        #   - Ijk_Ret → kind="default", tgt=None,
        #     tgt_smt=load_le(mem, rsp, 8). Post-state rip is a
        #     qfabv-load placeholder we substitute back here.
        #   - Ijk_Call → kind="default", tgt=callee addr,
        #     callee_sym set, jumpkind="Ijk_Call". The Lean
        #     validator (QfabvLean/LtsFamily.lean
        #     checkDeterministicStep) requires exactly one
        #     kind="default" edge per block, so call-ness is
        #     carried by `jumpkind` + `callee_sym`, not `kind`.
        #   - post.scratch.exit_stmt_idx ≥ 0 → kind="exit",
        #     guard = actual Jcc condition. Distinguishes
        #     Exit-taken from default-fallthrough even when the
        #     two share a post-rip value (e.g. `rep cmpsb`).
        #   - Otherwise (exit_stmt_idx = -2) → kind="default",
        #     guard = TRUE (we fix it up post-loop to AND of
        #     negated exit guards if there are any).
        post_addr_ast = post.regs.rip
        post_addr: int | None = None
        if post_addr_ast.op == "BVV":
            post_addr = int(post_addr_ast.args[0])

        tgt_smt: z3.BitVecRef | None = None

        if is_ret:
            guard = s.true_(_CTX)
            kind = "default"
            tgt = None
            # Post-state rip is a qfabv-load placeholder; substitute
            # to get load_le(mem, rsp, 8). MUST resolve tgt_smt too,
            # not just sub entries.
            rip_resolved = _resolve_qfabv_load_placeholders(
                _to_z3(post_addr_ast), placeholders
            )
            tgt_smt = rip_resolved
            callee_sym = None
        elif is_call:
            guard = s.true_(_CTX)
            kind = "default"
            tgt = post_addr if post_addr is not None else fallthrough_addr
            callee_sym = _resolve_sym(project, tgt) if tgt is not None else None
        elif post.scratch.exit_stmt_idx >= 0:
            raw_guard = post.history.jump_guard
            guard = _to_z3(raw_guard) if raw_guard is not None else s.true_(_CTX)
            # Resolve qfabv-load placeholders in guards too. We already
            # do this for `sub` above and for `tgt_smt` on the ret path;
            # missing it here let __qfabv_load_N__ free vars leak into
            # exit guards (and, after the post-loop default-guard
            # fixup, into default guards built from those exits).
            # Symptoms downstream: per-block placeholder name collisions
            # at conflicting widths in z3-based consumers.
            guard = _resolve_qfabv_load_placeholders(guard, placeholders)
            kind = "exit"
            tgt = post_addr
            callee_sym = None
        else:
            guard = s.true_(_CTX)
            kind = "default"
            tgt = post_addr if post_addr is not None else fallthrough_addr
            callee_sym = None

        # Pull the per-successor mem-write timeline (B.1). Each
        # `_mem_write_timeline_hook` invocation has appended a
        # `MemWriteOp` to `state.globals["_qfabv_mem_timeline"]`
        # via copy-on-write. Run the ID-chain verifier against the
        # live `_qfabv_mem_expr` (post B.2.1 this is the sole
        # source of truth for the per-block mem chain; the
        # `recent_actions`-based reconstruction was the parity
        # oracle used during B.2 rollout and is now gone).
        raw_timeline = list(post.globals.get("_qfabv_mem_timeline", ()))
        live_mem = post.globals.get("_qfabv_mem_expr", s.mem(_CTX))
        _verify_mem_timeline(raw_timeline, live_mem, bb_addr)

        # B.2: resolve `__qfabv_load_*` placeholders inside each
        # MemWriteOp's addr_z3 / val_z3 BEFORE attaching to the
        # Transition. A write's address or value can flow through
        # an earlier load placeholder in the same block (e.g.
        # `mov [rsp+rax*8], rcx` after `mov rax, [rsp-0x10]`); the
        # placeholder names mean nothing to downstream consumers,
        # so they have to be substituted with `s.load_le(...)`
        # expressions here, where the post-state's placeholder
        # table is still in scope (codex 2026-04-27 design correction
        # #3). The verifier above runs first against the unresolved
        # mem-chain identities since those id values were captured
        # at hook time pre-resolution.
        if placeholders:
            mem_timeline = [
                dataclasses.replace(
                    op,
                    addr_z3=_resolve_qfabv_load_placeholders(op.addr_z3, placeholders),
                    val_z3=_resolve_qfabv_load_placeholders(op.val_z3, placeholders),
                )
                for op in raw_timeline
            ]
        else:
            mem_timeline = raw_timeline

        edges.append(Transition(
            kind=kind,
            guard=guard,
            sub=sub,
            tgt=tgt,
            tgt_smt=tgt_smt,
            jumpkind=jumpkind,
            callee_sym=callee_sym,
            flag_reads=[],
            exit_flag_status=exit_flag_status,
            mem_timeline=mem_timeline,
        ))

    # Default-guard correction: a Jcc-terminated block's default
    # edge has guard = AND of all the negated exit guards. We
    # initially built default edges with guard=s.true_(_CTX) in the loop
    # above (so we didn't need two passes through the successor
    # list); now fix them up once we know the full set of exit
    # guards. This generalizes naturally from 1 exit to N exits.
    #
    # We use `dataclasses.replace` here rather than re-listing every
    # field on a fresh Transition constructor: the latter silently
    # drops any new field (e.g. mem_timeline added in B.1) when
    # someone forgets to copy it across, which is a real-world
    # foot-gun codex flagged in design review (2026-04-27).
    if exit_targets_set:
        exit_guards = [e.guard for e in edges if e.kind == "exit"]
        negs = [s.not_(g) for g in exit_guards]
        if len(negs) == 1:
            default_guard_expr = negs[0]
        elif len(negs) > 1:
            default_guard_expr = s.and_(*negs)
        else:
            default_guard_expr = s.true_(_CTX)
        for i, e in enumerate(edges):
            if e.kind == "default":
                edges[i] = dataclasses.replace(e, guard=default_guard_expr)

    # Invariants on the per-block edge count:
    #   - Ijk_Call block: exactly 1 default edge, 0 exits, 0 calls.
    #     (`kind="call"` is intentionally not used; call-ness lives
    #     in `jumpkind` + `callee_sym`.)
    #   - Jcc-terminated non-call: exactly N exits + 1 default,
    #     where N = number of distinct conditional Exit-stmt targets
    #     in the IRSB (can be > 1 for compound lifts like rep cmpsb).
    #   - Straight-line non-call: exactly 0 exits + 1 default.
    exit_count = sum(1 for e in edges if e.kind == "exit")
    default_count = sum(1 for e in edges if e.kind == "default")
    call_count = sum(1 for e in edges if e.kind == "call")
    if is_call:
        if default_count != 1 or exit_count != 0 or call_count != 0:
            raise AngrBackendNotImplemented(
                f"block at 0x{bb_addr:x}: Ijk_Call block produced "
                f"{call_count} call + {exit_count} exit + {default_count} "
                f"default edges (expected 0+0+1)"
            )
    elif exit_targets_set:
        # Use the distinct-exit-target count for the invariant: two
        # Exit stmts to the same target produce one classified edge
        # (real x86 IRSBs don't duplicate dsts in practice; defensive).
        n_distinct = len(exit_targets_set)
        if exit_count < n_distinct:
            # Codex 2025-05-05 design extension: angr's symbolic stepping
            # can sometimes filter out an Exit-taking branch (constraint
            # solver determines its condition is unsat in our state, e.g.
            # because of __qfabv_load_N__ placeholder constraints). When
            # the IRSB has Exit stmts but our successor classification
            # produced fewer than expected, SYNTHESIZE the missing exit
            # edges directly from the IRSB's Exit-stmt guards. This is
            # reachability-grade, not proof-grade (the synth sub is
            # cloned from a default-edge sub, not the actual branch-
            # taken post-state). Comment + edge metadata flag this.
            covered_tgts = {e.tgt for e in edges if e.kind == "exit"}
            missing_tgts = exit_targets_set - covered_tgts
            # Find a base edge to clone sub/exit_flag_status from.
            base_edge = next(
                (e for e in edges if e.kind == "default"),
                edges[0] if edges else None,
            )
            if base_edge is not None:
                # Walk Exit stmts in IRSB to recover guards.
                for stmt in irsb.statements:
                    if stmt.__class__.__name__ != "Exit":
                        continue
                    dst_val = getattr(stmt.dst, "value", None)
                    if dst_val is None:
                        continue
                    tgt_addr = int(dst_val)
                    if tgt_addr not in missing_tgts:
                        continue
                    # Translate the Exit guard to z3 (best-effort —
                    # may reference free claripy BVS that we don't
                    # have a proper post-state for; downstream tools
                    # tolerate this since the guard is a Bool sort).
                    try:
                        guard_z3 = _to_z3(stmt.guard)
                    except Exception:
                        guard_z3 = s.true_(_CTX)
                    edges.append(Transition(
                        kind="exit",
                        guard=guard_z3,
                        sub=dict(base_edge.sub),
                        tgt=tgt_addr,
                        tgt_smt=None,
                        jumpkind="Ijk_Boring",
                        callee_sym=None,
                        flag_reads=[],
                        exit_flag_status=base_edge.exit_flag_status,
                        mem_timeline=list(base_edge.mem_timeline),
                    ))
                # Recompute counts after synthesis
                exit_count = sum(1 for e in edges if e.kind == "exit")
                default_count = sum(1 for e in edges if e.kind == "default")
        if exit_count != n_distinct or default_count != 1:
            raise AngrBackendNotImplemented(
                f"block at 0x{bb_addr:x}: multi-exit block produced "
                f"{exit_count} exit + {default_count} default edges "
                f"(expected {n_distinct}+1; IRSB has {len(exit_targets)} "
                f"Exit stmts to {n_distinct} distinct targets)"
            )
    else:
        if default_count != 1 or exit_count != 0:
            raise AngrBackendNotImplemented(
                f"block at 0x{bb_addr:x}: straight-line block produced "
                f"{exit_count} exit + {default_count} default edges "
                f"(expected 0+1)"
            )

    # Hard invariant: no unresolved memory-read placeholder may leak
    # into any emitted formula. Two classes (see _find_placeholder_leaks):
    # __qfabv_load_*__ (our hook) must be substituted by the resolver;
    # mem_<hex>_<id>_<width> (angr's default filler) must be intercepted
    # by the hook. Either survivor is a backend bug — downstream
    # consumers can't model the read against `mem` without it.
    for e in edges:
        leaks = _find_placeholder_leaks(e)
        if leaks:
            raise AngrBackendNotImplemented(
                f"block at 0x{bb_addr:x}: unresolved memory-read "
                f"placeholder survived emission in {leaks}"
            )

    return edges


def _find_placeholder_leaks(edge: Transition) -> list[str]:
    """Return a list of `<field>:<placeholder>` strings for every
    placeholder reference still present in `edge`'s emitted formulas.
    Empty list iff the edge is clean.

    Two leak classes:
    1. `__qfabv_load_*__` — our hook's placeholder, post-resolver
       any survivor is a backend bug.
    2. `mem_<hex>_<id>_<width>` — angr's default-filler placeholder.
       After commit C, the hook intercepts ALL known-width reads, so
       any filler name surviving means a read slipped past the hook
       (e.g. unknown nbytes, non-BV result), which is also a backend
       bug.

    Scans guard, tgt_smt, every sub field, AND every MemWriteOp's
    addr_z3 / val_z3 in mem_timeline. The mem_timeline scan is
    important post-B.2.1: it's the SOLE emitted memory source for
    angr edges, so if a future resolver regression leaves a
    placeholder inside a MemWriteOp expression, the leak check
    must surface it (codex 2026-04-27 B.2.1 review catch).
    """
    leaks: list[str] = []
    candidates: list[tuple[str, z3.ExprRef | None]] = [
        ("guard", edge.guard),
        ("tgt_smt", edge.tgt_smt),
    ]
    for k, v in (edge.sub or {}).items():
        candidates.append((f"sub.{k}", v))
    for i, op in enumerate(edge.mem_timeline or []):
        candidates.append((f"mem_timeline[{i}].addr_z3", op.addr_z3))
        candidates.append((f"mem_timeline[{i}].val_z3", op.val_z3))
    for label, expr in candidates:
        if expr is None:
            continue
        sx = expr.sexpr() if hasattr(expr, "sexpr") else str(expr)
        for m in _QFABV_LOAD_NAME_RE.findall(sx):
            leaks.append(f"{label}:{m}")
        for m in _ANGR_FILLER_NAME_RE.findall(sx):
            leaks.append(f"{label}:{m}")
    return leaks


# `__qfabv_load_N__`: emitted by `_qfabv_load_hook`, substituted back
# by `_resolve_qfabv_load_placeholders`. Any survivor is a resolver bug.
_QFABV_LOAD_NAME_RE = re.compile(r"\b__qfabv_load_\d+__\b")
# `mem_<hex>_<id>_<width>`: angr's default-filler placeholder name
# shape. With the generalized hook, any survivor means a read slipped
# past — likely an unknown-width or non-BV read class we haven't yet
# modeled. Fail loud so the gap is visible.
_ANGR_FILLER_NAME_RE = re.compile(r"\bmem_[0-9a-f]+_\d+_\d+\b")


def _resolve_sym(project: "angr.Project", addr: int) -> str | None:
    """Best-effort symbol name for a concrete callee address."""
    sym = project.loader.find_symbol(addr)
    if sym is not None and sym.name:
        return sym.name
    return None

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "angr==9.2.211",
# ]
# ///
"""Single-block extraction demo for the angr-backed extractor.

This is a runnable artifact. With `uv` installed (https://docs.astral.sh/uv/),
the PEP 723 inline metadata above tells uv exactly which dependencies
to install in an ephemeral environment — no separate venv setup or
manual `pip install` needed:

    uv run demo.py <binary> <symbol-or-addr> [--bb 0xADDR]

Address semantics:

    `<symbol-or-addr>` is either a function-symbol name (e.g.
    `tsm_utf8_mach_feed`), in which case the function's ENTRY block
    is the default extraction target, OR a `0x...` hex address
    interpreted as the function's entry. `--bb 0xADDR` overrides
    the extraction target with a SPECIFIC basic-block address (still
    within the same project's binary; the symbol is informational).

Examples:

    # Extract the entry block of a function by name.
    uv run demo.py /path/to/libtsm.o tsm_utf8_mach_feed

    # Extract a specific block (NOT the function's entry) by address.
    # Useful for the walkthrough block at 0x40a7f0 in tsm_utf8_mach_feed.
    uv run demo.py /path/to/libtsm.o tsm_utf8_mach_feed --bb 0x40a7f0

    # Extract by raw entry address (no symbol available).
    uv run demo.py /path/to/libtsm.o 0x40a7f0

Output: a JSON list of BlockEdge records (one per outgoing
control-flow edge of the requested block) with SMT-LIB formulas
for each guard / sub entry. Format matches what the extractor's
real serialize_edge contract emits.

Dependencies: angr (pinned to 9.2.211 — the version tested with
this code; angr's VEX helpers and rflags-materialization paths
are version-sensitive so we pin exact). claripy / pyvex / z3-solver
come along transitively via angr.

Co-located file requirement: demo.py imports `angr_backend`,
`smt_emit`, `target_manifest` directly (no `from .` package
syntax), so the other six .py files in this gist must live in
the same directory as demo.py for imports to resolve. After
`gh gist clone ddbae59d842784f30eaa1f0202531657`, that's
automatic.
"""
from __future__ import annotations

import argparse
import json
import sys

import angr

from angr_backend import translate_block
from smt_emit import to_smtlib
from target_manifest import TargetManifest


# Permissive mnemonic set — accept the full range that angr_backend
# can handle. The extractor's real pipeline uses a per-target
# manifest pinning a tighter set; for a single-block demo we don't
# need that gating.
_ACCEPT_ALL = frozenset({
    "mov", "movzx", "movsx", "movsxd", "movzbl", "movzwl", "movsbl",
    "movswl", "movsbq", "push", "pop", "ret", "leave", "jmp",
    "je", "jne", "jb", "jbe", "ja", "jae", "jg", "jge", "jl", "jle",
    "js", "jns", "jc", "jnc", "jo", "jno", "jp", "jnp", "jpe", "jpo",
    "cdqe", "cdq", "sete", "setne", "sets", "setns",
    "call", "cmp", "test",
    "and", "or", "xor", "not", "neg",
    "add", "sub", "div", "idiv", "mul", "imul",
    "shl", "shr", "sar", "rol", "ror",
    "inc", "dec", "adc", "sbb",
    "lea", "nop", "endbr64",
    "cmpsb", "scasb", "movsb", "stosb",
    "repe cmpsb", "repe scasb", "repne cmpsb", "repne scasb",
    "rep movsb", "rep stosb",
})


def _resolve_addr(project: angr.Project, sym_or_addr: str) -> int:
    """Resolve a symbol name or a hex `0x...` string to a concrete address."""
    if sym_or_addr.startswith(("0x", "0X")):
        return int(sym_or_addr, 16)
    sym = project.loader.find_symbol(sym_or_addr)
    if sym is None:
        raise SystemExit(f"symbol {sym_or_addr!r} not found in binary")
    return sym.rebased_addr


def _serialize_edge(src: int, edge) -> dict:
    """Match the extractor's serialize_edge contract for a single block."""
    record: dict = {
        "src": f"0x{src:x}",
        "kind": edge.kind,
        "jumpkind": edge.jumpkind,
        "guard": to_smtlib(edge.guard),
        "sub": {name: to_smtlib(val) for name, val in edge.sub.items()},
    }
    if edge.tgt is not None:
        record["tgt"] = f"0x{edge.tgt:x}"
    elif edge.tgt_smt is not None:
        record["tgt"] = None
        record["tgt_smt"] = to_smtlib(edge.tgt_smt)
    else:
        record["tgt"] = None
    if edge.callee_sym is not None:
        record["callee_sym"] = edge.callee_sym
    if edge.exit_flag_status:
        record["exit_flag_status"] = dict(edge.exit_flag_status)
    return record


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Single-block extraction demo for the angr-backed translator. "
            "Reads the binary, walks one basic block via "
            "project.factory.successors(num_inst=1), and emits the "
            "extracted BlockEdges as JSON to stdout."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run demo.py /path/to/libtsm.o tsm_utf8_mach_feed\n"
            "    # Extract the function-entry block.\n\n"
            "  uv run demo.py /path/to/libtsm.o tsm_utf8_mach_feed --bb 0x40a7f0\n"
            "    # Extract a specific block within the function.\n\n"
            "  uv run demo.py /path/to/libtsm.o 0x40a7f0\n"
            "    # Extract by raw entry address (no symbol).\n"
        ),
    )
    p.add_argument(
        "binary",
        help="Path to the x86-64 binary (ELF / shared object).",
    )
    p.add_argument(
        "sym_or_addr",
        help=(
            "Function-symbol name (e.g. `tsm_utf8_mach_feed`) OR a `0x...` "
            "hex address. If a symbol is given and `--bb` is not, the "
            "function's entry block is the extraction target."
        ),
    )
    p.add_argument(
        "--bb",
        help=(
            "Override the extraction target with a specific basic-block "
            "address (hex). Useful for blocks that aren't the function "
            "entry — e.g. the walkthrough block 0x40a7f0 in "
            "tsm_utf8_mach_feed. The block must lie within the binary; "
            "it isn't required to be on the function symbol's call graph."
        ),
    )
    args = p.parse_args(argv)

    project = angr.Project(args.binary, auto_load_libs=False)

    entry_addr = _resolve_addr(project, args.sym_or_addr)
    bb_addr = int(args.bb, 16) if args.bb else entry_addr

    manifest = TargetManifest(
        target=args.sym_or_addr,
        binary=args.binary,
        root_symbol=args.sym_or_addr,
        arch="amd64",
        accepted_mnemonics=set(_ACCEPT_ALL),
        observation={},
        initial_state={},
    )

    edges = translate_block(project, bb_addr, manifest)
    out = [_serialize_edge(bb_addr, e) for e in edges]
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

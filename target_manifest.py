"""
Per-target instruction manifest.

A manifest pins exactly which x86 mnemonics (optionally + operand widths)
are accepted by the extractor for a given target. Extraction fails fast
on any instruction outside the manifest. The pinning is a discipline
constraint: it keeps the extractor's accepted-instruction surface from
silently expanding to "half of amd64" as new targets are added.

Format (JSON):
  {
    "target": "tsm_utf8_mach_feed",
    "binary": "reference/libtsm/libtsm.o",
    "root_symbol": "tsm_utf8_mach_feed",
    "accepted_mnemonics": [
      "movq", "movl", "movb", "movzbl", "movzwl", "movsbl",
      "pushq", "popq", "callq", "retq", "jmp", "leaq",
      "je", "jne", "jg", "jge", "jl", "jle", "jns",
      "cmpq", "cmpl", "testl",
      "addl", "addq", "subl",
      "andl", "orl", "xorl",
      "shll", "nop"
    ],
    "observation": {
      "kind": "return_plus_caller_struct",
      "return_reg": "rax",
      "return_width": 32,
      "caller_struct_arg": "rdi",
      "caller_struct_bytes": 8
    },
    "initial_state": {
      "registers": { "rip": "<entry>", "rsp": "<stack_top>" },
      "caller_struct_init": [ 0, 0, 0, 0, 0, 0, 0, 0 ]
    }
  }
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TargetManifest:
    target: str
    binary: str
    root_symbol: str
    arch: str
    accepted_mnemonics: set[str]
    observation: dict
    initial_state: dict
    # name → primitive-denotation key. For the multi-function extractor's
    # call-target classifier: if a call lands on a PLT stub whose symbol
    # is in this dict, it's a declared primitive. Empty or absent = no
    # primitives allowed; any PLT call fails-fast.
    primitives: dict = None

    def __post_init__(self):
        if self.primitives is None:
            self.primitives = {}

    @classmethod
    def load(cls, path: Path | str) -> "TargetManifest":
        data = json.loads(Path(path).read_text())
        return cls(
            target=data["target"],
            binary=data["binary"],
            root_symbol=data["root_symbol"],
            arch=data["arch"],
            accepted_mnemonics=set(data["accepted_mnemonics"]),
            observation=data["observation"],
            initial_state=data["initial_state"],
            primitives=data.get("primitives", {}),
        )

    def accepts(self, mnemonic: str) -> bool:
        return mnemonic in self.accepted_mnemonics

    def require(self, mnemonic: str, addr: int) -> None:
        if not self.accepts(mnemonic):
            raise UnacceptedInstruction(
                f"instruction 0x{addr:x} ({mnemonic!r}) not in manifest for target "
                f"{self.target!r}; accepted set: {sorted(self.accepted_mnemonics)}"
            )


class UnacceptedInstruction(Exception):
    """Raised when an instruction mnemonic is outside the target's manifest."""

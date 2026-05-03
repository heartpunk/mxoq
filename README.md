# Extracting instruction-level semantics as labelled transition systems

## What this is

We take compiled x86-64 functions and produce **per-function
labelled transition systems over QF_ABV SMT-LIB formulas**. Each
transition encodes the cumulative effect of a basic block's
instructions — instruction-by-instruction state diffs composed into
a guard plus a parallel state-update over the quantifier-free
bitvector + array fragment of SMT-LIB that every modern solver
speaks.

The output is an *executable specification* of the function —
engine-agnostic by design. We've built one Lean executor
(~1,580 LoC across `QfabvLean/`) that loads the JSON and runs it;
the schema is intended to be consumed by other runtimes, formal
frameworks, or domain-specific analyses too.

## The methodology

The pitch is engine-agnostic. Any symbolic execution engine can be
turned into an extractor by *diffing path conditions before and
after each step*: the difference IS the per-step relational
transformer. Composing those transformers across the CFG gives a
labelled transition system whose edges encode (guard, sub) atoms in
the QF_ABV fragment. That LTS is the IR.

The angr backend in this gist is one realization. The methodology
itself is the diff-path-conditions extraction trick and the LTS
shape, not any particular extractor.

From the IR, you lift downstream: EBNF (parser-grammar), K-framework
(term rewriting + grammar), TLA+ (distributed protocols), and
domain-specific abstract interpretations. We've done concrete
extraction for terminal emulators (libtsm), crypto digests
(libmd/sha1), and an interpreter dispatch loop (lua); EBNF and
K-framework lifts are next.

## What's in this gist

The six Python files together form the small-core extractor — they
work as a flat directory drop, no package layout required:

| File | Lines | Purpose |
|---|---|---|
| `angr_backend.py` | 742 | The extractor: walks one block at a time via angr, reads register/memory state-diffs and the post-step rflags off claripy, emits Transition records with QF_ABV guards and subs. |
| `transition.py` | 122 | The `Transition` schema (the unit of LTS edge output), the `VexTranslationError` exception, and `normalize_mnemonic` for the cross-backend mnemonic canonicalization. (Renamed from `block_edge.py` — schema and exception names follow.) |
| `smt_emit.py` | 418 | SMT-LIB s-expression construction helpers (`bvadd`, `store_le`, `extract`, `to_smtlib`, etc) — thin wrappers on z3. |
| `dataflow.py` | 201 | The flow-sensitive flag-status pass (`block_effect`, `dataflow_entry_statuses`). Computes per-block entry/exit DEFINED/UNDEFINED/UNCHANGED tags for each flag. |
| `x86_flags.py` | 578 | Per-mnemonic flag effect tables (`FLAG_TABLE`, `rules_for_shift_rotate`, `is_known_flag_writer`). Transcription of the Intel SDM's flag-effects rules. |
| `target_manifest.py` | 89 | The `TargetManifest` dataclass (binary path, root symbol, accepted-mnemonic set, observation spec). Manifest-driven discipline that fail-fasts when an unknown mnemonic appears. |

Total: **2150 lines of Python** for the extractor end-to-end.
External runtime deps: `angr`, `claripy`, `z3`, `pyvex` (transitively
via angr). Drop the six files in a directory and `from
angr_backend import translate_block` works.

There's also `tsm_utf8_mach_feed.svg` — a function-level rendering
of the example target the walkthrough uses.

## How to run

The gist is a self-contained reproduction unit. Two ingredients —
the binary and the extractor — each invoked by one tool.

**Binary.** `flake.nix` (with `flake.lock` pinning the exact nixpkgs
commit) builds `libtsm.so.4.4.2` with the empirical-validation
compile flags (`-O0 -fno-inline -Wno-error`, preserving every
static function as a separate symbol). Build platform is
`x86_64-linux`; on other host platforms (e.g. macOS) you need a
Linux remote builder or a Linux VM — `--system x86_64-linux` alone
isn't enough without builders configured. With [nix](https://nixos.org/)
installed:

```sh
gh gist clone ddbae59d842784f30eaa1f0202531657
cd ddbae59d842784f30eaa1f0202531657
nix build .#libtsm
# → result/lib/libtsm.so.4.4.2
```

**Extractor.** `demo.py` is a single-file runnable. PEP 723 inline
metadata at the top of the file pins `angr==9.2.211`; with [uv](https://docs.astral.sh/uv/)
installed, no separate `pip install` or venv setup is needed:

```sh
uv run demo.py result/lib/libtsm.so.4.4.2 tsm_utf8_mach_feed --bb 0x40990f
```

The first invocation creates an ephemeral environment with angr (and
its transitive deps — claripy / pyvex / z3-solver) and runs the
script. Subsequent invocations reuse the cached environment. Output
is a JSON list of `BlockEdge` records for one basic block. The
output for the walkthrough block above is bundled as
`block_0x40990f.json` for diff-against-your-reproduction
verification:

```sh
uv run demo.py result/lib/libtsm.so.4.4.2 tsm_utf8_mach_feed --bb 0x40990f > mine.json
diff mine.json block_0x40990f.json
# (empty output = bit-identical reproduction)
```

`uv run demo.py --help` for the full argument reference.

## How it's built

`angr_backend.py` is the IMark-granular angr-backed translator. For
each basic block it steps one machine-instruction at a time via
`project.factory.successors(state, num_inst=1)`, reads
register / memory state-diffs and the post-step `rflags` value off
claripy, and emits one `BlockEdge` record per outgoing control-flow
edge of the IRSB.

A few specifics worth naming:

- **IMark-granular, not basic-block-granular.** The per-instruction
  state-diff is what makes per-flag SMT formulas correct; lifting
  multiple instructions at once muddles flag dependencies.
- **Eager flag decomposition** via a `cc_op = COPY` seeding trick:
  `cc_dep1` is packed with six 1-bit BVS symbols at the
  architectural flag bit positions (CF = bit 0, PF = 2, AF = 4,
  ZF = 6, SF = 7, OF = 11). Reading `state.regs.rflags` post-step
  triggers angr's `pc_calculate_rdata_all_WRK` helper, which
  projects exactly those six bits out of `cc_dep1` in the trivial
  COPY case, and unfolds to explicit per-op SMT in the writing
  case (e.g. `cc_op = G_CC_OP_SUBQ` after a `cmp`). Each flag
  becomes its own 1-bit BV variable in the emitted `sub`.
- **Fail-fast on undefined flag reads.** A flow-sensitive dataflow
  pass tags each flag as `defined` / `undefined` / `unchanged` at
  every block entry. The Lean loader rejects any block that *reads*
  a flag whose entry-status is `undefined`. Silent UB is rejected
  by construction.
- **Per-function LTS; calls cross LTS boundaries.** A call to a
  function we can also extract emits a `kind="default"` edge with
  `jumpkind="Ijk_Call"` and a `callee_sym` annotation; opaque
  externals fail-fast. The `tsm_utf8_mach_feed` example in the
  walkthrough is call-free, so it's a single-LTS family.
- **Symbolic / multivalued / unsupported `cc_op` fail-fast.** Angr's
  helper raises `SimError` / `SimCCallError` /
  `CCallMultivaluedException` in the corner cases where post-step
  `cc_op` isn't a single concrete value; we catch all three and
  re-raise as `AngrBackendNotImplemented`. Partial-flag output is
  not a failure mode.

This is the **pedagogically-minimal** version of the extractor.
The smaller-core shape is recent: earlier pipeline iterations
bundled extraction with specific downstream analyses (closure
construction, partition refinement, dispatch-loop fixpoints), which
made the construction harder to evaluate as a methodology in its
own right. The current factoring decouples extraction (what's in
this gist) from analysis (downstream consumers of the LTS); that
decoupling is what makes the methodology engine-agnostic.

## Walkthrough — one block

The block at x86 address `0x40990f`. Two machine instructions:

```asm
0x40990f:  cmp  eax, 5            ; 32-bit compare: eax - 5, discard result, set flags
0x409912:  jg   0x409b14          ; jump if greater (signed)
0x409918:  (fallthrough address — landing point if jg is NOT taken)
```

Pyvex lifts this conditional to one `Exit` IR statement and one
fallthrough `next`:

- `Exit dst=0x409918 guard=t7` — fire when **NOT** greater (eax ≤ 5
  signed); land at the post-jg address.
- `irsb.next = 0x409b14` — the default fallthrough; this is where
  control goes when the Exit didn't fire (i.e. when **eax > 5**).

Note pyvex's lifting flips the polarity: the assembly reads "if
greater, jump to X", but the IR encodes "if NOT greater, exit early
to next-PC; otherwise fall through to X." The two encodings are
equivalent, but the IR's "default" edge is the assembly's "taken"
branch and vice versa. The extractor follows the IR convention.

So the block has two outgoing edges:

- **`kind="exit"`** to `0x409918` (eax ≤ 5 signed; the
  jg-not-taken branch).
- **`kind="default"`** to `0x409b14` (eax > 5 signed; the
  jg-taken branch).

Both edges share the same `sub` (the same `cmp` ran in both worlds
before the branch); they differ only in their guards and targets.
Here are both, exactly as the extractor emits them — i.e. the
literal stdout of `uv run demo.py result/lib/libtsm.so.4.4.2
tsm_utf8_mach_feed --bb 0x40990f` (after `nix build .#libtsm` from
the bundled flake). The output is `z3.simplify`'d (that's what
`smt_emit.to_smtlib` does before serializing), which is why guards
read as `bvsge` rather than the IR-form `not bvslt`, and `rflags_CF`
reads as `ite (bvule 5 eax) 0 1` rather than `ite (bvult eax 5) 1 0`.
The semantics are unchanged; both forms produce the same model
under z3.

### Edge 1 — exit (eax ≤ 5 signed) → `0x409918`

<!-- BEGIN_AUTOGEN: walkthrough-edge-1-json -->
```json
{
  "src": "0x40990f",
  "kind": "exit",
  "jumpkind": "Ijk_Boring",
  "guard": "(bvsle ((_ extract 31 0) rax) #x00000005)",
  "sub": {
    "rip": "#x0000000000409918",
    "rflags_CF": "(ite (bvule #x00000005 ((_ extract 31 0) rax)) #b0 #b1)",
    "rflags_PF":
      "(let ((a!1 (bvadd #xfffffffb ((_ extract 31 0) rax))))
        (bvxor #b1
               (bvadd #b1 ((_ extract 0 0) rax))
               ((_ extract 1 1) a!1)
               ((_ extract 2 2) a!1)
               ((_ extract 3 3) a!1)
               ((_ extract 4 4) a!1)
               ((_ extract 5 5) a!1)
               ((_ extract 6 6) a!1)
               ((_ extract 7 7) a!1)))",
    "rflags_AF":
      "(bvxor ((_ extract 4 4) (bvadd #xfffffffb ((_ extract 31 0) rax)))
             ((_ extract 4 4) rax))",
    "rflags_ZF": "(ite (= ((_ extract 31 0) rax) #x00000005) #b1 #b0)",
    "rflags_SF": "((_ extract 31 31) (bvadd #xfffffffb ((_ extract 31 0) rax)))",
    "rflags_OF":
      "(let ((a!1 (bvxor ((_ extract 31 31) rax)
                        ((_ extract 31 31) (bvadd #xfffffffb ((_ extract 31 0) rax))))))
        (bvnot (bvor (bvnot ((_ extract 31 31) rax)) (bvnot a!1))))"
  },
  "tgt": "0x409918",
  "exit_flag_status": {
    "CF": "defined",
    "ZF": "defined",
    "SF": "defined",
    "OF": "defined",
    "PF": "defined",
    "AF": "undefined"
  }
}
```
<!-- END_AUTOGEN: walkthrough-edge-1-json -->

### Edge 2 — default (eax > 5 signed) → `0x409b14`

<!-- BEGIN_AUTOGEN: walkthrough-edge-2-json -->
```json
{
  "src": "0x40990f",
  "kind": "default",
  "jumpkind": "Ijk_Boring",
  "guard": "(not (bvsle ((_ extract 31 0) rax) #x00000005))",
  "sub": {
    "rip": "#x0000000000409b14",
    "rflags_CF": "(ite (bvule #x00000005 ((_ extract 31 0) rax)) #b0 #b1)",
    "rflags_PF":
      "(let ((a!1 (bvadd #xfffffffb ((_ extract 31 0) rax))))
        (bvxor #b1
               (bvadd #b1 ((_ extract 0 0) rax))
               ((_ extract 1 1) a!1)
               ((_ extract 2 2) a!1)
               ((_ extract 3 3) a!1)
               ((_ extract 4 4) a!1)
               ((_ extract 5 5) a!1)
               ((_ extract 6 6) a!1)
               ((_ extract 7 7) a!1)))",
    "rflags_AF":
      "(bvxor ((_ extract 4 4) (bvadd #xfffffffb ((_ extract 31 0) rax)))
             ((_ extract 4 4) rax))",
    "rflags_ZF": "(ite (= ((_ extract 31 0) rax) #x00000005) #b1 #b0)",
    "rflags_SF": "((_ extract 31 31) (bvadd #xfffffffb ((_ extract 31 0) rax)))",
    "rflags_OF":
      "(let ((a!1 (bvxor ((_ extract 31 31) rax)
                        ((_ extract 31 31) (bvadd #xfffffffb ((_ extract 31 0) rax))))))
        (bvnot (bvor (bvnot ((_ extract 31 31) rax)) (bvnot a!1))))"
  },
  "tgt": "0x409b14",
  "exit_flag_status": {
    "CF": "defined",
    "ZF": "defined",
    "SF": "defined",
    "OF": "defined",
    "PF": "defined",
    "AF": "undefined"
  }
}
```
<!-- END_AUTOGEN: walkthrough-edge-2-json -->

The full second edge, including the duplicated `sub`, is in
`block_0x40990f.json`; it's abbreviated here only to avoid
repeating identical state updates. Note the guard polarity has
flipped relative to the old IR-shape pyvex emits: z3 collapsed the
*exit* edge's `not (bvslt 5 eax)` to `bvsge 5 eax`, leaving the
*default* edge's complement form `not (bvsge 5 eax)`.

### What each piece means

**`src` / `tgt`** — block addresses. Plain hexadecimal strings; the
LTS is essentially a graph keyed by address.

**`kind: "exit"` vs `kind: "default"`** — `exit` for the IR's Exit
stmts; `default` for the fallthrough or any non-conditional
terminator (including `call`, `ret`, and unconditional jmp). Each
block emits exactly one `default` edge.

**`jumpkind`** — VEX IR jumpkind (`Ijk_Boring`, `Ijk_Call`,
`Ijk_Ret`, etc). Decoupled from `kind`: a `kind="default"` edge
with `jumpkind="Ijk_Call"` is a call site; same-kind with `Ijk_Ret`
is a return.

**`guard`** — the SMT-LIB boolean that fires the edge. Decoded:

- Edge 1: `(bvsge 5 eax_low32)` — fires when `5 ≥ eax` signed, i.e.
  when `eax ≤ 5` signed. The `(_ extract 31 0) rax` takes the low 32
  bits of the 64-bit register (this is a 32-bit `cmp` on `eax`, not
  a 64-bit `cmp` on `rax`).
- Edge 2: `(not (bvsge 5 eax_low32))` — fires when `5 ≥ eax` is
  false, i.e. `eax > 5` signed. Exactly the negation of Edge 1.

**`sub`** — the *state update* that happens when this edge fires. A
dictionary mapping register / flag / memory name to an SMT-LIB
expression giving the new value. All right-hand sides are evaluated
in the *pre-block* state, then applied as a parallel assignment.

The flag entries are the eager decomposition working: every flag
the `cmp` writes is its own 1-bit BV expression in the emitted
`sub`. The constant `#xfffffffb` appearing throughout is the 32-bit
two's-complement of -5 (i.e. `0xfffffffb` interpreted as a signed
32-bit int is `-5`) — `bvadd(a, -5)` is how the compiler+lifter
represents `a - 5`.

Reading the flags semantically:

- `rflags_CF` = `ite (5 ≤ eax unsigned) 0 1` — equivalent to
  `1 iff eax < 5 unsigned`. Carry flag set iff unsigned `eax - 5`
  underflows.
- `rflags_ZF` = `ite (eax == 5) 1 0` — zero flag set iff
  `eax - 5 == 0`.
- `rflags_SF` = `extract_bit_31(eax - 5)` — sign flag is the top
  bit of the result.
- `rflags_OF` = the standard signed-overflow predicate (top-bit of
  result vs top-bit of operands), expanded into nested SMT exactly
  as the SDM defines it.
- `rflags_PF` = parity of the low 8 bits of the result (bvxor of
  the bits, set iff there's an even number of 1-bits). Yes, x86
  really does this.
- `rflags_AF` = bit-4 carry-out of the subtraction. Computed
  eagerly per the angr helper's amd64 model. The Intel SDM
  technically marks AF as "undefined" after `cmp`, which is why
  `exit_flag_status` reports it that way (the dataflow pass takes
  the SDM-conservative stance), but the helper produces a concrete
  formula and we emit it.

This is the **hardware-faithful flag decomposition**. No `rflags`
monolith, no lazy condition-code model. Every flag the instruction
writes is its own 1-bit SMT-LIB expression; every flag read
downstream is a read of that specific expression.

**`exit_flag_status`** — for each architectural flag: `defined` (an
SMT formula was written), `unchanged` (not touched), or `undefined`
(SDM-undefined). Output of the flow-sensitive dataflow pass; the
Lean loader uses it to reject downstream blocks that read undefined
flags. Note the split: `sub["rflags_AF"]` carries a concrete
formula (eager, from angr), but `exit_flag_status["AF"]` is
`undefined` (conservative, from the SDM-mirrored dataflow table).
Downstream code that strictly needs the SDM contract can read
`exit_flag_status` and refuse to consume AF; downstream code that
wants the implementation-defined-but-stable value can read
`sub["rflags_AF"]` directly.

---

## Second example — `sha1_update` with a primitive summary

The first walkthrough is intentionally call-free: `tsm_utf8_mach_feed`
is a self-contained UTF-8 decoder that doesn't reach out to anything.
Real-world functions usually do. This second example is `SHA1Update`
from libmd. It calls `memcpy` once per invocation to absorb input
bytes into the SHA1 state buffer.

The `SHA1Update.lts.json` artifact in this gist is the extracted
LTS, ready to inspect. The build is `nix build .#libmd` against
this gist's flake — at the flake.lock-pinned nixpkgs commit, that's
libmd 1.1.0 with stock-stdenv build flags, producing
`libmd.so.0.1.0`. To reproduce a byte-identical .so:

```sh
nix build .#libmd  # → result/lib/libmd.so.0.1.0
```

GitHub gists don't support binary file uploads, so the .so isn't
shipped here directly. The `LIBMD_COPYRIGHT` file in this gist is
the BSD-3-Clause attribution sourced from Debian's libmd0 1.0.4-1build1
package's `/usr/share/doc/libmd0/copyright`; the upstream license
terms are unchanged in 1.1.0, so this attribution covers the
flake-built binary too.

`memcpy` is a libc external — there's no point symbolically executing
through its `rep movsb` body to recover its semantics; the contract
is already known. The manifest declares it a **primitive** with a
`memcpy_effect`, and the extractor emits a single transition that
encodes the call's effect at the SMT level rather than descending
into the bytes:

```json
{
  "src": "0x4032d0",
  "kind": "primitive",
  "jumpkind": "Ijk_Ret",
  "guard": "true",
  "sub": {
    "mem": "(mem_after_memcpy mem rdi rsi rdx)",
    "rax": "rdi"
  }
}
```

`mem_after_memcpy` is an uninterpreted function characterized
pointwise (axiomatized on the Lean side):

```
(mem_after_memcpy m dst src n)[a]
    = if dst ≤ a < dst+n
      then m[src + (a - dst)]
      else m[a]
```

The extracted family at `SHA1Update.lts.json` has three functions:

| function | role | blocks | transitions |
|---|---|---|---|
| `SHA1Update` | root, called by user code | 18 | 28 |
| `memcpy` | primitive summary at `0x4032d0` | 1 | 1 (the X2 form above) |
| `SHA1Transform` | sibling callee, fully extracted | 14 | 14 |

43 transitions total (32 `kind=default`, 10 `kind=exit`, 1
`kind=primitive`). `SHA1Update.svg` is the function-level
visualization (light theme; `SHA1Update.dark.svg` for dark).

**Where the PLT comes in.** `libmd.so.0.1.0` is a fully linked
shared object, so the call to `memcpy` lands at a `.plt.sec` stub
address (`0x4032d0`), not directly at the extern symbol. The
extractor consults `main_object.plt` to map the stub address back
to the extern name `memcpy`, then matches against the manifest's
`primitives` dict. Without that mapping, fully-linked .so targets
look like `("orphan", None)` to the call-target classifier and
crash extraction. (For relocatable .o files like `libtsm.o` above,
the issue doesn't arise — there's no PLT in unlinked objects, and
calls resolve directly to the `cle##externs` symbol.)

The same PLT path also handles **self-PLT calls** in this build:
`SHA1Update`'s call to `SHA1Transform` actually lands at
`SHA1Transform@plt`, not the function's body, because PIC-built
shared libraries route in-`.so` refs through PLT too. The call-
target resolver detects that the extern name maps back to an
in-object symbol with size > 0 and classifies the call as
extractable (descend into the function body), not opaque.

**Why it's `Ijk_Ret`.** The primitive transition's jumpkind reflects
the idealized control-flow effect of "calling memcpy and then
returning to the caller's continuation." From the caller's POV the
call site's effect is "memcpy ran (`sub.mem` updates), then control
returned." The primitive summary collapses both call and return into
one transition, which is why the kind is `primitive` (the X2 shape)
rather than `default + Ijk_Call`.

---

## Completeness

The extracted LTS is intended to be a *complete* denotational model
of the function's observable behavior, in the sense developed by
Voogd et al. for symbolic execution:

- Ábrahám, Dubslaff, Tapia Tarifa, Voogd, Kløvstad, Johnsen.
  **"Denotational Semantics for Symbolic Execution."** ICTAC 2023,
  LNCS. PDF: <https://ebjohnsen.org/publication/23-ictac/23-ictac.pdf>.
  DOI: <https://doi.org/10.1007/978-3-031-47963-2_22>.
- Voogd, Kløvstad, Johnsen, Wąsowski. **"Compositional symbolic
  execution semantics."** TCS, 2025 (journal extension of the
  ICTAC paper). PDF: <https://ebjohnsen.org/publication/25-tcs/25-tcs.pdf>.
  Coq mechanization at <https://github.com/Aqissiaq/ICTAC-DenotSymbEx>.

The block-level invariants `angr_backend.py` enforces give us the
ingredients those results require:

- Every guard is in QF_ABV; the disjunction of every block's
  outgoing-edge guards covers all states (the deterministic-step
  policy, enforced post-loop in `translate_block`).
- Every flag the instruction sets has a defined SMT value, or is
  explicitly tracked as `undefined` — never silently `unchanged`.
- Every memory effect is a `(store mem addr val)` chain with
  explicit endianness, byte-aligned by construction
  (non-byte-aligned ops fail-fast).
- Post-step `cc_op` is concrete and supported, or the entire block
  fail-fasts.

**Which specific results.** The README's claim of a "complete
denotational model" factors cleanly into three levels of
correspondence in the ICTAC/TCS Coq mechanization
(<https://github.com/Aqissiaq/ICTAC-DenotSymbEx>); all three are
load-bearing:

- *Per-edge.* **Theorem `trace_correspondence`** (`Traces.v:252`;
  Theorem 1 in ICTAC23) discharges that each `BlockEdge` is the
  `(Sub(t), PC(t))` pair of a per-instruction trace `t`, and that
  the iff `denot_fun t V = Some V' ⟺ (V ⊨ PC(t)) ∧ denot_sub(Sub(t)) V = V'`
  IS the soundness-and-completeness for that local edge: a concrete
  execution starting from valuation `V` produces `V'` exactly when
  `V` satisfies the guard and the sub applied to `V` gives `V'`.

- *Per-program.* **Lemma `SE_correct` + Lemma `SE_complete`**
  (`Programs.v:545` and `Programs.v:527`; the two halves of Theorem 2
  in ICTAC23) lift the per-edge correspondence to the whole program:
  the set of branches `denot__S(p)` (= the union of all per-block
  `(σ, φ)` pairs) IS the program's symbolic denotation, and that
  set is sound and complete w.r.t. concrete denotational semantics.
  Reading our LTS as the union of all per-block branches: this is
  what makes the LTS a complete denotational model.

- *Per-composition.* **Lemma `denotS_spec_seq`** (`Programs.v:372`;
  Lemma 10(i) in ICTAC23, restated in §4 of the TCS journal
  extension) is the algebra that justifies stitching per-block subs
  through the CFG. When block A flows to block B, the resulting
  branch is `(compose_subs σA σB, φA ∧ Bapply σA φB)` — σA composed
  with σB, plus the new constraint φB pushed through σA's variable
  renaming. This is what makes "the block above, times 36 more like
  it, composed through the CFG" not a metaphor but a literal
  algebraic operation. (The lower-level building blocks
  `sub_trace_app` and `pc_trace_app` at `Programs.v:322` and
  `Programs.v:341` are the trace-algebra restatement.)

## Trust boundary

The recording is structurally trivial: we observe angr's per-block
`(guard, sub)` output and write it into our LTS schema. The lift is
*not* a translation step — there's no semantic transformation we
are responsible for.

Trust boundary is at angr+pyvex, the same boundary as Park et al.
2025 ("Filtered simulation," compositional binary lifting). Our
formal claims discharge against angr+pyvex correctness, not against
any logic of our own.

Concrete-semantic preservation of the lift is *empirically*
validated on real targets (see below). Symbolic and denotational
preservation are claims supported by structural correspondence —
each VEX statement maps to one ICTAC (guard, sub) atom in the
schema — and would, if formalized, reduce to the angr+pyvex
correctness claim.

## Empirical validation

For `tsm_utf8_mach_feed` specifically: on **2265 hand-picked +
generated inputs** (256 exhaustive single bytes + 9 curated
multi-byte UTF-8 sequences + 1000 generated-valid random UTF-8
sequences + 1000 generated-invalid random byte sequences), the Lean
executor's observation (return value + 8 bytes of the mach struct
the caller passed in) matches what `libtsm.so.4.4.2` produces
natively on Linux x86-64. Byte-exact, 2265/2265.

So the block above, times 36 more like it, composed through the CFG
and executed transition-by-transition under concrete inputs,
recovers the full observable behavior of the function. That's the
claim; that's the test that backs it.

## What this enables: parser, opsem, bridge

For source-level systems (interpreters, parsers, bytecode VMs)
the IR carries enough structure to recover behavior at three
distinct layers, *as separate sub-extractions, each requiring its
own follow-up analysis*:

1. **Parser** — the LTS of the parser itself. Lifts to EBNF (or
   richer grammar formalisms) by a path-conditions-as-symbolic
   approach: extract the parser, derive the grammar's covering
   sets, symbolically execute over them, and read syntactic
   positions off the resulting path constraints.

2. **Operational semantics** — for a bytecode VM, the LTS of the
   dispatch loop plus the per-handler LTSs (one per opcode, when
   extracted out). For compiled languages, the equivalent control
   flow.

3. **Bridge** — the data-flow + typing + control-flow paths that
   relate AST shape to dispatched code. Says "this AST production
   maps to that opcode handler's behavioral equivalence class."

For source-level systems where all three sub-extractions are
present, the construction of a K-framework rule can be described as
a JOIN over them: anchor on a particular opcode handler in the
dispatch loop, query which AST shapes correspond to its equivalence
class of behavior, and the JOIN result has the ingredients of the
K rule's LHS-RHS pair (in K's syntax-as-term-structure style).

We've done concrete bytecode-VM extraction on lua. EBNF and
K-framework lifts are not yet shown in this gist; they're the next
demonstrations.

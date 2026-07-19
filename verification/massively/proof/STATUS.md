# Formal proof status

This ledger states exactly what the Lean development proves and what remains
outside its theorem boundary.

## Proved central theorem

`InstructionNormalization.compileToCore_run_decode_correct` proves that, for
every word width, finite processor count, finite nonempty shared memory,
finite instruction table, well-typed instruction program, initial
configuration, and finite step count, executing the compiled Massively Core
program and decoding its barrier state gives exactly the instruction-level
priority-CRCW PRAM execution.

Equality includes shared memory, every processor's program counter (including
the halted state), and every processor's recursive-product register value.
The same `steps` argument occurs on both sides: one source instruction clock
is represented by exactly one normalized PRAM round and one Core bulk round.

The checked proof chain is:

```text
InstructionPRAM.Program
  -- InstructionNormalization.normalize --> PriorityCRCW.Program
  -- Compilation.compile                --> MassivelyCore.Program
```

### Instruction normalization

- `InstructionPRAM.Instruction` independently defines local, read, write,
  branch, and halt instructions with per-processor control flow.
- `InstructionNormalization.evaluate_liftLocalRead`,
  `evaluate_liftLocalTransition`, and `evaluate_liftReadTransition` prove the
  typed substitution used to encode user registers behind a program counter.
- `normalized_readAddressAt`, `normalized_nextRegistersAt`,
  `normalized_writeEnabledAt`, `normalized_writeAddressAt`, and
  `normalized_writeValueAt` preserve every local instruction observation.
- `normalized_priorityOwnerAt` preserves priority write conflicts.
- `normalize_step_correct`, `normalize_run_correct`, and
  `normalize_run_decode_correct` prove one-step and every-finite-run
  normalization correctness.

### Normalized PRAM to Core

- `Compilation.winner_compile_correct`: Core conflict resolution chooses the
  same least processor identifier as priority-CRCW.
- `Compilation.scheduled_winner_compile_correct`: every permutation of the
  compacted proposal schedule chooses the same owner.
- `Compilation.memory_compile_correct` and `compile_step_correct`: controlled
  scatter and one complete Core round preserve the normalized PRAM state.
- `Compilation.compile_run_decode_correct`: the result holds for every finite
  normalized run.
- `InstructionNormalization.compileToCore_run_decode_correct`: composition of
  both layers, without treating either transition function as the other one's
  definition.

`Example.instructionProgram_two_steps` is a concrete read-then-write
instruction program compiled through both layers. The earlier collision
example remains as a direct test of priority conflict resolution.

## Checked cost facts

- `normalizedRoundCount_factor_one`: instruction clocks and normalized/Core
  bulk rounds have a factor-one count.
- `Compilation.compile_nodeCount` and `compileToCore_nodeCount`: the second
  compiler does not duplicate normalized scalar syntax.
- `compiled_proposalCount_le_processors`: at most one proposal is emitted per
  logical processor per round.
- `MassivelyCore.Program.runCost_*` and
  `InstructionNormalization.compileToCore_runCost_*`: under the documented
  general sort/group/scatter schedule, `s` steps have exactly `5s` launches,
  `4s` barriers, and zero atomics. Scalar work retains its explicit
  state-dependent routing term.

These are symbolic schedule counts, not calibrated GPU elapsed-time claims.
There is not yet a machine-independent asymptotic work/span refinement theorem
connecting this schedule to physical GPU execution.

## Current scope

- finite synchronous executions and finite instruction tables;
- a separate program counter for every processor;
- one local, read, write, branch, or halt instruction per processor clock;
- at most one shared-memory access per conventional instruction;
- pre-step snapshot reads and priority-CRCW writes, where the least processor
  identifier wins;
- finite words, bounded indices, booleans, and recursive product registers;
- typed pure scalar terms with constants, product construction/projection,
  conditionals, finite-index cases, explicit sharing, and signature-supplied
  pure primitives;
- no hidden global-memory, atomic, barrier, allocation, or launch effect
  inside scalar terms.

The explicit idle address carried by an instruction program witnesses that
shared memory is nonempty. Processor and instruction counts may otherwise be
zero.

## Deliberate boundary

This establishes semantic completeness of the mathematical Core calculus
relative to the formal conventional priority-CRCW PRAM model. It does not
formalize the informal set called "all parallel programs" independently of
that external model.

The mathematical Core is also not yet a refinement model of the Rust API,
CubeCL IR, a device compiler, or GPU hardware. Therefore the theorem does not
by itself establish that the current implementation realizes every formal
Core primitive. That implementation-refinement layer is separate from
Traversal Algebra and from the semantic completeness theorem above.

Docker pins and checks the Lean artifact; it does not turn implementation
conformance tests into formal premises.

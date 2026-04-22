# Ship Checklist (P0/P1) — ETE Completion Gate

Date: 2026-04-22

This checklist defines the minimum work needed to call ETE "complete" for this branch, based on the current project docs and code state.

## Recheck Snapshot (2026-04-22)

Status based on direct verification in this workspace:

- [x] CUDA build works: `nvcc` found at `/usr/bin/nvcc`, `make USE_CUDA=1 USE_PTHREAD=1 -j$(nproc)` succeeds.
- [x] Full CUDA test gate is green: `make USE_CUDA=1 USE_PTHREAD=1 test` passes.
- [x] Quant correctness/parity evidence exists and passes: `test_quant` passes including `cuda_parity_q4k` (max err 0.000000).
- [x] Backend/device scaffolding tests pass: `test_backend` passes (`backend_cpu_gemm_vtable`, thread consistency, CUDA guard).
- [ ] E2E oracle smoke is still not done: `tests/test_e2e_smoke.c` is still a stub test.
- [ ] Phase 1.5 benchmark closure remains open in status docs (`handover.md` still lists Phase 1.5 metrics as TBD/pending).

---

## P0 — Must close before declaring ETE complete

### P0.1 CUDA build + test gate is green
- [ ] `make USE_CUDA=1 USE_PTHREAD=1 -j$(nproc)` succeeds on the target machine.
- [ ] `make USE_CUDA=1 USE_PTHREAD=1 test` succeeds.
- [ ] CUDA toolchain issue is resolved (`nvcc` available in PATH).

**Exit criterion:** no compile/link/runtime failures in CUDA-enabled build.

### P0.2 Device-selection behavior is verified end-to-end
- [ ] Runtime test: `--device auto` selects CUDA when available.
- [ ] Runtime test: `--device auto` falls back to CPU when CUDA backend is unavailable.
- [ ] Runtime test: `--device cuda` fails clearly when CUDA backend is unavailable.
- [ ] Runtime test: `--device cpu` always forces CPU path.

**Exit criterion:** all four behavior paths are reproducible and documented.

### P0.3 GPU parity + numerical drift check
- [ ] GPU path compared against CPU on representative prompts (short + long prompt).
- [ ] Compare logits / generated greedy tokens for first N tokens (recommend N=8 or 16).
- [ ] Drift tolerance defined and enforced (target around 1e-5 where practical).

**Exit criterion:** parity report exists with pass/fail and accepted tolerance.

### P0.4 E2E oracle smoke test (non-stub)
- [ ] Replace placeholder smoke with real TinyLlama greedy next-token assertion.
- [ ] Test is runnable in CI/dev with clear model path strategy.

**Exit criterion:** deterministic oracle smoke test is part of `make test` (or a clearly documented optional target if model artifact cannot be bundled).

### P0.5 Phase 1.5 benchmark closure
- [ ] Record TTFT + tok/s for CPU baseline vs current heterogeneous path.
- [ ] Include long-prompt benchmark case.
- [ ] Publish result in project status docs.

**Exit criterion:** status table has measured values (no TBD) and a reproducible command list.

---

## P1 — Important, but can follow P0 ship

### P1.1 Quant + GPU regression harness hardening
- [ ] Keep quant golden tests and add at least one mixed-path regression scenario.
- [ ] Add script/target to run parity suite quickly after CUDA kernel changes.

### P1.2 Thermal/perf guardrail policy
- [ ] Define accepted thermal range and throttle policy behavior.
- [ ] Add a concise perf note (when to inspect `metrics.log`, acceptable degradation window).

### P1.3 Documentation consistency cleanup
- [ ] Reconcile stale ETE notes that still mention old stubs.
- [ ] Keep one source-of-truth status section for current milestone state.

---

## Suggested 5-day closure sequence

### Day 1
- P0.1 toolchain/build gate
- P0.2 device-selection matrix validation

### Day 2
- P0.3 parity runs (short prompt set)
- P0.4 smoke-oracle replacement draft

### Day 3
- P0.3 long-prompt parity + drift report
- P0.4 smoke-oracle finalization

### Day 4
- P0.5 benchmark sweep and write-up

### Day 5
- Final rerun: build + tests + docs updated
- Mark ETE complete if all P0 criteria pass

---

## Ship Decision Rule

ETE is "ship complete" when all P0 items are checked with reproducible evidence (commands, outputs, and updated status docs). P1 items are post-ship hardening/perf hygiene.

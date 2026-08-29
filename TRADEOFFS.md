# Experiment limits

Read these limits before a GPU session and before Phase 4. Each limit includes the action that controls its risk.

## Summary

| # | Limit | Required action |
| --- | --- | --- |
| 1 | The fp16 GPU check covers one model path. | Let an uncovered kernel failure stop its Phase 1 run. |
| 2 | Bit reproduction needs matching hardware and software. | Record the GPU and package versions for each run. |
| 3 | Deterministic operations reduce training speed. | Use the measured Phase 1 rate for resource estimates. |
| 4 | The GPU idles during the end-of-session analysis. | Accept the few minutes; pull outputs, then stop the pod. |
| 5 | The bootstrap forks one worker per CPU core. | Let the parity cell verify it; a fork failure falls back to serial. |
| 6 | The run manifest has no lock. | Run one training process per phase. |
| 7 | Phase notebooks load shared code with `%run definitions.ipynb`. | Keep `definitions.ipynb` fixed during model runs. |
| 8 | Cached image pixels depend on Pillow. | Use one pinned Pillow version for the experiment. |
| 9 | The data pipeline preloads all images into RAM. | Keep the cache at 224 pixels; budget about 0.2 GB RAM per run. |
| 10 | The inference covers one cohort and fold assignment. | State this scope beside each contrast result. |
| 11 | Bootstrap bit-identity holds per machine, not across machines. | Run each reported bootstrap on one machine and keep its environment record. |

## 1. The fp16 GPU check covers one model path

The check runs one fp16 training step through the DKA-51 adapter and the mHC neck. It does not cover the other kernels or necks.

PyTorch stops if the target GPU lacks a deterministic operation. An uncovered path can fail during a Phase 1 kernel run.

Accept this small cost because Phase 1 runs use less compute than Phase 2 or Phase 3 runs.

## 2. Bit reproduction needs matching hardware and software

Fixed seeds and deterministic operations can reproduce bits on the same GPU model, driver, and package set.

A different GPU or library build can produce small numeric changes. Record the environment with each run.

If a retry uses different hardware, add the hardware change to the Phase 4 change record.

## 3. Deterministic operations reduce training speed

The code disables cuDNN benchmarking and requires deterministic algorithms. These choices reduce convolution throughput.

Keep these choices fixed to support repeat runs. Replace resource estimates with the Phase 1 measurement before Phase 2.

## 4. The GPU idles during the end-of-session analysis

The Phase 2 and Phase 3 analysis sections and Phase 4 use no GPU, and the whole sequence runs on the pod so the experiment finishes in one place. The GPU sits idle while they run.

The parallel bootstrap keeps that idle time to minutes, which costs cents. Accept it: one machine for every reported bootstrap and the report satisfies limit 11 by construction. Pull the outputs, then stop the pod. A manifest rerun elsewhere skips finished runs and repeats only the analysis.

## 5. The bootstrap forks one worker per CPU core

The C1-through-C4 bootstrap fans its 10,000 replicates across CPU cores with forked workers. Replicate indices come from one seeded serial stream, and a permanent 500-replicate parity cell proves the parallel and serial paths bit-identical before every full run.

If forking fails, the loop falls back to one process with identical output and a longer wall clock. Finish the GPU runs of a phase before its bootstrap either way. Limit 11 covers reproducibility across machines.

## 6. The run manifest has no lock

Each phase records finished runs in a manifest file. The resume logic skips runs with a successful manifest row.

Two training processes on the same phase can both claim the same run and write conflicting outputs. Run one training process per phase. One process also removes GPU memory contention.

## 7. Phase notebooks load shared code with `%run definitions.ipynb`

Every phase notebook executes `definitions.ipynb` first. A change to `definitions.ipynb` changes every later phase.

Keep `definitions.ipynb` fixed while model runs execute. Each run records the SHA-256 of `definitions.ipynb` and of its own phase notebook. If a definition must change between phases, record the change in the Phase 4 `EXPERIMENT_CHANGES` list.

## 8. Cached image pixels depend on Pillow

The cache stores each field-of-view crop as a 224-by-224 PNG. Pillow controls image decoding and resizing.

Build the cache with the pinned Pillow version. Keep that version for all phases. The study cache was built on 2026-08-23 with Pillow 12.0.0; its per-file hashes are in `phase0/artifacts/cache_manifest.csv`.

If Pillow or the image-processing code changes, build a new cache and add a Phase 4 change record.

## 9. The data pipeline preloads all images into RAM

Each run loads its cached 224-pixel crops into memory as one tensor, at most about 0.2 GB. No data-loader worker processes exist, so worker count cannot affect results or determinism.

Batch order and augmentation come from generators seeded by fold, training seed, sample size, stage, and epoch. Matched cells therefore see identical orders and draws even when their epoch counts differ.

This design assumes the 224-pixel cache. A larger input size would need a new memory budget and a change record.

## 10. The inference covers one cohort and fold assignment

The paired bootstrap represents uncertainty from held-out analysis groups. It does not represent new training seeds, folds, sites, or imaging systems.

The outer training pools overlap. Treat fold effects as descriptions of variation, not as five independent replications.

State this scope beside C1, C2, C3, and C4. A non-significant result does not prove equivalence.

## 11. Bootstrap bit-identity holds per machine, not across machines

The bootstrap replicate loops run in parallel across CPU cores. A parity cell in phase2 and phase3 proves the parallel and serial paths bit-identical on 500 replicates before every full run.

That proof covers the machine that runs it. A different CPU, BLAS build, or numpy version can shift floating-point results in the last digit, for the serial path as much as the parallel one.

Run each reported bootstrap start to finish on one machine. The analysis file records the environment; keep that record beside the result. This is the analysis-side counterpart of limit 2.

## Add a new limit

Add one summary row and one numbered section. State the limit, reason, and required action.

If a decision changes the experiment after registered training begins, add it to the Phase 4 `EXPERIMENT_CHANGES` list (protocol section 10). Keep the original result beside it. Earlier decisions belong in the docs and the provenance files, not in the deviation list.

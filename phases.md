# Retinal Adapter Experiment

Start the experiment at Phase 0. Complete each phase before you start the next phase.

All code lives in six notebooks at the project root: `definitions.ipynb` plus one notebook per phase. Each phase notebook starts with `%run definitions.ipynb` and passes state to later phases through files on disk. RUNBOOK.md describes the layout and the run procedure.

## Experiment question

The experiment measures two changes to a five-label retinal image classifier:

- a Dual-Kernel Adapter (DKA) after the third ResNet50 stage
- a manifold Hyper-Connection (mHC) neck after the pooled backbone features

The target labels are DR, NORMAL, ODC, MH, and DN. The primary score is macro AUC.

## Phase sequence

| Phase | Purpose | Start condition | Finish condition |
| --- | --- | --- | --- |
| Phase 0 | Prepare the data | The source images and label tables are available. | All data checks pass. |
| Phase 1 | Build and check the model pipeline | Phase 0 is complete. | All model checks and eight kernel runs pass. |
| Phase 2 | Compare models with the full training set | Phase 1 is complete. | All 35 runs and the C1-C3 analysis finish. |
| Phase 3 | Measure the effect of training-set size | Phase 2 is complete. | The stability gate and required runs finish. |
| Phase 4 | Create the final report | Phase 3 is complete. | The report and artifact hashes pass their checks. |

## Status on 2026-08-23

| Phase | State |
| --- | --- |
| Phase 0 | Complete. Every finish check passes. Artifacts and hashes: `phase0/artifacts/provenance.json`. |
| Phase 1 | The model checks pass on the local machine. The smoke test, the eight kernel runs, and the runtime measurement wait for the GPU pod. |
| Phase 2 | Waiting for Phase 1. The notebook is built and verified in its pending state. |
| Phase 3 | Waiting for Phase 2. The notebook is built and verified in its pending state. |
| Phase 4 | Waiting for Phases 0 through 3. The notebook is built and verified in its pending state. |

## Phase 0: Prepare the data

### Goal

Create one clean cohort and one fixed set of splits for the experiment.

### Work

1. Read `phase0/source/train_data.csv` and `phase0/source/val_data.csv`, merge them into one 2,208-row table, and read the image files in `images/`. The historical train and validation split is not reused.
2. Keep images with at least one positive target label.
3. Record the image path, labels, source, group identifier, and SHA-256 value.
4. Inspect the field-of-view crop for each image.
5. Find exact duplicates by SHA-256. Screen visual pairs with a 63-bit perceptual hash at Hamming distance 8 or less, then verify every candidate with seeded geometric matching on the original images. Pairs at or over 40 RANSAC inliers confirm, pairs under 18 reject, and the band between needs a human row in `phase0/source/duplicate_decisions.csv`.
6. Keep related images in the same analysis group.
7. Create five outer test folds.
8. Create one inner-validation set inside each outer training pool.
9. Create nested training subsets for 50, 100, 251, 501, and 1,044 images.

### Outputs

Phase 0 writes cohort and split files to `phase0/artifacts/`. It writes audit results to `phase0/reports/`.

### Finish checks

- The cohort contains 1,489 unique images.
- Every image passes the crop audit.
- No analysis group appears in two outer folds.
- Every evaluation set contains both values for each target.
- Every smaller training subset is part of the next larger subset.

### Result

Phase 0 completed on 2026-08-23 with every finish check passing. The cohort holds 1,489 images in 1,432 analysis groups. The duplicate audit confirmed nine near-duplicate pairs (eight by rule, one by human decision) forming eight clusters. The label balance per fold is in `phase0/reports/split_prevalence.csv`.

## Phase 1: Build and check the model pipeline

### Goal

Build the shared training system before the main comparisons begin.

### Work

1. Load the cohort and splits from Phase 0.
2. Build the image transforms and data loaders.
3. Build the frozen ResNet50 backbone, both adapters, and all neck designs.
4. Build the training, prediction, metric, threshold, and bootstrap functions.
5. Run shape, gradient, parameter-count, and deterministic-operation checks.
6. Run eight DKA kernel checks with fold 1 and seed 42.
7. Measure the run time and peak GPU memory.

The kernel checks use the inner-validation set. They do not read the outer test fold.

### Outputs

Phase 1 writes model-check results, kernel results, histories, and run-time measurements to `phase1/outputs/`.

### Finish checks

- All model checks pass.
- The kernel table contains eight successful runs.
- The kernel checks contain no outer-test predictions.
- The run-time summary contains measured values.

## Phase 2: Compare models with the full training set

### Goal

Measure the DKA effect, the mHC effect, and their interaction with 1,044 training images.

### Work

1. Run the four primary model combinations across five outer folds.
2. Run two capacity controls across the same five folds.
3. Run one linear probe across the same five folds.
4. Select each checkpoint and class threshold with inner-validation data.
5. Score the outer test fold once after selection ends.
6. Calculate the C1, C2, and C3 contrasts.
7. Calculate confidence intervals and adjusted p-values with 10,000 paired bootstrap samples. The replicates run in parallel across CPU cores after a permanent 500-replicate parity check proves the parallel and serial paths bit-identical.

Phase 2 uses seed 42 for all 35 runs.

### Outputs

Phase 2 writes checkpoints, predictions, metrics, run records, and analysis tables to `phase2/outputs/`.

### Finish checks

- All 35 runs succeed.
- Every run has a checkpoint, history, prediction file, and metadata file.
- The analysis contains C1, C2, and C3.
- The Phase 2 manifest contains one successful record for each run.

## Phase 3: Measure the effect of training-set size

### Goal

Measure how the DKA effect changes as the training set grows.

### Work

1. Run all primary model combinations at 50 images across five folds and three seeds.
2. Calculate the inner-validation stability value, `SD(50)`.
3. Continue if `SD(50) < 0.030`.
4. If the gate passes, run the remaining models for 100, 251, and 501 images.
5. Combine these results with the 1,044-image results from Phase 2.
6. Calculate C4, the DKA-effect change per tenfold increase in training images.
7. Calculate the C4 interval and adjusted p-value with 10,000 paired bootstrap samples. The replicates run in parallel across CPU cores after the same 500-replicate parity check as Phase 2.

The first step contains 60 runs. A passed gate adds 80 runs.

### Outputs

Phase 3 writes the gate result, run records, predictions, sample-size effects, and C4 analysis to `phase3/outputs/`.

### Finish checks

- A stopped gate has 60 successful runs and a clear not-estimated C4 record.
- A passed gate has 140 successful runs and a complete C4 result.

## Phase 4: Create the final report

### Goal

Combine the completed experiment results into one reviewable report.

### Work

1. Read the completed outputs from Phases 0 through 3.
2. Report the cohort, label balance, sources, and split sizes.
3. Report C1 through C4 with estimates, intervals, adjusted p-values, and fold effects.
4. Report the capacity controls, kernel checks, linear probe, and stability gate.
5. Report run times, failures, reruns, experiment changes, and limitations.
6. Create the final tables and figures.
7. Record a SHA-256 value for each report artifact.

### Outputs

Phase 4 writes the report, tables, figures, and artifact hashes to `phase4/outputs/`.

### Finish checks

- The report contains all required sections.
- The confirmatory table contains C1, C2, C3, and C4.
- The report explains a stopped Phase 3 gate without inventing a C4 value.
- The artifact-hash table covers every final output.

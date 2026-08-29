# Experiment run guide

Run Phase 0 through Phase 4 in order. Finish the checks for one phase before you start the next phase.

## Current state on 2026-08-23

- Phase 0 is complete on the local machine. Every finish check passes.
- All six notebooks exist and are verified: the model checks pass locally, and Phases 1 through 4 execute cleanly in their pending states.
- A study pod exists and is stopped: `abundant_crimson_octopus` (id `te2y7nrk9epazg`), one A100 SXM 80GB, secure cloud, EUR-IS-1, $1.59 per hour, image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, network volume at `/workspace`. A stopped pod bills only volume storage.
- The Runpod MCP server is connected, so pod control works from the Claude session.
- No GPU run has started. No project file is on the pod yet.

## Code layout

All experiment code lives in six notebooks at the project root:

| Notebook | Contents |
| --- | --- |
| `definitions.ipynb` | Package pins, constants, preprocessing, splits, models, training, metrics, thresholds, bootstrap. Defines functions and runs nothing. |
| `phase0.ipynb` | Cohort build, audits, folds, inner-validation sets, subsets, data checks. |
| `phase1.ipynb` | Model checks, kernel study, runtime measurement. |
| `phase2.ipynb` | Full-data factorial runs, capacity controls, linear probe, C1-C3 analysis. |
| `phase3.ipynb` | Stability gate, sample-size sweep, C4 analysis. |
| `phase4.ipynb` | Final report, tables, figures, artifact hashes. |

Each phase notebook starts with `%run definitions.ipynb`. Phases pass state through files on disk: `phase0/artifacts/`, `phase1/outputs/`, `phase2/outputs/`, `phase3/outputs/`, `phase4/outputs/`. There are no side `.py` modules and no test folders. Checks are notebook cells that raise on failure.

The label tables live at `phase0/source/train_data.csv` and `phase0/source/val_data.csv`, downloaded from the MuReD dataset (doi:10.17632/pc4mb3h8hz.1) on 2026-08-22. Their SHA-256 values:

```text
a40f340cba6dd7141a3194214434b51484186dec8903e25d096853b995a5a7ee  train_data.csv
d7967f05e7f422002e071cdd5288cf822185ac86ad9f237fe698b29915223fb7  val_data.csv
```

## Prepare the environment

1. Start the study pod named above. If it is gone, rent a new pod with the Runpod PyTorch 2.8.0 image and one A100 SXM.
2. Record the GPU model and driver version. Use the same GPU model for every phase.
3. Set `PYTHONHASHSEED=2026` as a pod environment variable before the pod starts. The template launches Jupyter at boot, and the server and its kernels must inherit the value.
4. Move the data in two directions. The pod pulls `images.zip` from the MuReD dataset (doi:10.17632/pc4mb3h8hz.1) and unzips it into `images/`; the datacenter link makes this faster than a home upload. The local machine pushes the light pieces: the six notebooks, the three docs, `phase0/source/`, `phase0/artifacts/` including the image cache, and `phase0/reports/`.
5. The training loop reads only the cache. The raw images serve the Phase 0 spot check, which also re-verifies the label-table hashes.
6. Open `definitions.ipynb` and run the `%pip install` cell once. It pins every package version.
7. Restart the kernel after the install cell, then run `definitions.ipynb` top to bottom.

## Cache the ResNet50 weights

1. Download `ResNet50_Weights.IMAGENET1K_V2` before the first training run.
2. Put the checkpoint in the Torch Hub cache of the pod.
3. Make sure that the preflight cell finds the cached checkpoint.

The notebooks block network downloads during model runs.

## Verify the image cache on the pod

Phase 0 builds the 224-pixel image cache and its manifest with the pinned Pillow version. They live in `phase0/artifacts/image_cache/` and `phase0/artifacts/cache_manifest.csv`.

1. Upload the cache and manifest with the project.
2. Run `phase0.ipynb` once on the pod. Its fast path verifies the manifest with a seeded 20-image spot check and finishes in seconds.
3. If Pillow or the image-processing code changes, delete the manifest, rerun Phase 0 in full, and record the change.

## Start Jupyter

Set `PYTHONHASHSEED=2026` before Python starts. Python cannot apply it after the interpreter starts. On the pod, the environment variable from the preparation step covers the Jupyter server that the template launches. On any other machine, launch Jupyter with:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=2026 jupyter lab
```

Run one training process. Two processes on the same phase can duplicate runs, because the run manifest has no lock.

## Run the GPU check

Run `run_mixed_precision_determinism_smoke_test()` on the target GPU before training. The check runs one fp16 training step through the DKA-51 and mHC path. A failure stops the session before the main training block.

## Run the phase sequence

1. Run `phase0.ipynb` top to bottom on any machine. No GPU is needed.
2. The duplicate audit screens image pairs with a perceptual hash and verifies every candidate on the original images with seeded geometric matching. Pairs at or over 40 RANSAC inliers confirm, pairs under 18 reject. If a pair lands between and has no human row in `phase0/source/duplicate_decisions.csv`, the notebook stops and writes the pair there with an empty decision. Fill in `confirm` or `reject` with `decision_source` set to `human`, then rerun the notebook. A finished image pass leaves a verified cache manifest, so the rerun skips the slow image stage.
3. Make sure every Phase 0 check cell passes.
4. Run `phase1.ipynb` on the pod. Finish the Phase 1 checks.
5. Run `phase2.ipynb` training on the pod. All 35 runs execute in sequence.
6. Run the Phase 2 analysis and bootstrap sections on the pod. The parity cell verifies the parallel bootstrap first, and the replicates fan out across the pod's CPU cores, so the analysis takes minutes.
7. Make sure Phase 2 writes `phase2/outputs/run_manifest.csv`.
8. Run the 60 Phase 3 gate runs, then the gate cell.
9. If the gate passes, run the 80 continuation runs.
10. Run the Phase 3 analysis and bootstrap on the pod.
11. Run `phase4.ipynb` on the pod. It needs no GPU and takes under a minute.
12. Pull the executed notebooks and every `phase*/outputs/` directory back to the local machine.
13. Stop the pod. A stopped pod bills only volume storage.

Running every phase on the pod keeps each reported bootstrap and the report on one recorded machine, which satisfies TRADEOFFS.md limit 11 by construction.

## Resume after a crash

Each training run appends one row to its phase manifest when it finishes. To resume after a crash or a stopped pod:

1. Relaunch Jupyter with the same environment variables.
2. Run the phase notebook top to bottom.
3. The run controller skips every run with a successful manifest row and continues with the rest.
4. Record the interruption in the run metadata.

## Record failures and changes

The run metadata records model-run failures. Add failures without attempt records to the Phase 4 manual list.

Record a design or execution change made after registered training begins in the Phase 4 `EXPERIMENT_CHANGES` list (protocol section 10). Keep the original result beside the change record. The list starts empty: nothing changed after training began, because training has not begun.

Data-provenance note, reported in the Phase 4 cohort section rather than as a change: on 2026-08-22, 221 RFMiD PNG files disappeared from the local `images/` folder before any experiment code ran. They were restored from the official `images.zip` (doi:10.17632/pc4mb3h8hz.1) with CRC checks against the zip records. `phase0/source/restored_files.csv` lists every restored file with its SHA-256. No split, subset, or model existed at that time, so the study is unaffected.

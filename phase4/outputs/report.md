# Retinal adaptation study: final report

Generated 2026-08-24T06:39:00Z from the frozen protocol `retinal_adaptation_study_v1.md`.

## Cohort and data rules

The cohort holds 1489 images from the cleaned 2,208-row MuReD table under the registered five-target rule. cohort.csv sha256: 9252929f5b55e091a27e22ab18a198c2ce7705661acb7dd57afa0136f7e76e00. Splits, subsets, and audits are in phase0/.

Data note: before Phase 0 ran, 221 RFMiD files disappeared from the local images folder and were restored from the official images.zip with CRC checks (phase0/source/restored_files.csv). No split, subset, or model existed at that time.

## Confirmatory results (C1-C4, Bonferroni across four tests)

Estimates and 98.75% intervals come before significance labels. The bootstrap is conditional on the trained models, the registered seeds, and the fixed fold assignment; it covers held-out analysis-group sampling only, and a non-significant result is not equivalence.

```text
  ci_high    ci_low contrast  estimate  p_adjusted    p_raw  replicates
 0.013579  0.001243       C1  0.007534    0.007999 0.002000       10000
-0.004442 -0.016435       C2 -0.010361    0.000400 0.000100       10000
 0.018294 -0.005061       C3  0.006575    0.652735 0.163184       10000
-0.001972 -0.010619       C4 -0.006277    0.003600 0.000900       10000
```

## Cells at 1,044 images (fold-averaged)

```text
       test_macro_auc_mean  test_macro_auc_sd  macro_f1_mean  hamming_mean  exact_match_mean  wall_seconds_mean  trainable_parameters
cell                                                                                                                                 
A                   0.9089             0.0084         0.6815        0.1218            0.6226             509.98                741162
B                   0.8952             0.0088         0.6499        0.1355            0.5836             530.06                740181
C                   0.9131             0.0107         0.6710        0.1173            0.6347             549.90                839574
D                   0.9060             0.0049         0.6615        0.1272            0.6018             527.70                838593
E                   0.9093             0.0106         0.6785        0.1224            0.6112             506.30                839658
F                   0.9073             0.0117         0.6606        0.1247            0.6091             514.40                838677
probe               0.8820             0.0134         0.6076        0.1616            0.4789             422.68                 10245
```

## Exploratory capacity contrasts (no family correction)

```text
 ci_high    ci_low            contrast  estimate p_adjusted    p_raw  replicates
0.012079  0.000458     capacity_effect  0.006263       None 0.006999       10000
0.006381 -0.004128 mhc_versus_capacity  0.001271       None 0.557544       10000
```

## Kernel study (exploratory, inner validation only)

```text
 large_kernel    n  inner_macro_auc  wall_seconds  peak_gpu_gb
            7   50         0.784448          36.0          0.6
           15   50         0.785150          35.6          0.6
           31   50         0.785184          35.8          0.6
           51   50         0.784955          40.8          0.6
            7 1044         0.913588         508.3          0.6
           15 1044         0.910951         509.5          0.6
           31 1044         0.913661         513.3          0.6
           51 1044         0.907544         463.1          0.6
```

## Failures, reruns, and changes

Failed or rerun rows: 0 (failed_runs.csv). Changes after registered training began: 0 (experiment_changes.csv).

## Scope

Inference covers this cohort and this fold assignment. Fold effects describe variation; the outer training pools overlap, so they are not five independent replications.

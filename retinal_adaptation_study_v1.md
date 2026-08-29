# Retinal Adaptation Study — Prospective Protocol (V1)

**Status:** Ready for Phase 0. No model in this protocol has started training.

**Primary question:** How do DKA and mHC affect macro AUC on a small,
imbalanced, multi-label retinal classification task? Do the two designs
complement each other?

**Dataset target:** A MuReD-derived cohort with 1,489 images and five labels:
DR, NORMAL, ODC, MH, and DN.



---

## 1. Scope and prior motivation

This study compares two recent adaptation designs:

- The [Dual-Kernel Adapter](https://arxiv.org/abs/2602.18888) (DKA) changes
  spatial adaptation of pretrained features.
- [Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880)
  (mHC) change residual-stream connectivity.

DKA and mHC act at different model locations. Their distinct functions make
the interaction test meaningful.

---

## 2. Data contract

### 2.1 Source and cohort construction

MuReD combines ARIA, STARE, and the RFMiD training set. Its first composite
version contains 2,451 images. The published cleaning process retains 2,208
images with 20 labels. The source is the
[MuReD paper](https://arxiv.org/abs/2207.02335) and dataset DOI
[`10.17632/pc4mb3h8hz.1`](https://doi.org/10.17632/pc4mb3h8hz.1).

The local `images/` inventory contains 2,451 files: 2,308 PNG and 143 TIFF.
The cleaned label table, not the raw file count, defines the study cohort.

Phase 0 constructs the study cohort from the official 2,208-image cleaned
table. An image enters the cohort when at least one target label is positive:

`DR OR NORMAL OR ODC OR MH OR DN`

The study keeps all five target columns. It ignores non-target columns during
model training, but preserves them in the provenance table. An image with a
target label and additional non-target labels remains in the cohort.

This rule is expected to produce 1,489 images. If it produces a different
count, Phase 0 stops. The dataset version, label mapping, and cleaning list
must be resolved before any split is created.

The cohort builder writes `cohort.csv` with these fields:

- `image_id`
- `relative_path`
- `source_dataset`
- `source_subject_id`, when available
- `duplicate_cluster_id`, when applicable
- `analysis_group_id`
- `DR`, `NORMAL`, `ODC`, `MH`, and `DN`
- the retained non-target labels
- `sha256`

The builder performs these data tests:

1. Every path resolves to one image.
2. Every image has one label row.
3. Image identifiers and SHA-256 values are unique. A duplicate SHA stops the build.
4. Every target value is 0 or 1.
5. Every included row has at least one positive target.
6. A row with `NORMAL=1` and any disease label is listed for source review.
7. Perceptual duplicate candidates are reviewed before splitting.
8. Confirmed near-duplicates receive one shared `duplicate_cluster_id`.

No historical train, validation, or test split is reused.

### 2.2 Outer folds and inner validation

The study creates five outer folds once. The folds remain fixed for every
cell, seed, sample size, and secondary control.

The base group key combines the source name and subject identifier when that
identifier exists. Otherwise, it uses the image identifier. Confirmed
near-duplicate links join their base groups. The `analysis_group_id` is a hash
of the sorted keys in each resulting connected component. All images from one
analysis group stay in one outer fold.

The splitter balances the five target labels and the three source datasets.
It uses iterative multi-label stratification with source indicators added as
balancing columns. Group constraints take priority over exact balance.
Randomized tie breaks use `SPLIT_SEED=2026`. The code version, package version,
seed, and final assignment are saved.

For a 1,489-image cohort, four outer folds contain 298 test images. One outer
fold contains 297 test images. The corresponding outer-training pools contain
1,191 or 1,192 images.

Each outer-training pool supplies one fixed inner-validation set. This split
uses the same group, label, and source rules as the outer split. No analysis
group can appear in both the fit pool and inner-validation set.

Four folds reserve 147 images. The fold with 1,192 outer-training images
reserves 148. Every fold therefore contains exactly 1,044 fit-eligible images.

The inner-validation set serves four purposes:

- warmup stopping
- checkpoint selection
- threshold selection
- the Phase 3 stability gate

The outer test fold is used only for final scoring. Model or threshold choices
cannot use outer-test results.

Every outer test fold and inner-validation set must contain both label values
for every target. The split builder stops if this requirement fails.

### 2.3 Nested training subsets

The symbol `n` means the number of images that update model parameters. The
inner-validation images do not count toward `n`.

The registered ladder is:

| Training images | 50 | 100 | 251 | 501 | 1,044 |
|---|---:|---:|---:|---:|---:|
| Seeds | 3 | 2 | 1 | 1 | 1 |

The seed identifiers are 42, 43, and 44. Seed 42 runs at every sample size.
Seed 43 runs at 50 and 100 images. Seed 44 runs at 50 images.

Subsets are nested within each fold and seed. The 50-image subset is inside
the 100-image subset for seeds that use both sizes. Seed 42 remains nested
through the complete ladder.

The subset builder uses constrained iterative stratification. Every subset
must contain at least one positive and one negative image for every target.
The builder stops if it cannot meet this requirement. It does not remove a
class from the training loss.

All four primary cells use the same subset for a matched fold, seed, and
sample size.

### 2.4 Image preprocessing

The preprocessing module is written and frozen during Phase 0. It applies the
same deterministic base processing to every image:

1. Decode the image as RGB.
2. Find `max_intensity` across the full red channel.
3. Set the foreground threshold to `0.06 * max_intensity`.
4. Scan the center row for the first and last values above the threshold.
5. Scan the center column for the first and last values above the threshold.
6. Crop the inclusive rectangle defined by those four detected edges.
7. Pad the crop to a black square without stretching the field of view.
8. Resize the square to 224 by 224 pixels with bilinear interpolation.

Square padding is centered. When an odd number of pixels is required, the
right or bottom edge receives the extra pixel.

If the field-of-view scan fails, Phase 0 lists the image for source review. It
does not apply a silent fallback.

Training augmentation occurs after resizing and before normalization. It uses
a random horizontal flip with probability 0.5. The rotation angle is uniform
on the continuous interval from -10 to +10 degrees. Rotation uses bilinear
interpolation and black fill.

Validation and test images receive no random augmentation. Vertical flips and
label-dependent augmentation are not used.

The final tensor uses the ImageNet normalization values:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

The preprocessing tests save example outputs from each source dataset. They
also save the module version and the cohort hash.

---

## 3. Factorial design and registered contrasts

### 3.1 Primary cells

The primary design is a 2 by 2 factorial:

| | Same-width residual neck | mHC neck |
|---|---|---|
| **Standard adapter** | A | C |
| **DKA** | B | D |

Each cell starts from the same ImageNet-pretrained ResNet50 weights. The
backbone stays frozen. The adapter, common neck stem, neck blocks, and
classifier head can update.

The four confirmatory contrasts are:

- **C1:** mHC package effect against the same-width residual neck
- **C2:** DKA design effect
- **C3:** mHC by DKA interaction
- **C4:** slope of the DKA design effect on `log10(n)`

C1-C3 use the full `n=1,044` runs with seed 42. C4 uses the complete registered
sample-size ladder. C1 includes the mHC router and its added parameters. The
secondary capacity control tests whether those added parameters explain C1.

For fold-averaged cell estimates, the exact formulas are:

```text
C1 = 0.5 * [(C - A) + (D - B)]
C2 = 0.5 * [(B - A) + (D - C)]
C3 = (D - C) - (B - A)
```

A positive C3 means that the DKA advantage is larger with mHC than with the
same-width residual neck.

These four contrasts form one confirmatory family. No other contrast can
become a headline claim.

### 3.2 Secondary capacity control

The primary residual neck matches the mHC transformation blocks. It does not
match the parameters used by the mHC router.

Both primary necks use the same insertion point, four transformations,
width-64 MLPs, LayerNorm, GELU, and zero-output initialization. Their planned
difference is the mHC stream-routing package. The router adds parameters, so
the secondary control checks that capacity difference.

A wider residual neck provides a secondary capacity control. It runs at
1,044 training images across both adapter designs and all five folds.

| Neck | Streams | MLP width | Neck parameters |
|---|---:|---:|---:|
| Same-width residual | 1 | 64 | 134,400 |
| mHC | 4 | 64 | 232,812 |
| Capacity-matched residual | 1 | 112 | 232,896 |

The mHC and capacity-control counts differ by 84 parameters, or 0.036% of the
mHC neck count. The common stem and classifier are identical and excluded
from this table.

The capacity contrasts are exploratory. They cannot change C1-C4 or their
multiplicity correction.

Let `E` be the standard-adapter capacity-control cell. Let `F` be the DKA
capacity-control cell. The registered exploratory summaries are:

```text
capacity effect = 0.5 * [(E - A) + (F - B)]
mHC versus capacity = 0.5 * [(C - E) + (D - F)]
```

The interpretation follows fixed rules:

- If mHC exceeds both residual controls, the result favors the stream design.
- If mHC exceeds only the width-64 control, ordinary capacity can explain it.
- If mHC exceeds neither control, the study finds no mHC advantage.

### 3.3 Contextual linear probe

A frozen-backbone ResNet50 linear probe runs across all five folds at full
data. It trains only a five-output classifier on the 2,048 pooled backbone
features. It provides context and does not enter C1-C4.

---

## 4. Model definitions

### 4.1 Frozen backbone

The backbone is `torchvision` ResNet50 with
`ResNet50_Weights.IMAGENET1K_V2`. Its parameters never receive gradients.
Backbone batch-normalization layers stay in evaluation mode during every
training phase. Their running statistics cannot change.

The primary forward path is:

```text
image
-> frozen stem and layers 1-3
-> standard adapter or DKA
-> frozen layer 4
-> global average pooling
-> common neck stem
-> residual or mHC neck
-> five-output classifier
```

### 4.2 Standard adapter and DKA

Both adapters act once, after the complete layer-3 output and before layer 4.
Their input and output shape is `batch x 1024 x 14 x 14`.

The standard adapter is:

```text
x + Conv1x1(37 -> 1024)(GELU(Conv1x1(1024 -> 37)(x)))
```

The DKA is:

```text
u = Conv1x1(1024 -> 16)(x)
v = DepthwiseConv51(u) + DepthwiseConv5(u)
output = x + Conv1x1(16 -> 1024)(GELU(v))
```

The depthwise convolutions use strides of 1 and same padding. The large and
small padding values are 25 and 2. All projections and convolutions include
bias terms.

The DKA contains 75,856 trainable parameters. The standard adapter contains
76,837. Their difference is 1.29% of the DKA count.

Both output projections start with zero weights and zero biases. Each adapter
is therefore an identity operation at initialization.

The primary DKA values are fixed:

- bottleneck width: 16
- large kernel: 51
- small kernel: 5
- insertion point: after layer 3

An exploratory kernel study uses large kernels 7, 15, 31, and 51. It uses
the DKA plus same-width residual cell, fold 1, seed 42, and sample sizes 50
and 1,044. It uses inner-validation results only. Its result cannot change
the primary DKA values.

### 4.3 Common neck stem

Every primary and capacity-control cell uses this trainable stem:

```text
LayerNorm(2048)
-> Linear(2048 -> 256)
-> GELU
```

The stem receives the globally pooled layer-4 feature. It returns one
256-dimensional vector. The stem stays frozen during head-only warmup and
trains during fine-tuning.

### 4.4 Same-width residual neck

The primary residual neck contains four blocks. Each block uses pre-norm:

```text
z_next = z + Linear(64 -> 256)(GELU(Linear(256 -> 64)(LayerNorm(z))))
```

The second linear layer in every block starts with zero weights and zero
biases. The full neck is an identity operation at initialization.

### 4.5 mHC neck

The retinal mHC neck adapts the source method to pooled ResNet features. The
source method was developed for language models, so this is a new application.

The neck repeats the 256-dimensional stem output into four residual streams.
Its initial shape is `batch x 4 x 256`. It applies four mHC blocks and averages
the four output streams before classification.

For one image, `X_l` has shape `4 x 256`. `H_pre,l` and `H_post,l` each have
shape `1 x 4`. `H_res,l` has shape `4 x 4`. Both `u_l` and `r_l` have shape
`1 x 256`. The transposed post map expands `r_l` back to four streams.

For block `l`, the update is:

```text
u_l = H_pre,l * X_l
r_l = MLP64(LayerNorm(u_l))
X_(l+1) = H_res,l * X_l + transpose(H_post,l) * r_l
```

The `MLP64` transformation is identical to the width-64 residual-block
transformation. Its second linear layer starts at zero.

The router flattens the four streams to 1,024 features. It applies non-affine
RMS normalization with epsilon `1e-6`. A dynamic projection without bias
produces 24 values:
four for `H_pre`, four for `H_post`, and sixteen for `H_res`. Each block also
contains 24 static biases and three scalar gates.

For mapping type `q`, the logits are:

```text
logits_q = alpha_q * projection_q(RMSNorm(vectorized_streams)) + bias_q
```

The constrained mappings are:

```text
H_pre  = sigmoid(pre_logits)
H_post = 2 * sigmoid(post_logits)
H_res  = Sinkhorn-Knopp(residual_logits, iterations=20)
```

Sinkhorn calculations use float32. Before normalization, the implementation
subtracts the largest logit for numerical stability and exponentiates the
remaining logits. One Sinkhorn iteration normalizes all rows and then all
columns. The procedure performs exactly 20 iterations.

The projection weights use Xavier-uniform initialization. All three scalar
gates start at 0.01. The `H_pre` static bias is `logit(1/4)`. The `H_post`
static bias is zero. The static `H_res` target is:

```text
0.99 * I + 0.01 * J / 4
```

Here, `I` is the 4 by 4 identity matrix and `J` is the 4 by 4 all-ones matrix.
The target is positive and doubly stochastic. The `H_res` static bias is the
elementwise natural logarithm of this target.

At initialization, all four streams are equal and each transformation returns
zero. A doubly stochastic residual map preserves the equal streams. The mHC
neck therefore returns the same initial output as the residual neck.

The mHC count has this exact breakdown:

| Component | Per block | Four blocks |
|---|---:|---:|
| LayerNorm and width-64 MLP | 33,600 | 134,400 |
| Dynamic router projection | 24,576 | 98,304 |
| Static router biases | 24 | 96 |
| Router gates | 3 | 12 |
| **Total** | **58,203** | **232,812** |

### 4.6 Capacity-matched residual neck

The capacity control uses the same four-block residual structure. Its only
change is an MLP hidden width of 112. Its output projections also start at
zero.

### 4.7 Output head and default initialization

Primary and capacity-control cells use one biased `Linear(256 -> 5)` output
head. The linear probe uses one biased `Linear(2048 -> 5)` output head. The
loss consumes logits directly. Prediction export applies a sigmoid.

Unless this protocol gives a different rule, convolution and linear modules
use their PyTorch default `reset_parameters` initialization. LayerNorm weights
start at one, and LayerNorm biases start at zero. A matched set of common-stem
and output-head parameters is created once and copied into every matched cell.

### 4.8 Required model tests

The implementation must pass these tests before any study run:

1. Adapter input and output shapes are equal.
2. Both adapters return their inputs at initialization within `1e-6`.
3. Both primary necks return equal outputs at initialization within `1e-6`.
4. Every Sinkhorn row and column sum differs from 1 by less than `1e-5`.
5. No backbone parameter receives a gradient.
6. No backbone batch-normalization statistic changes.
7. Trainable parameter counts equal the registered counts.
8. The adapter mismatch remains less than 2%.
9. The mHC and capacity-control mismatch remains less than 0.1%.

---

## 5. Training protocol

### 5.1 Reproducibility

Each run records the fold, seed, subset hash, cohort hash, code revision,
package versions, GPU type, and wall-clock time.

The implementation sets the Python, NumPy, PyTorch, CUDA, sampler, and data
loader seeds. It uses deterministic CUDA operations when PyTorch provides
them. It records any operation that lacks a deterministic implementation.

Within a matched fold, seed, and sample size, every cell uses the same image
order and augmentation draws. Common stems and classifier heads use identical
initial parameter values across the matched cells.

Training uses a batch size of 32 and automatic mixed precision. Every cell
uses identical precision and gradient-scaling rules. The optimizer clips the
global gradient norm to 1.0 in every cell.

### 5.2 Loss and class weights

The loss is `BCEWithLogitsLoss`. For each run and target class:

```text
pos_weight = number of negative fit images / number of positive fit images
```

Weights use the fit subset only. Validation and test labels cannot enter the
weight calculation. Subset construction guarantees a nonzero numerator and
denominator.

### 5.3 Head-only warmup

Only the classifier head trains during warmup. The adapter, common stem,
neck, and backbone remain frozen.

Warmup uses AdamW with learning rate `1e-3`. It stops after three epochs
without an inner-validation macro AUC improvement. It has a maximum of 15
epochs. A tie selects the earlier epoch.

### 5.4 Fine-tuning

Fine-tuning starts from the selected warmup checkpoint and creates a new
AdamW optimizer. It runs for 100 epochs with cosine annealing and no restarts.

| Parameter group | Learning rate | Cells |
|---|---:|---|
| ResNet50 backbone | frozen | all |
| Standard adapter or DKA | `1e-3` | primary and capacity cells |
| Common neck stem | `1e-3` | primary and capacity cells |
| Residual or mHC neck | `1e-3` | primary and capacity cells |
| Classifier head | `1e-4` | all |

AdamW uses betas `(0.9, 0.999)`, epsilon `1e-8`, and weight decay `1e-4` for
linear and convolution weights. Biases, normalization parameters, and router
gates use zero weight decay. Cosine annealing ends at learning rate zero.

The selected checkpoint has the highest inner-validation macro AUC. A tie
selects the earlier epoch.

The linear probe uses the same head warmup and 100-epoch head schedule. It has
no adapter, common stem, or neck parameter groups.

### 5.5 Threshold selection

Thresholds do not affect the primary AUC comparison. Each class threshold is
selected on the fixed inner-validation set after checkpoint selection.

Candidate thresholds are the unique inner-validation probabilities and 0.5.
The selected threshold maximizes class F1. A tie selects the threshold closest
to 0.5. A second tie selects the higher threshold. The selected thresholds
are applied once to the outer test fold.

---

## 6. Sample-size sweep

For each fold `f`, sample size `n`, and seed `s`, use the cells from Section
3.1. Let `A`, `B`, `C`, and `D` denote their outer-test macro AUC values.

The seed-level DKA effect is:

```text
e(f,n,s) = 0.5 * [(B - A) + (D - C)]
```

The analysis follows this fixed order:

1. Average seed effects within each fold and sample size.
2. Average the five fold effects equally at each sample size.
3. Regress the five effects on `log10(n)` with unweighted ordinary least squares.

C4 is the slope. It reports the macro AUC change per tenfold increase in
training images. The regression does not use data-derived weights.

### 6.1 Phase 3 stability gate

Phase 3 first runs all three 50-image seeds for every primary cell and fold.
The gate uses inner-validation DKA effects.

Within each fold, calculate the sample standard deviation of its three
seed-level DKA effects with one degree removed. Pool the five variances with
10 degrees of freedom.

For fold standard deviations `s_f`, the exact gate statistic is:

```text
SD(50) = sqrt(sum_f(2 * s_f^2) / 10)
```

- If the pooled standard deviation is less than 0.030, continue the sweep.
- If it is 0.030 or more, stop the sweep and report training instability.

The gate cannot change the sample sizes, seeds, contrasts, or regression. If
the gate stops the sweep, C4 is reported as not estimated.

At 100 images, calculate `SD(100)` by replacing 2 and 10 in the equation with
1 and 5. Report `SD(100) / SD(50)` from the inner-validation effects. This
diagnostic cannot change the analysis.

---

## 7. Outcomes and inference

### 7.1 Outcomes

The primary outcome is fold-averaged macro AUC-ROC. It averages the five
per-fold macro AUC values equally.

The report also provides pooled out-of-fold macro AUC. This value is secondary
because scores from separately trained folds can use different scales.

Secondary outcomes are:

- per-class AUC
- macro and weighted F1 at selected thresholds
- Hamming loss
- exact match
- warmup epochs and selected fine-tuning epoch
- trainable parameters, runtime, and peak GPU memory

### 7.2 Paired bootstrap

The primary inference uses a paired bootstrap within outer folds. Each
replicate resamples `analysis_group_id` values with replacement inside every
fold. It includes every image from each sampled group.

When no source subject identifier exists, each image forms its own analysis
group. In that case, the procedure becomes an image bootstrap.

The same sampled indices apply to all cells, sample sizes, and seeds. The
analysis recalculates fold metrics, C1-C3, the five DKA effects, and C4.

The final analysis uses:

```text
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 2026
```

Development can use 500 replicates. Pipeline testing can use 2,000. Only the
10,000-replicate analysis produces reported inference.

### 7.3 Degenerate bootstrap classes

Fixed folds and subsets contain both label values for every target. A
bootstrap replicate can still omit every positive or every negative example
for one class.

AUC is undefined in either case because it needs both label values.

If this occurs, remove that class from every compared cell in that fold and
replicate. Continue the replicate and record the number of included classes.
This rule preserves pairing between cells.

If no class remains in a fold, discard the replicate and draw its replacement.
Continue until the analysis contains 10,000 valid replicates.

### 7.4 Confidence intervals and p-values

Bonferroni correction applies across C1-C4. Each contrast uses
`alpha = 0.05 / 4 = 0.0125`.

The report gives a 98.75% two-sided percentile interval from bootstrap
quantiles 0.00625 and 0.99375.

For an observed contrast `T_hat` and bootstrap contrast `T_b`, define:

```text
D_b = T_b - T_hat

p_raw = (1 + sum(abs(D_b) >= abs(T_hat))) / (10000 + 1)
p_adjusted = min(1, 4 * p_raw)
```

This is the registered null-centered, two-sided bootstrap p-value. The added
one is the finite-replicate correction.

If the Phase 3 gate stops C4, C1-C3 retain the four-test correction. The
analysis does not replace it with a three-test correction.

### 7.5 Scope of inference

The bootstrap is conditional on the trained models, registered seeds, and
fixed fold assignment. It captures held-out analysis-group sampling
uncertainty. It does not capture uncertainty from a new training seed, a new
fold assignment, or a new clinical site.

For each contrast, the report also gives all five fold effects, their mean,
standard deviation, and range. These summaries are descriptive because the
outer-training folds overlap.

---

## 8. Execution plan

### Phase 0 — Cohort and frozen data contract

1. Obtain the official cleaned MuReD label table and cleaning list.
2. Build `cohort.csv` with the registered five-label inclusion rule.
3. Resolve source, subject, label, and duplicate audits.
4. Implement and test the preprocessing module.
5. Create and freeze outer folds and inner-validation sets.
6. Create and freeze all nested training subsets.
7. Save prevalence, source, subject, and class-support tables.

**Acceptance condition:** The cohort contains exactly 1,489 images. Every
fixed set passes the registered label-support rules.

**Deliverables:** `cohort.csv`, preprocessing code, fold CSVs, inner-validation
CSVs, subset CSVs, hashes, and a provenance report.

### Phase 1 — Modules and infrastructure

1. Implement the standard adapter and DKA.
2. Implement the common stem and three necks.
3. Pass every model test in Section 4.8.
4. Implement training, prediction export, metrics, and bootstrap analysis.
5. Run the registered exploratory kernel study.
6. Measure runtime and update the provisional time estimate.

The exploratory kernel scores cannot change the primary model.

**Deliverables:** Tested modules, run configuration, prediction schema,
analysis code, kernel table, and measured runtime.

### Phase 2 — Full-data study

1. Run the four primary cells across five folds at `n=1,044`, seed 42.
2. Run both capacity-control cells across five folds at the same `n` and seed.
3. Run the contextual linear probe across five folds.
4. Export probabilities, labels, thresholds, checkpoints, and run metadata.
5. Calculate C1-C3 and the secondary capacity comparisons.

**Deliverables:** Primary factorial table, capacity-control table, contextual
reference, C1-C3, and adjusted inference.

### Phase 3 — Registered sweep

1. Run all primary cells at 50 images with seeds 42, 43, and 44.
2. Apply the registered stability gate.
3. If the gate passes, run the remaining registered sizes and seeds.
4. Calculate C4 and the `SD(100) / SD(50)` diagnostic.

**Deliverable:** DKA effect versus `log10(n)` with its estimate, adjusted
interval, p-value, fold effects, and seed diagnostics.

### Phase 4 — Reporting

1. Report the cohort derivation and every fixed data rule.
2. Report estimates and intervals before significance labels.
3. Separate confirmatory and exploratory results.
4. Report all protocol deviations and failed runs.
5. State the conditional scope of the bootstrap with every primary result.

---

## 9. Provisional compute budget

The time model is six fixed minutes plus 32 minutes scaled by `n / 1,044`.
It came from earlier T4 runs and remains provisional until Phase 1 measures the
new frozen-backbone modules.

| Phase | Runs | GPU-hours |
|---|---:|---:|
| Phase 1 — kernel study at 50 and 1,044 images | 8 | 3.0 |
| Phase 2 — primary factorial, four cells by five folds | 20 | 12.7 |
| Phase 2 — capacity controls, two cells by five folds | 10 | 6.3 |
| Phase 2 — contextual linear probe by five folds | 5 | 3.2 |
| Phase 3 — four cells, four new sizes, registered seeds | 140 | 25.3 |
| **Provisional total** | **183** | **50.5** |

The 10,000 bootstrap replicates load saved prediction arrays and calculate
metrics on CPU. They do not retrain models, use GPU training, or count as
model runs.

If the Phase 3 gate fails, the study does not spend the remaining sweep
budget.

---

## 10. Reporting and deviation rules

- Report every confirmatory estimate with its 98.75% interval.
- Do not describe a nonsignificant result as equivalence.
- Do not promote a capacity, kernel, per-size, or per-class result to C1-C4.
- Report every failed run and its predefined rerun decision.
- A technical failure can rerun with the same configuration and seed.
- A completed run cannot rerun because its score is unfavorable.
- Label every change after registered training begins as a protocol deviation.
- Preserve the original configuration, logs, and result beside any rerun.

---

## 11. Main risks

| Risk | Registered response |
|---|---|
| Cohort rule does not yield 1,489 images | Stop Phase 0 and resolve the source version before splitting |
| Subject identifiers are unavailable | Split by image and report the leakage limitation |
| Source imbalance remains across folds | Report source counts and include source indicators in stratification |
| mHC gain reflects added capacity | Use the registered width-112 capacity control |
| Small-sample training is unstable | Apply the fixed 50-image gate |
| Bootstrap omits one class value | Apply the paired class-removal rule |
| The DKA port underperforms | Report the estimate without changing its position or kernels |
| Pooled and fold-averaged AUC disagree | Use fold-averaged AUC as primary |
| Runtime exceeds the provisional budget | Report the measured rate before Phase 2 and revise only the time estimate |

---

## Appendix A — Planning sensitivity

These values guide resource planning only. They do not become observed bounds
or substitute for confidence intervals.

The calculation assumes 80% power, two-sided `alpha=0.05/4`, an effect-level
seed standard deviation of 0.015 at 50 images, and `1/sqrt(n)` decay.

| Planning quantity | Approximate detectable effect |
|---|---:|
| C1 or C2 at 1,044 images | 0.015 macro AUC |
| C3 at 1,044 images | 0.029 macro AUC |
| C4 slope | 0.021 macro AUC per decade |
| C4 change across the ladder | 0.027 macro AUC |

The new architecture and cohort construction can change the actual variance.
Observed results are interpreted only through their estimates and registered
intervals.

## Appendix B — Registered constants

| Constant | Value |
|---|---|
| Target labels | `DR, NORMAL, ODC, MH, DN` |
| Outer folds | `5` |
| Fit images per fold | `1044` |
| Training sizes | `50, 100, 251, 501, 1044` |
| Training seeds | `42, 43, 44` |
| Backbone weights | `ResNet50_Weights.IMAGENET1K_V2` |
| Input size | `224 x 224` |
| DKA bottleneck | `16` |
| DKA kernels | `51, 5` |
| Standard-adapter width | `37` |
| Neck feature width | `256` |
| Neck blocks | `4` |
| Primary MLP width | `64` |
| Capacity-control width | `112` |
| mHC streams | `4` |
| Sinkhorn iterations | `20` |
| Fine-tuning epochs | `100` |
| Warmup maximum | `15` |
| Warmup patience | `3` |
| Batch size | `32` |
| Weight decay | `1e-4` |
| Gradient clip | `1.0` |
| Bootstrap replicates | `10000` |
| Bootstrap seed | `2026` |
| Split seed | `2026` |
| Confirmatory contrasts | `C1, C2, C3, C4` |

---

**V1 freeze rule:** Freeze this file after Phase 0 data acceptance and before
the first registered model run. Later changes require a dated deviation note.

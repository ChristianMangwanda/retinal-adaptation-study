# Retinal adaptation study

A pre-registered 2x2 factorial study of two adaptation designs on a frozen
ResNet-50, run on a 1,489-image MuReD-derived retinal cohort with five
disease labels: the Dual-Kernel Adapter (DKA) and Manifold-Constrained
Hyper-Connections (mHC), each against capacity-matched controls.

The whole experiment lives in six notebooks. Each phase notebook reads the
shared definitions and the outputs of the phases before it.

| File | Role |
|---|---|
| `retinal_adaptation_study_v1.md` | Frozen protocol: design, contrasts, constants, decision rules |
| `definitions.ipynb` | All shared code: data, models, training, metrics, bootstrap |
| `phase0.ipynb` | Cohort build, duplicate audit, splits, subsets, image cache |
| `phase1.ipynb` | Model tests, runtime measurement, kernel study |
| `phase2.ipynb` | Full-data factorial and capacity controls, C1-C3 |
| `phase3.ipynb` | Sample-size sweep, stability gate, C4 |
| `phase4.ipynb` | Final report and artifact hashes |
| `phases.md`, `RUNBOOK.md`, `TRADEOFFS.md` | Status, operating steps, known limits |

Images and run outputs are not in this repository. The dataset is public:
MuReD, doi:10.17632/pc4mb3h8hz.1. Running `phase0.ipynb` against the
official label tables and `images.zip` rebuilds the cohort, splits, and
cache; later phases rebuild everything else. Training is deterministic and
all analysis seeds are fixed in the protocol.

# V-JEPA vs VideoMAE on UCF101

This experiment compares frozen V-JEPA ViT-L/16 and VideoMAE Base
representations under the same downstream protocol on five UCF101 classes.
V-JEPA predicts latent video representations, while VideoMAE learns through
masked pixel reconstruction. A standardized linear probe measures how linearly
separable their frozen embeddings are.

## Protocol

- Classes: `ApplyEyeMakeup`, `Basketball`, `Biking`, `Diving`,
  `WalkingWithDog`.
- Official UCF101 recognition splits 1, 2, and 3.
- Per split and class: 60 official-train videos and 20 official-test videos.
- Per split: 300 train and 100 test samples.
- Deterministic selection with seed 42. Videos never move between official
  train and test partitions.
- Exactly 16 uniformly sampled RGB frames at 224 x 224.
- Batch size 1 and MPS when available, with a CPU fallback.
- Frozen backbones and mean pooling over all patch tokens for both models.
- `StandardScaler` followed by `LogisticRegression` with identical settings.
- One untimed warm-up inference before timing.
- Per-video feature caches allow interrupted runs and reuse overlapping videos
  across official splits.

The primary comparison uses mean pooling because neither backbone exposes a
true CLS token. Selecting VideoMAE's first patch token is not used as the main
benchmark.

## Setup

From the repository root:

```bash
chmod +x experiments/vjepa_vs_videomae/scripts/*.sh
experiments/vjepa_vs_videomae/scripts/setup_environment.sh
experiments/vjepa_vs_videomae/scripts/download_assets.sh
```

`download_assets.sh` downloads the official split lists, extracts only the five
required UCF101 classes, downloads the official V-JEPA ViT-L/16 checkpoint, and
deletes source archives after extraction. Set `KEEP_ARCHIVES=1` to retain them.

## Run

Execute the complete three-split benchmark:

```bash
experiments/vjepa_vs_videomae/scripts/run_full_benchmark.sh
```

Useful overrides:

```bash
DEVICE=cpu SPLIT_IDS="1" TRAIN_PER_CLASS=2 TEST_PER_CLASS=1 \
  experiments/vjepa_vs_videomae/scripts/run_full_benchmark.sh
```

The second command is a smoke test, not a reportable experiment. Re-running the
full command resumes from cached frames and embeddings.

## Outputs

Generated data and large artifacts remain ignored under `outputs/`:

```text
outputs/
  data/split_01..03/
  data/frames/
  cache/features/
  features/split_01..03/{vjepa,videomae}/
  results/split_01..03/{vjepa,videomae}/
  results/aggregate/
```

Each model/split result contains:

- scalar metrics, including accuracy, balanced accuracy, macro/weighted F1,
  macro precision/recall, and top-3 accuracy;
- classification reports and per-class metrics;
- raw and labeled confusion matrices;
- per-video predictions and confidences;
- detailed timing for loading, preprocessing, model inference, and total
  extraction;
- feature statistics, fitted linear probe, effective run configuration, and
  extraction manifest.

The versionable report is written to `reports/latest/` and includes:

- aggregate mean +/- standard deviation across the three official splits;
- metric and per-class charts;
- normalized confusion matrices;
- accuracy-versus-speed comparison;
- PCA projections of test embeddings;
- dataset distribution and temporal frame storyboards;
- a shared pipeline diagram;
- true-class confidence after revealing 25%, 50%, 75%, and 100% of selected
  videos.

Open `reports/latest/report.html` for presentation or use the PNG/PDF figures
individually.

Additional curated figures from extended runs and audits are kept alongside
`latest/` for slide decks and write-ups:

- `reports/server_10class_latest/` -- the same report figures regenerated for
  a 10-class extension of the benchmark.
- `reports/server_10class_external_test/` -- a storyboard from an external
  (out-of-distribution) video used to sanity-check the 10-class models.
- `reports/server_10class_protocol_audit/` -- a random-label control showing
  expected chance-level performance, used to validate the evaluation
  protocol.
- `reports/overfitting_audit/` -- train-vs-test accuracy and a regularization
  sweep used to check the linear probe is not overfitting.
- `reports/send_curves/` -- a few-shot accuracy curve.
- `reports/schemas/` -- system and V-JEPA architecture diagrams.

## Reproducibility

The pipeline records only the information needed to reproduce a run:

- official split, random seed, selected video IDs, and effective arguments;
- Git commit and core package versions;
- VideoMAE model revision;
- V-JEPA config and checkpoint SHA-256;
- device, frame count, image size, pooling, and timing policy.

## Interpretation and limitations

Accuracy gives the global proportion of correct predictions. Balanced accuracy
and macro-F1 weight each class equally. Per-class F1 exposes classes where one
representation is weaker. Timing should be interpreted on the same machine and
device only.

This remains a five-class subset, not full UCF101. The backbones are frozen, no
full fine-tuning is performed, and the selected classes affect difficulty.
MPS behavior can vary by PyTorch version; CPU fallback is slower but more
predictable. Progressive-confidence figures are qualitative case studies and
are not substitutes for the three-split aggregate metrics.

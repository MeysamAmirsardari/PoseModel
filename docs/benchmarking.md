# Benchmarking PoseModel

The benchmark runner compares every representation using the same immutable windows, labels,
recording groups, random seed, downstream evaluator, and corruption suite.

## Prepare a corpus

Convert each recording independently with `posemodel prepare`. Create one label CSV per recording:

```csv
frame,label
0,rest
1,rest
2,groom
```

Copy `configs/benchmark.example.yaml` and replace its paths and recording metadata. Every animal,
session, or recording group selected by `split_unit` must occur in exactly one split. The loader
rejects a manifest that leaks a group across splits.

## Run

```bash
posemodel benchmark configs/my-benchmark.yaml --output benchmarks/my-run
```

To generate the exact ordered window table required by an external method:

```bash
posemodel benchmark-index configs/my-benchmark.yaml benchmarks/window-index.csv
```

Outputs include:

- `benchmark.json`: complete machine-readable metrics and environment details;
- `benchmark.html`: a self-contained comparison report;
- `embeddings-<candidate>.npz`: aligned representations and test window identities.

## Built-in candidates

- `kinematics`: standardized pose and velocity summary statistics;
- `pca`: PCA over standardized pose windows;
- `tcn`: a non-graph temporal convolutional autoencoder;
- `posemodel`: a PoseModel trained only on manifest training recordings, or an existing checkpoint
  when `settings.checkpoint` is supplied;
- `imported`: aligned embeddings from CEBRA, Keypoint-MoSeq, or another external method.

An imported NPZ contains arrays named `train`, `validation`, and `test`. It may also contain
`<split>_recording_ids` and `<split>_starts`; when supplied, the runner aligns embeddings by these
keys and rejects missing windows. Otherwise arrays must already follow the exported index order.

When a `posemodel` candidate has no checkpoint, its model, loss, optimizer, epoch, masking, and
device settings are read from the candidate's `settings` mapping. The best validation state is
saved beside the report as `model-<candidate>.pt`.

## Metrics

When labels are present, the report includes frozen linear-probe accuracy, balanced accuracy,
macro F1, average precision, k-nearest-neighbor accuracy, and a label-efficiency curve. Unsupervised
metrics include GMM BIC, label AMI/ARI, cluster stability over five seeds, and normalized mutual
information with animal and session identity. Robustness results report latent cosine consistency
and downstream macro F1 under missing keypoints, joint-tube dropout, and coordinate jitter.

Normalization and representation fitting always use the training split. Labels are used only by
the evaluator, never by the representation candidate.

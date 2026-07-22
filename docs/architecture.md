# Architecture and validation strategy

## Selected foundation

PoseModel uses a factorized spatiotemporal masked autoencoder:

1. Coordinates, velocity, confidence, and observed status are embedded per joint token.
2. Graph-masked self-attention operates within each frame and individual.
3. Temporal self-attention operates along each joint trajectory.
4. For multi-animal recordings, interaction attention operates across individuals.
5. A student observes masked/corrupted tokens and predicts contextual token targets produced by
   an exponential-moving-average teacher that receives the clean sequence.
6. Attention pooling produces a window representation. Optional graph-level `mu` and `logvar`
   heads regularize this representation as a variational latent.
7. A coordinate decoder reconstructs the complete pose using time, joint, and individual queries.

This combines the useful inductive bias of a skeleton graph with the representation quality of
masked contextual prediction. Raw reconstruction remains useful for quality control but is not
the only learning signal.

## Required baselines

Every substantive architecture change should be compared with:

- PCA over normalized windows;
- a temporal convolutional autoencoder without graph structure;
- the graph-temporal model without the teacher objective;
- the full model;
- Keypoint-MoSeq when behavioral segmentation is the target and its assumptions apply.

## Split policy

Random window splits are invalid when windows overlap. Use held-out recordings, animals, or
sessions. If only one recording exists, split it into contiguous blocks with a gap at least as
large as the model window, then create windows independently within each block.

## Evaluation

Report normalized MPJPE, masked reconstruction error, linear-probe performance, cluster stability,
cross-seed latent alignment, and animal/session predictability. For annotated data, also report
macro F1, adjusted mutual information, and bout-level agreement. Evaluate tracking corruptions
separately: missing joints, identity swaps, jitter, and contiguous occlusions.

## Research basis

- MotionBERT showed the value of dual-stream spatial and temporal transformers trained to recover
  motion from partial noisy observations.
- SkeletonMAE established graph-based masked reconstruction for skeleton sequences.
- Skeleton2vec improved masked skeleton learning with contextual targets from a teacher encoder.
- CEBRA demonstrated the value of contrastive, temporally structured embeddings and consistency
  measurements across subjects and sessions.
- Keypoint-MoSeq showed why animal behavior state discovery needs an explicit dynamical model and
  keypoint-specific noise treatment rather than framewise clustering alone.

These works motivate the starting point; they do not remove the need for dataset-specific
ablation and leakage-safe evaluation.

Primary sources:

- [MotionBERT (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_MotionBERT_A_Unified_Perspective_on_Learning_Human_Motion_Representations_ICCV_2023_paper.html)
- [SkeletonMAE (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Yan_SkeletonMAE_Graph-Based_Masked_Autoencoder_for_Skeleton_Sequence_Pre-Training_ICCV_2023_paper.html)
- [Skeleton2vec](https://arxiv.org/abs/2401.00921)
- [CEBRA (Nature 2023)](https://www.nature.com/articles/s41586-023-06031-6)
- [Keypoint-MoSeq (Nature Methods 2024)](https://www.nature.com/articles/s41592-024-02318-2)

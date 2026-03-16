# VoxPocket Research Plan

## Thesis-Friendly Direction

Primary direction:
Hybrid multi-task pocket recovery with voxel segmentation and explicit pocket localization/ranking.

Motivation:
- VoxPocket is strongest when treated as a dense 3D segmentation model.
- P2Rank is stronger for ranked pocket localization.
- A good research contribution is not just "beat P2Rank everywhere", but to build a model that preserves dense voxel predictions while improving localization and pocket selection.

## Core Research Questions

1. Can multi-task learning improve pocket recovery relative to segmentation-only VoxPocket?
2. Can explicit localization-aware supervision improve centroid distance without sacrificing Dice/IoU?
3. Can candidate ranking on top of voxel predictions narrow the gap to P2Rank on pocket selection?
4. Can a hybrid voxel-plus-surface-aware model outperform either representation alone?

## Recommended Experimental Roadmap

### Phase 1: Strong Supervised Baseline

Goal:
Establish the best segmentation-only VoxPocket baseline under a strict train/val/test split.

Current best direction:
- `augment=false`
- `loss=bce_dice`
- `pos_weight=6.0`
- validation threshold sweep
- early stopping

Deliverables:
- Best held-out test Dice/IoU/precision/recall/F1
- Pocket detection rate
- Mean and median distance to reference pocket
- P2Rank comparison on the same test set

### Phase 2: Multi-Task VoxPocket

Goal:
Add an auxiliary head that predicts pocket-level properties in addition to the segmentation mask.

Recommended auxiliary tasks:
- pocket presence score
- centroid heatmap or center likelihood map
- optional connected-component ranking score

Suggested ablations:
- segmentation only
- segmentation + presence head
- segmentation + localization head
- segmentation + presence + localization

Primary hypothesis:
Auxiliary supervision will improve pocket detection and localization while maintaining overlap quality.

Current status:
- Segmentation + presence improved the best VoxPocket result.
- Segmentation + centroid heatmap did not improve performance and reduced detection.
- The immediate conclusion is that pocket-presence supervision is useful, but the current centroid target formulation is not aligned enough with the downstream recovery objective.

### Phase 3: Candidate Ranking

Goal:
Convert voxel predictions into candidate pockets, then learn to rank them.

Candidate pipeline:
- threshold voxel predictions
- extract connected components
- compute candidate features: size, confidence, centroid, local chemistry
- rank candidates using a small learned model

Evaluation additions:
- top-1 recovery
- top-k recovery
- distance threshold success rates

Primary hypothesis:
Ranking will improve sample-level detection and pocket selection relative to raw thresholding.

Updated priority:
This is now the highest-priority next direction because ranking addresses the main remaining gap to P2Rank more directly than the current centroid-head formulation.

Current implementation status:
- Heuristic connected-component candidate extraction has been added.
- Candidate features now include component size, confidence statistics, local neighborhood scores, shape descriptors, and sample-level presence confidence.
- Dedicated evaluation scripts now report top-1 and top-k candidate recovery from existing VoxPocket checkpoints.
- Multi-threshold candidate union with centroid-based deduplication has been implemented.
- Binary and pairwise learned ranking baselines were tested, but neither clearly outperformed heuristic ranking across the main top-k metrics.
- Current conclusion: the candidate framework is useful, but ranking is not yet a headline improvement over the heuristic baseline and is best treated as exploratory/future work.

### Phase 4: Hybrid VoxPocket + P2Rank

Goal:
Use P2Rank as either a teacher, candidate generator, or auxiliary feature source.

Options:
- use P2Rank top-k pockets as candidate proposals for refinement
- use P2Rank centers as an auxiliary target
- combine voxel predictions with surface-based scores

Primary hypothesis:
Hybrid modeling will preserve VoxPocket's segmentation advantage while closing the localization gap.

## Evaluation Plan

Use the same strict split for all experiments:
- train: 739
- val: 158
- test: 160

Track:
- Dice / IoU
- precision / recall / F1
- pocket detection rate
- mean / median distance to reference pocket
- top-1 and top-3 success rates where applicable

Best-practice rule:
- tune thresholds and hyperparameters on validation only
- report final numbers on test only once per chosen configuration

## Immediate Next Coding Steps

1. Keep the segmentation + presence model as the current best VoxPocket checkpoint.
2. Keep the candidate framework as a working top-k recovery extension.
3. Treat learned ranking as exploratory rather than core until it shows a clearer gain over heuristic ordering.
4. If candidate work continues, focus on better proposals or hybrid P2Rank-derived candidate features rather than more MLP tuning alone.
5. Consolidate the thesis around the multitask segmentation result plus the candidate-based evaluation/ranking framework as a secondary contribution.

## Expected Thesis Contribution

A strong thesis framing would be:
"VoxPocket: a multi-task voxel-based framework for protein pocket recovery that combines dense pocket segmentation with explicit pocket localization and selection."

This framing is strong because it:
- differentiates VoxPocket from surface-only ranking methods
- naturally motivates comparison to P2Rank
- supports multiple clean ablations
- allows both engineering and methodological contributions

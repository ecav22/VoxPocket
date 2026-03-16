# VoxPocket Progress Summary

## Main Result

The strongest VoxPocket model so far is the multitask segmentation model with an auxiliary pocket-presence head.

Held-out test set result (`n = 160` proteins, threshold `0.3`):
- Dice / F1: `0.4537`
- IoU: `0.2934`
- Precision: `0.5028`
- Recall: `0.4133`
- Pocket detection rate: `143 / 160 = 89.4%`
- Mean distance to reference pocket: `0.9449 nm`
- Median distance to reference pocket: `0.4529 nm`

This outperformed the previous segmentation-only VoxPocket baselines and is the current best model.

## Comparison to P2Rank

P2Rank remains stronger for ranked pocket localization on the same test set:
- Top-1 mean distance: `0.8067 nm`
- Top-1 median distance: `0.3120 nm`
- Top-3 best mean distance: `0.4132 nm`
- Top-3 best median distance: `0.2671 nm`
- Detection rate: `160 / 160 = 100%`

Interpretation:
- VoxPocket is currently strongest as a dense pocket segmentation model.
- P2Rank is currently stronger as a ranked pocket-localization method.

## Auxiliary Head Experiments

- Adding a pocket-presence head improved VoxPocket and produced the best overall result.
- Adding a centroid heatmap head reduced performance and did not improve the overall localization story enough to justify keeping it.

Conclusion:
- Presence supervision is helpful.
- The current centroid-supervision formulation is not.

## Candidate Ranking Experiments

A candidate extraction and ranking framework was implemented on top of VoxPocket predictions.

What now works:
- connected-component candidate extraction
- top-1 and top-k candidate evaluation
- multi-threshold candidate union with deduplication
- heuristic ranking baseline
- learned binary ranking baseline
- learned pairwise ranking baseline

Key findings:
- Best single-threshold candidate pool: `threshold = 0.15`, `min_size = 1`
- Multi-threshold union increased candidate diversity and slightly improved tight top-k localization
- Learned ranking did not clearly outperform heuristic ranking across the main top-k success metrics

Conclusion:
- Candidate extraction is a useful extension and evaluation framework
- learned ranking is not yet a major headline improvement

## Recommended Thesis Framing

Recommended main story:
"VoxPocket is a multi-task voxel-based framework for dense protein pocket recovery, with strong segmentation performance and an extensible candidate-based top-k evaluation pipeline."

Recommended emphasis:
- main contribution: multitask voxel segmentation with presence supervision
- secondary contribution: candidate extraction / ranking framework for top-k pocket recovery
- future work: improved candidate proposals, hybrid VoxPocket + P2Rank signals, and surface-aware ranking features

## Practical Next Step

The best immediate next step is to consolidate the thesis around the current best multitask VoxPocket model and present the candidate-ranking framework as a working extension rather than continuing to tune the current small rankers.

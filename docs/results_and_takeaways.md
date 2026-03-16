# Results And Takeaways

## Final Experimental Takeaway

The strongest VoxPocket result from this round is still the multitask voxel-segmentation model with an auxiliary pocket-presence head.

Best held-out test result (`n = 160`, threshold `0.3`):
- Dice / F1: `0.4537`
- IoU: `0.2934`
- Precision: `0.5028`
- Recall: `0.4133`
- Pocket detection rate: `89.4%`
- Mean distance to reference pocket: `0.9449 nm`
- Median distance to reference pocket: `0.4529 nm`

This remains the main positive result of the project so far.

## What Improved

1. Strict train/val/test evaluation stabilized the benchmark and made comparisons more trustworthy.
2. Removing augmentation and tuning `pos_weight` improved segmentation quality.
3. Adding the pocket-presence head produced the best overall VoxPocket model.
4. Candidate extraction and top-k evaluation now work end-to-end on top of VoxPocket outputs.

## What Did Not Help

1. The centroid heatmap head did not improve performance and reduced overall detection.
2. Learned candidate ranking, in both binary and pairwise forms, did not clearly outperform heuristic ordering.
3. Multi-threshold candidate generation increased candidate diversity, but only gave modest gains in tight top-k localization and did not fundamentally change the ranking story.

## Comparison To P2Rank

Current interpretation:
- VoxPocket is stronger as a dense segmentation model.
- P2Rank is stronger as a ranked localization method.

P2Rank still has the better localization metrics on the same test set, so the project’s strongest angle is not "VoxPocket beats P2Rank overall." The stronger claim is that VoxPocket provides dense pocket recovery and an extensible candidate-based framework, while P2Rank remains a powerful localization baseline.

## Best Thesis Framing

The clearest thesis framing after this experimental round is:

"VoxPocket is a multi-task voxel-based framework for dense protein pocket recovery, with strong segmentation performance and a working candidate-based top-k evaluation pipeline."

This framing lets the project emphasize:
- the best multitask segmentation result as the core contribution
- the candidate framework as a methodological extension
- ranking and hybrid fusion as future directions rather than forced headline claims

## Recommended Next Research Step

If the project continues beyond this round, the most promising next direction is not more tuning of the current small learned rankers. The better next step is hybrid modeling:
- incorporate P2Rank-derived candidate or score features
- use surface-aware descriptors
- or build better candidate proposals before revisiting learned ranking

That is more likely to produce a meaningful gain than more iterations on the current heuristic-vs-ranker setup alone.

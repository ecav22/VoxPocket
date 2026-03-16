# VoxPocket Comparison Table

Held-out test set size: 160 proteins

| Model | Threshold | Dice/F1 | IoU | Precision | Recall | Detection Rate | Mean Dist. (nm) | Median Dist. (nm) | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| VoxPocket baseline clean split | 0.5 | 0.3998 | 0.2498 | 0.3837 | 0.4173 | 0.8500 | 0.9951 | 0.4462 | First strict train/val/test evaluation |
| VoxPocket no-aug + early stopping | 0.3 | 0.4108 | 0.2585 | 0.4301 | 0.3931 | 0.8500 | 0.9463 | 0.4322 | Better overlap than baseline |
| VoxPocket no-aug + `pos_weight=6.0` | 0.3 | 0.4405 | 0.2825 | 0.4730 | 0.4122 | 0.8938 | 0.9730 | 0.4508 | Strongest segmentation-only VoxPocket |
| VoxPocket multitask (segmentation + presence) | 0.3 | 0.4537 | 0.2934 | 0.5028 | 0.4133 | 0.8938 | 0.9449 | 0.4529 | Best VoxPocket result so far |
| VoxPocket multitask (segmentation + presence + centroid) | 0.3 | 0.4174 | 0.2638 | 0.5254 | 0.3462 | 0.7750 | 1.0282 | 0.3328 | Centroid head improved neither overlap nor detection |
| VoxPocket heuristic candidates | `t=0.15`, `min=1` | N/A | N/A | N/A | N/A | 0.8250 candidate presence | 1.1304 | 0.4379 | Best single-threshold candidate pool; top-3 success @ 1.0 nm = 0.6063 |
| VoxPocket multi-threshold candidates | `[0.10,0.15,0.20,0.25,0.30]`, `min=1` | N/A | N/A | N/A | N/A | 0.8250 candidate presence | 1.1237 | 0.4379 | More candidate diversity; top-3 success @ 0.5 nm improved to 0.5313 |
| VoxPocket pairwise ranker on multi-threshold pool | same as above | N/A | N/A | N/A | N/A | N/A | 1.1723 | 0.4451 | Did not outperform multi-threshold heuristic ordering on the main top-k metrics |
| P2Rank top-1 | N/A | N/A | N/A | N/A | N/A | 1.0000 | 0.8067 | 0.3120 | Surface-based ranked pocket localization |
| P2Rank top-3 best pocket | N/A | N/A | N/A | N/A | N/A | 1.0000 | 0.4132 | 0.2671 | Best of top-3 pocket centers |

## Interpretation

- VoxPocket currently performs best as a dense pocket segmentation model under the multitask presence-head configuration.
- P2Rank remains stronger for pure sample-level pocket localization.
- Adding a centroid heatmap head did not improve performance and reduced detection substantially.
- Candidate extraction and top-k evaluation are now working end-to-end on top of VoxPocket segmentations.
- Multi-threshold candidate union modestly improved tight top-k localization, but learned ranking did not outperform heuristic ordering and is currently best treated as exploratory.
- The current strongest thesis result is still the multitask presence-head segmentation model; the candidate framework is best presented as a secondary methodological extension and future-work foundation.

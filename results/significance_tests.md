# Significance Tests

Outputs of the significance tests behind the markers in Tables 2 and 3.

**Procedure**
- Unit of analysis: test users. For each model, the per-user metric (Recall@K / NDCG@K) is computed on the test set.
- Seeds: each model is run with five seeds {999, 42, 2023, 2024, 2025}; the per-user metric is averaged over the five seeds before testing.
- Test: paired, two-sided t-test (`scipy.stats.ttest_rel`) between DGMRec and the strongest baseline, i.e., the baseline with the highest mean for that dataset and metric.
- Markers: `**` p < 0.01, `*` p < 0.05, `n.s.` otherwise.

## Table 2: Missing Modality Setting

| Dataset | Metric | Strongest baseline | t | p | Marker |
|---|---|---|---|---|---|
| Baby | Recall@20 | GUME | 5.818 | 6.04e-09 | ** |
| Baby | Recall@50 | GUME | 6.903 | 5.27e-12 | ** |
| Baby | NDCG@20 | GUME | 5.021 | 5.19e-07 | ** |
| Baby | NDCG@50 | GUME | 6.334 | 2.44e-10 | ** |
| Sports | Recall@20 | GUME | 5.885 | 4.00e-09 | ** |
| Sports | Recall@50 | GUME | 5.196 | 2.04e-07 | ** |
| Sports | NDCG@20 | GUME | 6.983 | 2.94e-12 | ** |
| Sports | NDCG@50 | GUME | 7.370 | 1.74e-13 | ** |
| Clothing | Recall@20 | DAMRS | 7.353 | 1.97e-13 | ** |
| Clothing | Recall@50 | DAMRS | 8.202 | 2.44e-16 | ** |
| Clothing | NDCG@20 | DAMRS | 7.949 | 1.92e-15 | ** |
| Clothing | NDCG@50 | DAMRS | 9.023 | 1.92e-19 | ** |
| TikTok | Recall@20 | DAMRS | 2.185 | 2.89e-02 | * |
| TikTok | Recall@50 | GUME | 2.054 | 4.00e-02 | * |
| TikTok | NDCG@20 | DAMRS | 2.039 | 4.15e-02 | * |
| TikTok | NDCG@50 | GUME | 2.011 | 4.43e-02 | * |

## Table 3: Missing Modality + New Items Setting

| Dataset | Metric | Strongest baseline | t | p | Marker |
|---|---|---|---|---|---|
| Baby | Recall@20 | MGCN | 7.552 | 4.47e-14 | ** |
| Baby | Recall@50 | MGCN | 7.393 | 1.49e-13 | ** |
| Baby | NDCG@20 | MGCN | 6.759 | 1.43e-11 | ** |
| Baby | NDCG@50 | DAMRS | 6.894 | 5.58e-12 | ** |
| Sports | Recall@20 | BM3 | 6.309 | 2.85e-10 | ** |
| Sports | Recall@50 | BM3 | 8.602 | 8.13e-18 | ** |
| Sports | NDCG@20 | BM3 | 7.003 | 2.54e-12 | ** |
| Sports | NDCG@50 | BM3 | 8.958 | 3.47e-19 | ** |
| Clothing | Recall@20 | DAMRS | 6.103 | 1.05e-09 | ** |
| Clothing | Recall@50 | MGCN | 7.401 | 1.37e-13 | ** |
| Clothing | NDCG@20 | GUME | 10.908 | 1.16e-27 | ** |
| Clothing | NDCG@50 | DAMRS | 7.206 | 5.86e-13 | ** |
| TikTok | Recall@20 | DAMRS | 5.509 | 3.81e-08 | ** |
| TikTok | Recall@50 | GUME | 5.636 | 1.85e-08 | ** |
| TikTok | NDCG@20 | LightGCN | 3.817 | 1.37e-04 | ** |
| TikTok | NDCG@50 | DAMRS | 7.969 | 2.03e-15 | ** |

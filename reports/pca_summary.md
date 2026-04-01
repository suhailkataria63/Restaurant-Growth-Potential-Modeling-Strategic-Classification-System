# PCA Explained Variance Summary

- Observations: 1696
- Original features: 38
- PCA components retained: 10

## Explained Variance by Component

| Component | Explained Variance | Cumulative Variance |
|-----------|-------------------|---------------------|
| PC1 | 0.2782 | 0.2782 |
| PC2 | 0.1933 | 0.4715 |
| PC3 | 0.1716 | 0.6431 |
| PC4 | 0.0939 | 0.7371 |
| PC5 | 0.0525 | 0.7896 |
| PC6 | 0.0477 | 0.8373 |
| PC7 | 0.0380 | 0.8753 |
| PC8 | 0.0330 | 0.9083 |
| PC9 | 0.0207 | 0.9290 |
| PC10 | 0.0128 | 0.9418 |

## Candidate Latent Factors (Top Loadings)

### PC1 - Suggested theme: Growth momentum

Positive direction (higher component score):
- `monthlyorders` (0.333)
- `selfdeliveryrevenue` (0.331)
- `selfdeliveryorders` (0.329)
- `instorerevenue` (0.304)
- `instoreorders` (0.300)

Negative direction (lower component score):
- `dd_share` (-0.183)
- `ue_share` (-0.161)
- `doordashnetprofit` (-0.064)
- `ubereatsnetprofit` (-0.063)
- `cuisinetype_Indian` (-0.033)

### PC2 - Suggested theme: Growth momentum

Positive direction (higher component score):
- `doordashnetprofit` (0.385)
- `ubereatsnetprofit` (0.385)
- `selfdeliverynetprofit` (0.306)
- `doordashrevenue` (0.194)
- `ubereatsrevenue` (0.179)

Negative direction (lower component score):
- `opexrate` (-0.357)
- `cogsrate` (-0.325)
- `instoreshare` (-0.314)
- `instoreorders` (-0.192)
- `instorerevenue` (-0.179)

### PC3 - Suggested theme: Channel leverage

Positive direction (higher component score):
- `sd_share` (0.382)
- `selfdeliverynetprofit` (0.197)
- `ubereatsnetprofit` (0.152)
- `doordashnetprofit` (0.152)
- `selfdeliveryorders` (0.127)

Negative direction (lower component score):
- `ue_share` (-0.363)
- `dd_share` (-0.338)
- `ubereatsorders` (-0.320)
- `ubereatsrevenue` (-0.308)
- `doordashorders` (-0.305)

### PC4 - Suggested theme: Cost pressure

Positive direction (higher component score):
- `deliverycostperorder` (0.619)
- `deliveryradiuskm` (0.619)
- `sd_deliverytotalcost` (0.421)
- `dd_share` (0.046)
- `doordashorders` (0.028)

Negative direction (lower component score):
- `selfdeliverynetprofit` (-0.138)
- `aov` (-0.085)
- `instorenetprofit` (-0.083)
- `instorerevenue` (-0.070)
- `instoreshare` (-0.070)

## Notes

- These themes are hypothesis labels to support cluster storytelling, not fixed causal truth.
- Use the scree plot elbow and cumulative variance to choose the PCA dimensionality for clustering models.
- Re-run this step whenever preprocessing features change so factor definitions remain stable.
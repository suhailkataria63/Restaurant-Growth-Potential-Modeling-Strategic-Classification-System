# Growth Potential Index (GPI) Methodology

## Objective

Create a single composite index (0-100) that ranks restaurants by structural growth potential while accounting for scale, profitability quality, and channel risk.

## Normalization

- All inputs are normalized to 0-1 before weighting.
- Most features use robust min-max scaling (5th to 95th percentile clipping).
- `aggregator_dependence` is inverted into a penalty score so high dependence lowers GPI.
- `expansion_headroom` uses a balanced scoring function (peaks at moderate-high headroom and penalizes extremes).

## Weighted Formula

`GPI_raw = Σ(weight_i × normalized_component_i)`

`GPI_score = GPI_raw × 100`

| Component | Weight | Direction |
|-----------|--------|-----------|
| scale_score | 0.25 | Higher is better |
| cost_discipline_score | 0.20 | Higher is better |
| aggregator_dependence (penalty) | 0.15 | Lower dependence is better |
| expansion_headroom (balanced) | 0.10 | Moderate-high preferred |
| revenue_quality_score | 0.20 | Higher is better |
| delivery_revenue_mix | 0.05 | Higher contribution to omnichannel growth |
| instore_reliance | 0.05 | Higher direct-channel resilience |

## Score Bands

- `High Potential`: GPI >= 70
- `Moderate Potential`: 45 <= GPI < 70
- `Caution Zone`: GPI < 45

## Current Distribution

| Band | Restaurants | Share % | Avg GPI | Min GPI | Max GPI |
|------|-------------|---------|---------|---------|---------|
| High Potential | 185 | 10.91% | 76.78 | 70.20 | 90.69 |
| Moderate Potential | 785 | 46.29% | 56.12 | 45.08 | 69.82 |
| Caution Zone | 726 | 42.81% | 33.91 | 7.49 | 44.96 |

## Interpretation Guidance

- Use GPI with cluster archetype labels for strategy prioritization, not as a standalone decision rule.
- Recalibrate weights if business strategy changes (for example, margin-first vs growth-first periods).
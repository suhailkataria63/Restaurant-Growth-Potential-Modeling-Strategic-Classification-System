# SkyCity Auckland Restaurants & Bars Analysis

## Project Overview
This project analyzes restaurant and bar data from SkyCity Auckland, focusing on performance metrics, delivery platforms, and profitability.

## Data Cleaning and Preprocessing

### Initial Data Exploration
- **Dataset Source**: SkyCity Auckland Restaurants & Bars.csv
- **Shape**: 1,696 rows × 30 columns
- **Missing Values**: None - all columns complete
- **Data Types**: Mixed (numeric, categorical)

### Preprocessing Steps Completed
1. **Datatype Validation**:
   - Numeric columns converted to appropriate numeric types
   - Categorical columns converted to strings
   - No invalid conversions detected

2. **Column Name Standardization**:
   - All column names converted to lowercase
   - Spaces replaced with underscores
   - Special characters removed
   - Example: 'RestaurantID' → 'restaurantid', 'InStoreShare' → 'instoreshare'

3. **Percentage/Rate Column Verification**:
   - Verified columns: cogsrate, opexrate, commissionrate, instoreshare, ue_share, dd_share, sd_share
   - All columns are numeric and in decimal format (0-1)
   - Value ranges validated:
     - cogsrate: 0.200 - 0.400
     - opexrate: 0.200 - 0.550
     - commissionrate: 0.270 - 0.330
     - instoreshare: 0.030 - 0.550
     - ue_share: 0.350 - 0.600
     - dd_share: 0.200 - 0.300
     - sd_share: 0.150 - 0.450

### Processed Dataset
- **Location**: `data/processed/restaurants_cleaned.csv`
- **Shape**: 1,696 rows × 30 columns (unchanged)
- **Status**: Ready for feature engineering and analysis

## Feature Engineering

### Strategic Business KPIs

Derived the following KPIs to capture key business dimensions for restaurant performance analysis:

1. **scale_score** = monthlyorders × growthfactor
   - **Business Meaning**: Measures current business scale multiplied by growth potential. Higher scores indicate restaurants with strong current performance and future growth prospects.

2. **cost_discipline_score** = 1 - (cogsrate + opexrate)
   - **Business Meaning**: Quantifies operational efficiency in cost management. Higher scores (closer to 1) indicate better cost control and profitability margins.

3. **aggregator_dependence** = ue_share + dd_share
   - **Business Meaning**: Measures reliance on third-party delivery platforms (UberEats + DoorDash). Higher scores indicate greater dependence on aggregators vs. direct channels.

4. **expansion_headroom** = deliveryradiuskm / monthlyorders
   - **Business Meaning**: Assesses geographic reach efficiency. Lower scores suggest better utilization of delivery radius per order. Safely handles division by zero (returns 0 when monthlyorders = 0).

5. **revenue_quality_score** = aov × total_net_profit (normalized)
   - **Business Meaning**: Combines order value with total profitability across all channels. Normalized using StandardScaler to create a standardized profitability metric.

**Helper Metrics for Interpretability**:
- **total_revenue**: Sum of all revenue streams (in-store + delivery platforms)
- **total_net_profit**: Combined net profit across all channels
- **delivery_revenue_mix**: Proportion of revenue from delivery channels vs. total revenue
- **instore_reliance**: Direct measure of in-store share (already available as instoreshare)

**Engineered Dataset**: `data/processed/restaurants_featured.csv` (1,696 rows × 39 columns, includes 5 KPIs + 4 helper metrics)

## Clustering Preparation

### Feature Scaling for Clustering

**Why Scaling is Necessary**:
Clustering algorithms, particularly distance-based methods like K-means, are sensitive to the scale of features. Features with larger numerical ranges can dominate the distance calculations, leading to biased clusters. StandardScaler transforms the data to have a mean of 0 and standard deviation of 1, ensuring all features contribute equally to the clustering process.

**Scaled Numerical Features** (25 features):
- growthfactor
- aov (Average Order Value)
- monthlyorders
- instoreorders
- instorerevenue
- ubereatsorders
- doordashorders
- selfdeliveryorders
- ubereatsrevenue
- doordashrevenue
- selfdeliveryrevenue
- cogsrate (Cost of Goods Sold Rate)
- opexrate (Operating Expense Rate)
- commissionrate
- deliveryradiuskm
- deliverycostperorder
- sd_deliverytotalcost
- instorenetprofit
- ubereatsnetprofit
- doordashnetprofit
- selfdeliverynetprofit
- instoreshare
- ue_share (UberEats Share)
- dd_share (DoorDash Share)
- sd_share (Self-Delivery Share)

**Categorical Features** (excluded from scaling):
- cuisinetype
- restaurantname
- segment
- subregion

**One-Hot Encoding**:
Applied to cuisinetype, segment, and subregion to convert categorical variables into numerical format for clustering. Used drop='first' to avoid multicollinearity. Resulted in 13 additional encoded features:
- cuisinetype_Chicken Dishes, cuisinetype_Chinese, cuisinetype_Indian, cuisinetype_Japanese, cuisinetype_Kebabs/Mediterranean, cuisinetype_Pizza, cuisinetype_Thai
- segment_Full-service, segment_Ghost Kitchen, segment_QSR
- subregion_North Shore, subregion_South Auckland, subregion_West Auckland

**Log Transformation for Skewness**:
Detected highly skewed numerical columns (absolute skewness > 1) among revenue, order count, and delivery cost variables. Applied log(1+x) transformation to normalize distributions:
- instoreorders
- instorerevenue  
- selfdeliveryorders
- selfdeliveryrevenue
- sd_deliverytotalcost

**Final Feature Matrix**: 1,696 rows × 38 columns (25 scaled numerical + 13 encoded categorical)

**Scaler Saved**: `data/processed/scaler.pkl` for transforming new data consistently.
**Encoder Saved**: `data/processed/encoder.pkl` for encoding new categorical data.
**Feature Matrix Saved**: `data/processed/feature_matrix.csv` ready for clustering algorithms.

## Dimensionality Reduction

### Latent Structure Discovery for Clustering

Implemented a dedicated dimensionality reduction step in `src/dimensionality_reduction.py` and exposed it through `src/clustering.py`.

### Methods Included
1. **PCA (Principal Component Analysis)**:
   - Linear dimensionality reduction for explainable latent factor discovery.
   - Generates principal component scores for each restaurant.
   - Produces explained variance summary and loadings-based interpretation notes.

2. **UMAP (Optional)**:
   - Nonlinear 2D embedding for visual structure discovery.
   - Runs only when the `umap-learn` package is available in the environment.

### Artifacts Generated
- **PCA features**: `data/processed/pca_features.csv`
- **UMAP features** (if available): `data/processed/umap_features.csv`
- **PCA variance + loadings summary**: `reports/pca_summary.md`
- **Scree plot**: `reports/figures/pca_scree_plot.png`
- **PCA 2D scatter**: `reports/figures/pca_2d_scatter.png`
- **UMAP 2D embedding** (if available): `reports/figures/umap_2d_embedding.png`

### Factor Interpretation Notes (Business Hypothesis Layer)
PCA components are interpreted using strongest positive and negative feature loadings. These component labels are directional hypotheses to support strategy interpretation.

- **Cost pressure**:
  - Usually linked to stronger loadings from `cogsrate`, `opexrate`, `commissionrate`, and delivery-cost fields.
  - Higher score direction often reflects tighter margin pressure or cost-heavy operating structures.

- **Channel leverage**:
  - Usually linked to delivery/in-store mix features such as `instoreshare`, `ue_share`, `dd_share`, and `sd_share`.
  - Distinguishes restaurants with stronger aggregator reliance versus direct-channel positioning.

- **Growth momentum**:
  - Usually linked to `growthfactor`, order volume, revenue, and net profit signals.
  - Captures trajectory strength and compounding demand movement.

- **Scalability**:
  - Usually linked to `deliveryradiuskm`, channel-operational spread, and volume-related utilization patterns.
  - Helps identify formats with stronger expansion headroom versus constrained operating models.

These themes are used as interpretation aids and should be validated against final clustering outputs before assigning archetype labels.

## Clustering Analysis

### Methods Tested
1. **K-Means** on PCA feature space (`PC1`-`PC10`)
   - Evaluated across `k = 2..8`
   - Metrics tracked: inertia (elbow) and silhouette score

2. **Agglomerative (Hierarchical) Clustering** on PCA feature space
   - Ward linkage
   - Cluster count matched to best K-Means `k` for direct comparison
   - Dendrogram generated on sampled observations for readability

3. **DBSCAN (Optional Robustness Check)** on full scaled feature matrix
   - Used as a density-based sensitivity check against centroid/linkage methods
   - Noise points retained as `-1` labels

### Model Selection Reasoning
- Primary candidate selection was based on silhouette score in PCA space.
- K-Means achieved the highest silhouette in the sweep:
  - **Best K-Means**: `k=3`, silhouette `0.2066`
  - **Hierarchical (k=3)** silhouette: `0.1897`
- DBSCAN produced 2 non-noise clusters with 86 noise points, which was useful for robustness but not selected as the primary segmentation method.

### Chosen Clustering Configuration
- **Selected method**: `K-Means`
- **Chosen number of clusters**: `3`
- **Final export**: `data/processed/clustered_restaurants.csv`

The final clustered file includes:
- original identifier fields (`restaurantid`, `restaurantname`, `cuisinetype`, `segment`, `subregion`)
- selected strategic KPIs
- PCA coordinates (`PC1`-`PC10`)
- assigned labels from K-Means, hierarchical, DBSCAN, and the selected primary label (`selected_cluster`)

### Clustering Artifacts Generated
- `reports/figures/kmeans_elbow_plot.png`
- `reports/figures/kmeans_silhouette_plot.png`
- `reports/figures/hierarchical_dendrogram.png`
- `data/processed/clustered_restaurants.csv`

## Model Evaluation

Implemented a dedicated evaluation module in `src/evaluation.py` to quantify clustering quality using the final selected cluster labels (`selected_cluster`) and the clustering-ready feature matrix.

### Final Selected Solution Metrics

For the current selected solution (`kmeans`, `k=3`):
- **Silhouette Score**: `0.1930`
- **Calinski-Harabasz Score**: `390.2882`
- **Davies-Bouldin Score**: `1.7151`

### K-Means k-Range Evaluation (`k=2..8`)

K-Means was re-evaluated across the tested range and exported to:
- `reports/model_evaluation/kmeans_k_comparison.csv`

Key pattern from the sweep:
- Best **Silhouette** at `k=3` (supports current selection).
- Best **Davies-Bouldin** at `k=4` (slightly lower overlap at 4 clusters).
- Best **Calinski-Harabasz** at `k=2` (strongest global separation at lower k).

This trade-off indicates `k=3` remains a balanced operating choice between interpretability and separation quality.

### Metric Interpretation Guide
- **Silhouette Score**: higher is better; reflects cohesion vs. separation.
- **Calinski-Harabasz Score**: higher is better; ratio of between-cluster to within-cluster dispersion.
- **Davies-Bouldin Score**: lower is better; penalizes cluster overlap/similarity.

### Evaluation Outputs
- Metrics summary: `reports/model_evaluation/clustering_metrics.md`
- K comparison table: `reports/model_evaluation/kmeans_k_comparison.csv`
- Silhouette trend: `reports/figures/eval_silhouette_by_k.png`
- Davies-Bouldin trend: `reports/figures/eval_davies_bouldin_by_k.png`
- Calinski-Harabasz trend: `reports/figures/eval_calinski_harabasz_by_k.png`

### Clustering Stability Evaluation

To evaluate robustness beyond point estimates, `src/evaluation.py` includes two stability checks for the selected K-Means solution:

1. **Seed Re-run Stability**  
   - Re-runs K-Means multiple times with different random seeds.
   - Compares assignments to a baseline run using:
     - **Adjusted Rand Index (ARI)**
     - **Normalized Mutual Information (NMI)**

2. **80% Subsample Stability**  
   - Randomly samples 80% of restaurants multiple times.
   - Re-clusters each sample and compares sample assignments against baseline sample assignments using ARI/NMI.
   - Also tracks centroid consistency via mean matched centroid shift.

Current stability summary indicates strong robustness:
- Seed reruns: ARI `1.0000`, NMI `1.0000`
- 80% subsamples: ARI `0.9761`, NMI `0.9651`
- Overall verdict: **Highly robust cluster structure**

### Stability Outputs
- Stability report: `reports/model_evaluation/cluster_stability.md`
- Stability run summary: `reports/model_evaluation/cluster_stability_summary.csv`
- ARI stability plot: `reports/figures/cluster_stability_ari.png`
- NMI stability plot: `reports/figures/cluster_stability_nmi.png`

## Cluster Interpretation and Labeling

### Business Archetypes Identified

Using cluster-level KPI comparisons against overall dataset averages, each cluster was assigned a business-friendly archetype and description.

1. **Cluster 0 - High-Growth / High-Risk** (371 restaurants, 21.9%)
   - **Interpretation**: High scale momentum, but weak cost discipline and negative profit quality indicate fragile growth.
   - **Business significance**: This segment can grow, but requires immediate margin stabilization (cost controls, pricing, and channel profitability governance).

2. **Cluster 1 - Scalable Profit Leaders** (664 restaurants, 39.2%)
   - **Interpretation**: High scale with strong profitability and healthier cost structure.
   - **Business significance**: This is the expansion-ready segment for targeted investment, premium positioning, and replication playbooks.

3. **Cluster 2 - Aggregator-Dependent Low Margin** (661 restaurants, 39.0%)
   - **Interpretation**: Strong dependence on aggregator channels with weaker relative return quality.
   - **Business significance**: Priority should be channel rebalancing, commission optimization, and direct/self-delivery conversion to protect margins.

### Interpretation Outputs
- Cluster statistics table: `reports/cluster_summary.csv`
- Narrative profiles: `reports/cluster_profiles.md`
- KPI comparison visual: `reports/figures/cluster_kpi_comparison.png`
- Enriched clustered dataset with archetype metadata:
  - `data/processed/clustered_restaurants.csv`
  - includes `cluster_label_name` and `cluster_description`

## Growth Potential Index (GPI)

### Composite Scoring Framework

Implemented a weighted composite Growth Potential Index in `src/scoring.py` and merged the result back into `data/processed/clustered_restaurants.csv`.

### GPI Formula

1. Normalize each component to a 0-1 range before combining.
2. Apply weighted aggregation:

`GPI_raw = Σ(weight_i × normalized_component_i)`

`GPI_score = GPI_raw × 100`

### Weights Used
- `scale_score`: 0.25
- `cost_discipline_score`: 0.20
- `aggregator_dependence` (penalty / inverted): 0.15
- `expansion_headroom` (balanced contribution): 0.10
- `revenue_quality_score`: 0.20
- `delivery_revenue_mix`: 0.05
- `instore_reliance`: 0.05

### Rationale for Weights
- **Scale + Revenue Quality (0.45 total)**: captures momentum and quality of growth.
- **Cost + Aggregator Risk (0.35 total)**: enforces margin discipline and channel-risk awareness.
- **Expansion + Channel Mix (0.20 total)**: adds strategic execution readiness and resilience context.

### Score Bands
- **High Potential**: `GPI >= 70`
- **Moderate Potential**: `45 <= GPI < 70`
- **Caution Zone**: `GPI < 45`

### Band Meanings
- **High Potential**: strong expansion candidates with healthy multi-factor fundamentals.
- **Moderate Potential**: viable but requires targeted optimization before aggressive scaling.
- **Caution Zone**: meaningful structural constraints; prioritize stabilization over expansion.

### GPI Outputs
- Methodology report: `reports/gpi_methodology.md`
- Score distribution summary: `reports/gpi_summary.csv`
- Enriched clustered dataset now includes:
  - `gpi_score` (0-100)
  - `gpi_band`
  - `gpi_rank`
  - normalized GPI component columns (`gpi_*_norm`)

## Strategy Recommendation Engine

### Recommendation Logic

Implemented a rule-based recommendation engine in `src/recommendation_engine.py` that assigns one primary strategic action per restaurant using:
- cluster archetype
- GPI score and GPI band
- channel mix and aggregator dependence
- cost discipline
- expansion headroom
- revenue quality

### Primary Action Labels
- `Scale Aggressively`
- `Expand Carefully`
- `Rebalance Channels`
- `Optimize`
- `Stabilize Operations`

Each restaurant receives:
- `strategy_recommendation`
- `recommendation_reason` (short explanation for why the action was assigned)

### Business Interpretation of Actions
- **Scale Aggressively**: used for high-potential, strong-economics operators where expansion can be accelerated.
- **Expand Carefully**: used when growth opportunity exists but risk signals require phased execution.
- **Rebalance Channels**: used when aggregator-heavy mix and channel risk call for direct/self-delivery strengthening.
- **Optimize**: used for moderate performers where operational and commercial tuning should come before scale.
- **Stabilize Operations**: used for low-potential, stressed economics segments that need margin and process recovery.

### Recommendation Outputs
- `reports/recommendation_summary.csv`
- `reports/strategy_playbook.md`
- `data/processed/clustered_restaurants.csv` (enriched with recommendation fields)

## Dashboard Data Preparation

### Dashboard Prep Module

Implemented `src/dashboard_prep.py` to convert `data/processed/clustered_restaurants.csv` into dashboard-ready datasets and summary payloads for frontend consumption.

### Generated Dashboard Artifacts

- `data/processed/dashboard_summary.json`
  - overall KPI summary cards payload
  - cluster-wise count + average GPI payload
  - recommendation-wise count payload
  - GPI band counts and output file references
- `data/processed/top_restaurants.csv`
  - top restaurants ranked by GPI (`gpi_rank`, `gpi_score`, cluster, recommendation, and key financial context)
- `data/processed/cluster_dashboard_summary.csv`
  - cluster-wise counts, share, average GPI, average profit, and dominant recommendation
- `data/processed/filter_summary_tables/`
  - `subregion_summary.csv`
  - `cuisine_summary.csv`
  - `segment_summary.csv`

### How These Files Feed the Streamlit Dashboard

- **KPI cards**: read `dashboard_summary.json -> overall_kpi_summary_cards`
- **Cluster charts/tables**: use `cluster_dashboard_summary.csv` or `dashboard_summary.json -> cluster_summary`
- **Recommendation distribution chart**: use `dashboard_summary.json -> recommendation_counts`
- **Top performers table**: use `top_restaurants.csv`
- **Filter drill-downs**: use `filter_summary_tables/*.csv` based on selected view (subregion/cuisine/segment)

## Frontend and Deployment

### Primary UI

- **Primary dashboard UI**: `app/streamlit_app.py` (Streamlit)
- **Reference UI**: `app/restaurant-growth-dashboard` (Next.js) retained as a design reference.

### Run Streamlit Dashboard Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Open: `http://localhost:8501`

### Deploy Streamlit Dashboard (Recommended: Streamlit Community Cloud)

1. Push repository to GitHub.
2. Go to Streamlit Community Cloud and click **New app**.
3. Select repository and branch.
4. Set **Main file path** to `app/streamlit_app.py`.
5. Deploy.

### Optional: Run Next.js Reference Dashboard Locally

```bash
cd app/restaurant-growth-dashboard
npm install
npm run dev
```

Open: `http://localhost:3000`

### Optional: Deploy Next.js Reference UI (Vercel)

#### Option A: Vercel Web UI
1. Push repository to GitHub.
2. Go to Vercel and click **New Project**.
3. Import this repository.
4. Set **Root Directory** to `app/restaurant-growth-dashboard`.
5. Keep build settings as default for Next.js:
   - Build command: `next build`
   - Output: `.next`
6. Deploy.

#### Option B: Vercel CLI
```bash
npm i -g vercel
cd app/restaurant-growth-dashboard
vercel
vercel --prod
```

## Reproducibility

### Reproduce From Fresh Clone

```bash
# 1) Clone into a new folder
git clone https://github.com/suhailkataria63/Restaurant-Growth-Potential-Modeling-Strategic-Classification-System.git restaurant-growth-repro
cd restaurant-growth-repro

# 2) Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Rebuild the full pipeline (raw data -> all processed/report artifacts)
make reproduce

# 5) Run the Streamlit web app
streamlit run app/streamlit_app.py
```

### One-Command Pipeline Rebuild

Run the full pipeline from raw data to final evaluation artifacts:

```bash
make reproduce
```

This executes:
- preprocessing (raw CSV -> cleaned + feature matrix)
- feature engineering
- dimensionality reduction (PCA + optional UMAP)
- clustering and interpretation
- GPI scoring
- recommendation generation
- dashboard prep
- model evaluation + stability checks

### Direct Script Option

```bash
bash scripts/reproduce.sh
```

### Reproducibility Notes

- `scripts/reproduce.sh` auto-uses `.venv/bin/python` if available.
- It sets reproducibility-friendly runtime env vars for matplotlib/numba caches.
- `dashboard_summary.json` uses `DASHBOARD_GENERATED_AT_UTC` if provided, otherwise current UTC time.
- For deterministic rebuild checks:

```bash
export DASHBOARD_GENERATED_AT_UTC="1970-01-01T00:00:00+00:00"
make reproduce
```

## Project Structure
```
sky/
├── Makefile
├── data/
│   ├── raw/
│   │   └── SkyCity Auckland Restaurants & Bars.csv
│   └── processed/
│       ├── restaurants_cleaned.csv
│       ├── restaurants_featured.csv
│       ├── scaler.pkl
│       ├── encoder.pkl
│       ├── feature_matrix.csv
│       ├── pca_features.csv
│       ├── clustered_restaurants.csv
│       ├── dashboard_summary.json
│       ├── top_restaurants.csv
│       ├── cluster_dashboard_summary.csv
│       ├── filter_summary_tables/
│       │   ├── subregion_summary.csv
│       │   ├── cuisine_summary.csv
│       │   └── segment_summary.csv
│       └── umap_features.csv
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── dimensionality_reduction.py
│   ├── cluster_interpretation.py
│   ├── clustering.py
│   ├── scoring.py
│   ├── recommendation_engine.py
│   ├── dashboard_prep.py
│   ├── evaluation.py
│   └── utils.py
├── app/
│   ├── streamlit_app.py
│   └── restaurant-growth-dashboard/
│       ├── app/
│       ├── components/
│       ├── lib/
│       └── package.json
├── scripts/
│   └── reproduce.sh
├── reports/
│   ├── analysis.md
│   ├── pca_summary.md
│   ├── cluster_summary.csv
│   ├── cluster_profiles.md
│   ├── gpi_methodology.md
│   ├── gpi_summary.csv
│   ├── recommendation_summary.csv
│   ├── strategy_playbook.md
│   ├── model_evaluation/
│   │   ├── clustering_metrics.md
│   │   ├── kmeans_k_comparison.csv
│   │   ├── cluster_stability.md
│   │   └── cluster_stability_summary.csv
│   └── figures/
│       ├── pca_scree_plot.png
│       ├── pca_2d_scatter.png
│       ├── kmeans_elbow_plot.png
│       ├── kmeans_silhouette_plot.png
│       ├── hierarchical_dendrogram.png
│       ├── cluster_kpi_comparison.png
│       ├── eval_silhouette_by_k.png
│       ├── eval_davies_bouldin_by_k.png
│       ├── eval_calinski_harabasz_by_k.png
│       ├── cluster_stability_ari.png
│       ├── cluster_stability_nmi.png
│       └── umap_2d_embedding.png
├── requirements.txt
└── README.md
```

## Next Steps
- Strategy simulator by scenario (cost shock, channel shift, demand growth)
- Recommendation confidence scoring and feedback loop
- Production deployment hardening for Streamlit (monitoring, auth, and observability)
- CI checks for data artifact freshness and dashboard smoke tests

## Progress Log (4-Week Retrospective)
This timeline is summarized by weekly phases for readability; Git commit history remains the exact source of truth.

- **Week 1 - Data Foundation**: Initial setup, data preprocessing pipeline, and strategic feature engineering (KPIs + helper metrics) completed with baseline processed artifacts.
- **Week 2 - Latent Structure + Clustering**: Added dimensionality reduction (PCA + optional UMAP), explained-variance reporting, K-Means sweep (`k=2..8`), hierarchical clustering, optional DBSCAN robustness checks, and clustering diagnostics outputs.
- **Week 3 - Business Intelligence Layer**: Added business-level cluster interpretation and archetype labeling, cluster-vs-overall KPI summaries, profile narratives, and enriched clustered dataset with `cluster_label_name` and `cluster_description`.
- **Week 4 - Scoring, Recommendations, and Dashboard Prep**: Added Growth Potential Index (GPI), recommendation engine (`Scale Aggressively`, `Expand Carefully`, `Rebalance Channels`, `Optimize`, `Stabilize Operations`), summary/playbook reports, dashboard-ready exports, and a polished Streamlit dashboard with theme-aware readability, hover tooltips, and improved chart usability.
- **Week 4 - Reproducibility Hardening**: Fixed raw-data preprocessing entry flow, added deterministic evaluation CSV handling, removed tracked bytecode noise, and added `make reproduce` / `scripts/reproduce.sh` for one-command end-to-end rebuilds.

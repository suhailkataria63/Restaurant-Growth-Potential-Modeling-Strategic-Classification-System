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

**Engineered Dataset**: `data/processed/restaurants_featured.csv` (1,696 rows × 35 columns, includes 5 new KPI features)

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

## Project Structure
```
sky/
├── data/
│   ├── raw/
│   │   └── SkyCity Auckland Restaurants & Bars.csv
│   └── processed/
│       ├── restaurants_cleaned.csv
│       ├── restaurants_featured.csv
│       ├── scaler.pkl
│       ├── encoder.pkl
│       └── feature_matrix.csv
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── clustering.py
│   ├── scoring.py
│   └── utils.py
├── app/
│   └── streamlit_app.py
├── reports/
│   └── analysis.md
├── requirements.txt
└── README.md
```

## Next Steps
- Feature engineering in `src/feature_engineering.py`
- Clustering analysis in `src/clustering.py`
- Scoring model in `src/scoring.py`
- Streamlit app development in `app/streamlit_app.py`
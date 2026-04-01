import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def engineer_features(input_path='data/processed/restaurants_cleaned.csv', output_path='data/processed/restaurants_featured.csv'):
    """
    Engineer strategic business KPIs from the cleaned restaurant dataset.
    
    KPIs:
    - scale_score: monthlyorders * growthfactor (business scale and growth potential)
    - cost_discipline_score: 1 - (cogsrate + opexrate) (efficiency in cost management)
    - aggregator_dependence: ue_share + dd_share (reliance on third-party platforms)
    - expansion_headroom: deliveryradiuskm / monthlyorders (geographic reach efficiency, safe divide)
    - revenue_quality_score: aov * total_net_profit (profitability per order, normalized)
    """
    # Load cleaned data
    df = pd.read_csv(input_path)
    
    # Compute KPIs
    df['scale_score'] = df['monthlyorders'] * df['growthfactor']
    
    df['cost_discipline_score'] = 1 - (df['cogsrate'] + df['opexrate'])
    
    df['aggregator_dependence'] = df['ue_share'] + df['dd_share']
    
    # Safe division for expansion_headroom
    df['expansion_headroom'] = np.where(df['monthlyorders'] == 0, 0, df['deliveryradiuskm'] / df['monthlyorders'])
    
    # Revenue quality score
    total_net_profit = (df['instorenetprofit'] + df['ubereatsnetprofit'] + 
                       df['doordashnetprofit'] + df['selfdeliverynetprofit'])
    df['revenue_quality_score_raw'] = df['aov'] * total_net_profit
    
    # Normalize revenue_quality_score using StandardScaler
    scaler = StandardScaler()
    df['revenue_quality_score'] = scaler.fit_transform(df[['revenue_quality_score_raw']])
    
    # Drop the raw column
    df = df.drop('revenue_quality_score_raw', axis=1)
    
    # Helper metrics for interpretability
    df['total_revenue'] = (df['instorerevenue'] + df['ubereatsrevenue'] + 
                          df['doordashrevenue'] + df['selfdeliveryrevenue'])
    
    df['total_net_profit'] = total_net_profit
    
    df['delivery_revenue_mix'] = (df['ubereatsrevenue'] + df['doordashrevenue'] + df['selfdeliveryrevenue']) / df['total_revenue']
    df['delivery_revenue_mix'] = df['delivery_revenue_mix'].fillna(0)  # Handle division by zero
    
    df['instore_reliance'] = df['instoreshare']
    
    # Save engineered dataset
    df.to_csv(output_path, index=False)
    print(f"Engineered features saved to {output_path}")
    print(f"New columns added: scale_score, cost_discipline_score, aggregator_dependence, expansion_headroom, revenue_quality_score, total_revenue, total_net_profit, delivery_revenue_mix, instore_reliance")
    print(f"Final shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    engineer_features()
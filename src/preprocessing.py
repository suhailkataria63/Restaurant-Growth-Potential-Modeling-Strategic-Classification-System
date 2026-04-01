import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
from scipy.stats import skew

def validate_datatypes(df):
    """
    Validate and convert datatypes for the dataset.
    Ensures numerical columns are float/int, categorical are string.
    """
    # Define expected dtypes
    numeric_cols = [
        'restaurantid', 'growthfactor', 'aov', 'monthlyorders', 'instoreorders', 'instorerevenue',
        'ubereatsorders', 'doordashorders', 'selfdeliveryorders', 'ubereatsrevenue', 'doordashrevenue',
        'selfdeliveryrevenue', 'cogsrate', 'opexrate', 'commissionrate', 'deliveryradiuskm',
        'deliverycostperorder', 'sd_deliverytotalcost', 'instorenetprofit', 'ubereatsnetprofit',
        'doordashnetprofit', 'selfdeliverynetprofit', 'instoreshare', 'ue_share', 'dd_share', 'sd_share'
    ]
    categorical_cols = ['cuisinetype', 'restaurantname', 'segment', 'subregion']

    # Convert numeric columns
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert categorical columns
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Check for conversion issues
    invalid_numeric = df[numeric_cols].isnull().sum().sum()
    if invalid_numeric > 0:
        print(f"Warning: {invalid_numeric} invalid numeric values found and converted to NaN")

    return df

def standardize_column_names(df):
    """
    Standardize column names: lowercase, replace spaces with underscores, remove special characters.
    """
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('[^a-z0-9_]', '', regex=True)
    return df

def verify_percentage_columns(df):
    """
    Verify that percentage/rate columns are numeric and within expected ranges (0-1 or 0-100).
    """
    percentage_cols = ['cogsrate', 'opexrate', 'commissionrate', 'instoreshare', 'ue_share', 'dd_share', 'sd_share']

    for col in percentage_cols:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"Warning: {col} is not numeric")
            else:
                # Check if values are between 0 and 1 (assuming decimal), or 0-100 (percentage)
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val <= 1:
                    print(f"{col}: Values in decimal format (0-1), min: {min_val}, max: {max_val}")
                elif max_val <= 100:
                    print(f"{col}: Values in percentage format (0-100), min: {min_val}, max: {max_val}")
                    # Optionally convert to decimal
                    # df[col] = df[col] / 100
                else:
                    print(f"Warning: {col} has unexpected range, min: {min_val}, max: {max_val}")

    return df

def preprocess_data(input_path, output_path='data/processed/restaurants_cleaned.csv'):
    """
    Main preprocessing function: load, validate, standardize, verify, and save.
    """
    # Load data
    df = pd.read_csv(input_path)

    # Apply preprocessing steps
    df = validate_datatypes(df)
    df = standardize_column_names(df)
    df = verify_percentage_columns(df)

    # Save processed data
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")
    print(f"Final shape: {df.shape}")
    print(f"Final columns: {df.columns.tolist()}")

    return df

def prepare_for_clustering(df):
    """
    Prepare data for clustering by separating numerical and categorical columns,
    applying StandardScaler to numerical features, one-hot encoding selected categorical features,
    and saving the scaler, encoder, and transformed feature matrix.
    Returns the final feature matrix.
    """
    # Separate numerical and categorical columns
    numerical_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col not in ['restaurantid']]
    categorical_cols = [col for col in df.columns if col not in numerical_cols]

    print(f"Numerical columns for clustering: {numerical_cols}")
    print(f"Categorical columns: {categorical_cols}")

    # Select numerical data
    df_num = df[numerical_cols]

    # Detect and transform highly skewed columns (revenue, order count, delivery cost)
    skewed_cols = []
    transform_candidates = [
        'instorerevenue', 'ubereatsrevenue', 'doordashrevenue', 'selfdeliveryrevenue',
        'monthlyorders', 'instoreorders', 'ubereatsorders', 'doordashorders', 'selfdeliveryorders',
        'deliverycostperorder', 'sd_deliverytotalcost'
    ]
    
    for col in df_num.columns:
        if col in transform_candidates:
            col_skew = skew(df_num[col])
            if abs(col_skew) > 1:  # highly skewed
                skewed_cols.append(col)
                df_num[col] = np.log1p(df_num[col])  # log(1+x) to handle potential zeros
    
    print(f"Log-transformed columns (skewness > 1): {skewed_cols}")

    # Apply StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_num)

    # Create scaled dataframe
    df_scaled = pd.DataFrame(scaled_data, columns=numerical_cols, index=df.index)

    # One-hot encode selected categorical columns
    encode_cols = ['cuisinetype', 'segment', 'subregion']
    encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop first to avoid multicollinearity
    cat_encoded = encoder.fit_transform(df[encode_cols])
    cat_feature_names = encoder.get_feature_names_out(encode_cols)
    df_cat_encoded = pd.DataFrame(cat_encoded, columns=cat_feature_names, index=df.index)

    # Combine scaled numerical and encoded categorical
    df_final = pd.concat([df_scaled, df_cat_encoded], axis=1)

    # Save scaler, encoder, and feature matrix
    scaler_path = 'data/processed/scaler.pkl'
    encoder_path = 'data/processed/encoder.pkl'
    matrix_path = 'data/processed/feature_matrix.csv'

    joblib.dump(scaler, scaler_path)
    joblib.dump(encoder, encoder_path)
    df_final.to_csv(matrix_path, index=False)

    print(f"One-hot encoded columns: {encode_cols}")
    print(f"Encoded features: {cat_feature_names.tolist()}")
    print(f"Final feature matrix shape: {df_final.shape}")
    print(f"Scaler saved to {scaler_path}")
    print(f"Encoder saved to {encoder_path}")
    print(f"Feature matrix saved to {matrix_path}")

    return df_final

if __name__ == "__main__":
    # Run preprocessing on the cleaned data
    df = preprocess_data('data/processed/restaurants_cleaned.csv')
    # Prepare for clustering
    df_final = prepare_for_clustering(df)
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
import os
from typing import List, Tuple

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/data_preprocessing.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Preprocessor")

class WalkForwardPreprocessor:
    """
    Handles chronological splitting and normalization to prevent data leakage.
    Splits data into multiple folds for walk-forward validation.
    """
    def __init__(self, 
                 train_years: float = 3.0, 
                 val_months: int = 6, 
                 test_months: int = 6,
                 target_col: str = "return_1d"):
        self.train_years = train_years
        self.val_months = val_months
        self.test_months = test_months
        self.target_col = target_col
        self.scaler = StandardScaler()

    def create_folds(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Creates a list of (train, val, test) DataFrames using a rolling window.
        """
        if 'Date' in df.columns:
            df = df.set_index('Date')
            
        df = df.sort_index()
        dates = pd.to_datetime(df.index.unique())
        start_date = dates[0]
        end_date = dates[-1]
        
        folds = []
        current_test_start = start_date + pd.DateOffset(years=self.train_years) + pd.DateOffset(months=self.val_months)
        
        while current_test_start + pd.DateOffset(months=self.test_months) <= end_date:
            test_end = current_test_start + pd.DateOffset(months=self.test_months)
            val_start = current_test_start - pd.DateOffset(months=self.val_months)
            train_start = val_start - pd.DateOffset(years=self.train_years)
            
            # If train_start is before available data, adjust to first date
            if train_start < start_date:
                train_start = start_date
                
            train_df = df[(df.index >= train_start) & (df.index < val_start)]
            val_df = df[(df.index >= val_start) & (df.index < current_test_start)]
            test_df = df[(df.index >= current_test_start) & (df.index < test_end)]
            
            if not train_df.empty and not val_df.empty and not test_df.empty:
                folds.append((train_df, val_df, test_df))
                logger.info(f"Fold created: Train {train_start.date()} to {val_start.date()}, "
                            f"Val to {current_test_start.date()}, Test to {test_end.date()}")
            
            # Slide window forward by test_months
            current_test_start = test_end
            
        return folds

    def prepare_fold_data(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Fits scaler on TRAIN ONLY and transforms all sets.
        Returns scaled arrays and feature names.
        """
        # Define features (Exclude Symbol, non-numeric, and internal tracking cols)
        exclude = ['Symbol', 'high_52w', 'low_52w', 'ema_20', 'ema_50', 'ema_200', 'vol_ma20', 'vol_percentile']
        features = [c for c in train.columns if c not in exclude and train[c].dtype in [np.float64, np.int64, np.int32]]
        
        X_train = train[features].values
        X_val = val[features].values
        X_test = test[features].values
        
        # Fit ONLY on training data
        self.scaler.fit(X_train)
        
        # Transform all
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Validation: check for NaNs
        assert not np.isnan(X_train_scaled).any(), "NaN found in scaled training data"
        assert not np.isnan(X_val_scaled).any(), "NaN found in scaled validation data"
        assert not np.isnan(X_test_scaled).any(), "NaN found in scaled test data"
        
        return X_train_scaled, X_val_scaled, X_test_scaled, features

if __name__ == "__main__":
    # Test script
    try:
        df = pd.read_parquet("data/processed_market_data.parquet")
        preprocessor = WalkForwardPreprocessor()
        folds = preprocessor.create_folds(df)
        
        if folds:
            train, val, test = folds[0]
            X_tr, X_va, X_te, feat_names = preprocessor.prepare_fold_data(train, val, test)
            print(f"Successfully processed Fold 1.")
            print(f"Feature count: {len(feat_names)}")
            print(f"X_train shape: {X_tr.shape}")
        else:
            print("No folds created. check data date range.")
    except FileNotFoundError:
        print("Data file not found. Run pipeline.py first.")

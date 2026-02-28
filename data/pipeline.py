import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import logging
from typing import List, Optional

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/data_pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("DataPipeline")

class DataFetcher:
    """
    Handles fetching market data from yfinance and nsepy (fallback).
    Caches data locally in parquet format.
    """
    def __init__(self, raw_data_dir: str = "data/raw"):
        self.raw_data_dir = raw_data_dir
        os.makedirs(self.raw_data_dir, exist_ok=True)
        self.benchmarks = ["^NSEI", "GOLDBEES.NS", "SILVERBEES.NS"]
        
    def fetch_stock_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches OHLCV data for multiple symbols.
        """
        all_data = []
        for symbol in symbols + self.benchmarks:
            df = self._get_single_stock(symbol, start_date, end_date)
            if df is not None:
                df['Symbol'] = symbol
                all_data.append(df)
        
        if not all_data:
            raise ValueError("No data fetched for any symbol.")
            
        return pd.concat(all_data)

    def _get_single_stock(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        cache_path = os.path.join(self.raw_data_dir, f"{symbol.replace('^', 'INDEX_')}.parquet")
        
        # Freshness check (1 day)
        if os.path.exists(cache_path):
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
            if datetime.now() - file_mod_time < timedelta(days=1):
                logger.info(f"Loading {symbol} from cache.")
                return pd.read_parquet(cache_path)
        
        logger.info(f"Fetching {symbol} from yfinance...")
        try:
            df = yf.download(symbol, start=start_date, end=end_date, interval="1d", auto_adjust=True)
            if df.empty:
                logger.warning(f"No data found for {symbol}")
                return None
            
            # If the dataframe has MultiIndex columns (recent yfinance change), flatten it
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Reset index to make Date a column for easier processing
            df = df.reset_index()
            
            # Save to cache
            df.to_parquet(cache_path)
            return df
        except Exception as e:
            logger.error(f"Error fetching {symbol} from yfinance: {e}")
            # Fallback to nsepy could be implemented here if needed
            return None

class DataCleaner:
    """
    Handles missing values, corporate actions, and data validation.
    """
    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies cleaning rules: forward fill max 3 days, else drop.
        """
        # Ensure index is datetime
        df.index = pd.to_datetime(df.index)
        
        # Sort by symbol and date
        df = df.sort_values(['Symbol', 'Date'])
        
        # Handle missing values per symbol
        def _clean_symbol(group):
            # Forward fill missing values with limit 3
            group = group.ffill(limit=3)
            # Drop any remaining NaNs
            group = group.dropna()
            return group

        cleaned_df = df.groupby('Symbol', group_keys=False).apply(_clean_symbol)
        
        logger.info(f"Data cleaning complete. Original rows: {len(df)}, Cleaned rows: {len(cleaned_df)}")
        return cleaned_df

class FeatureEngineer:
    """
    Calculates technical indicators and fundamental price/volume features.
    Ensures NO data leakage by using only historical data.
    """
    @staticmethod
    def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates returns and momentum features.
        """
        # Close is already adjusted by auto_adjust=True in yfinance
        df['return_1d'] = df.groupby('Symbol')['Close'].pct_change(1)
        df['return_5d'] = df.groupby('Symbol')['Close'].pct_change(5)
        df['return_10d'] = df.groupby('Symbol')['Close'].pct_change(10)
        df['return_21d'] = df.groupby('Symbol')['Close'].pct_change(21)
        
        df['log_return'] = np.log(df['Close'] / df.groupby('Symbol')['Close'].shift(1))
        
        # 52-week range position
        df['high_52w'] = df.groupby('Symbol')['High'].transform(lambda x: x.rolling(252, min_periods=1).max())
        df['low_52w'] = df.groupby('Symbol')['Low'].transform(lambda x: x.rolling(252, min_periods=1).min())
        df['price_vs_52w_high'] = (df['Close'] - df['low_52w']) / (df['high_52w'] - df['low_52w'])
        
        # EMA distances
        for span in [20, 50, 200]:
            df[f'ema_{span}'] = df.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=span, adjust=False).mean())
            df[f'dist_ema_{span}'] = (df['Close'] - df[f'ema_{span}']) / df[f'ema_{span}']
            
        return df

    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates RSI, MACD, Bollinger Bands, ATR, etc.
        """
        def _calc_indicators(group):
            # RSI (14)
            delta = group['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            group['rsi_14'] = 100 - (100 / (1 + rs))
            
            # MACD (12, 26, 9)
            exp1 = group['Close'].ewm(span=12, adjust=False).mean()
            exp2 = group['Close'].ewm(span=26, adjust=False).mean()
            group['macd_line'] = exp1 - exp2
            group['macd_signal'] = group['macd_line'].ewm(span=9, adjust=False).mean()
            group['macd_histogram'] = group['macd_line'] - group['macd_signal']
            
            # BB (20, 2)
            ma20 = group['Close'].rolling(window=20).mean()
            std20 = group['Close'].rolling(window=20).std()
            group['bb_upper'] = ma20 + (std20 * 2)
            group['bb_lower'] = ma20 - (std20 * 2)
            group['bb_percent_b'] = (group['Close'] - group['bb_lower']) / (group['bb_upper'] - group['bb_lower'])
            group['bb_bandwidth'] = (group['bb_upper'] - group['bb_lower']) / ma20
            
            # ATR (14)
            high_low = group['High'] - group['Low']
            high_cp = np.abs(group['High'] - group['Close'].shift())
            low_cp = np.abs(group['Low'] - group['Close'].shift())
            df_temp = pd.concat([high_low, high_cp, low_cp], axis=1)
            tr = df_temp.max(axis=1)
            group['atr_14'] = tr.rolling(window=14).mean()
            
            # OBV
            group['obv'] = (np.sign(group['Close'].diff()) * group['Volume']).fillna(0).cumsum()
            
            # Stoch (14, 3)
            low_14 = group['Low'].rolling(window=14).min()
            high_14 = group['High'].rolling(window=14).max()
            group['stoch_k'] = 100 * (group['Close'] - low_14) / (high_14 - low_14)
            group['stoch_d'] = group['stoch_k'].rolling(window=3).mean()
            
            return group

        return df.groupby('Symbol', group_keys=False).apply(_calc_indicators)

    @staticmethod
    def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates volume ratio and trends.
        """
        df['vol_ma20'] = df.groupby('Symbol')['Volume'].transform(lambda x: x.rolling(20).mean())
        df['volume_ratio'] = df['Volume'] / df['vol_ma20']
        
        # Volume trend (5-day slope approximation)
        df['volume_trend'] = df.groupby('Symbol')['Volume'].transform(lambda x: (x - x.shift(5)) / x.shift(5))
        
        # Unusual volume flag
        df['unusual_volume'] = (df['volume_ratio'] > 2.0).astype(int)
        
        return df

    @staticmethod
    def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates sector momentum and correlations.
        """
        # Pivot returns for cross-sectional calculation
        returns_pivot = df.pivot_table(index='Date', columns='Symbol', values='return_1d')
        
        # Metal Sector Momentum
        metals = ["TATASTEEL.NS", "NMDC.NS", "HINDALCO.NS", "VEDL.NS"]
        df_macro = pd.DataFrame(index=returns_pivot.index)
        df_macro['metals_momentum'] = returns_pivot[metals].mean(axis=1)
        
        # Benchmarks
        df_macro['nifty_momentum'] = returns_pivot['^NSEI']
        df_macro['gold_change'] = returns_pivot['GOLDBEES.NS']
        df_macro['silver_change'] = returns_pivot['SILVERBEES.NS']
        
        # Gold-Silver Ratio (Distance from mean)
        prices_pivot = df.pivot_table(index='Date', columns='Symbol', values='Close')
        df_macro['gold_silver_ratio'] = prices_pivot['GOLDBEES.NS'] / prices_pivot['SILVERBEES.NS']
        
        # Merge back
        df = df.join(df_macro, on='Date')
        return df

class MarketRegimeFeatures:
    """
    Categorical market state features.
    """
    @staticmethod
    def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
        # Volatility regime
        df['vol_percentile'] = df.groupby('Symbol')['atr_14'].transform(lambda x: x.rolling(100).rank(pct=True))
        df['volatility_regime'] = pd.cut(df['vol_percentile'], bins=[0, 0.33, 0.66, 1.0], labels=[0, 1, 2]).fillna(1).astype(int)
        
        # Trend regime (EMA alignment)
        df['trend_regime'] = 0 # Sideways
        df.loc[(df['Close'] > df['ema_50']) & (df['ema_50'] > df['ema_200']), 'trend_regime'] = 1 # Uptrend
        df.loc[(df['Close'] < df['ema_50']) & (df['ema_50'] < df['ema_200']), 'trend_regime'] = -1 # Downtrend
        
        return df

class NSEDataPipeline:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.fetcher = DataFetcher()
        self.cleaner = DataCleaner()
        self.engineer = FeatureEngineer()
        self.regime = MarketRegimeFeatures()
        
    def run(self, start_date: str = "2019-01-01"):
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        # 1. Fetch
        df = self.fetcher.fetch_stock_data(self.symbols, start_date, end_date)
        
        # 2. Clean
        df = self.cleaner.clean_data(df)
        
        # 3. Engineer
        df = self.engineer.add_price_features(df)
        df = self.engineer.add_technical_indicators(df)
        df = self.engineer.add_volume_features(df)
        df = self.engineer.add_macro_features(df)
        
        # 4. Regime
        df = self.regime.add_regime_features(df)
        
        # Replace infinity with NaN and then drop NaNs
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        # Save processed data
        processed_path = "data/processed_market_data.parquet"
        df.to_parquet(processed_path)
        logger.info(f"Pipeline complete. Processed data saved to {processed_path}")
        return df

if __name__ == "__main__":
    NSE_STOCKS = ["TATASTEEL.NS", "SUZLON.NS", "GOLDBEES.NS", "NMDC.NS", "YESBANK.NS", 
                  "TATAPOWER.NS", "SILVERBEES.NS", "HINDALCO.NS", "VEDL.NS"]
    
    pipeline = NSEDataPipeline(NSE_STOCKS)
    final_df = pipeline.run()
    
    print("\nFeature Engineering Complete!")
    print(f"Columns: {final_df.columns.tolist()}")
    print(f"Head:\n{final_df.tail(2)}")

"""
Market data handling for the statistical arbitrage strategy.
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import yfinance as yf
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MarketDataHandler:
    """
    Handler for fetching and managing market data.
    """
    
    def __init__(self, cache_dir=None):
        """
        Initialize the market data handler.
        
        Parameters:
        -----------
        cache_dir : str, optional
            Directory for caching data
        """
        self.cache_dir = cache_dir
        self.cache = {}
    
    def fetch_price_data(self, tickers, start_date, end_date, retry_count=3, retry_delay=10):
        """
        Fetch price data for the given tickers with robust error handling.
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        retry_count : int
            Number of retries for failed downloads
        retry_delay : int
            Delay in seconds between retries
            
        Returns:
        --------
        DataFrame with adjusted close prices
        """
        # Initialize an empty DataFrame with DatetimeIndex first
        result_df = pd.DataFrame()
        
        # Process one ticker at a time and merge into result_df
        for ticker in tqdm(tickers, desc="Downloading ticker data"):
            ticker_series = None
            
            for attempt in range(retry_count):
                try:
                    # First try with auto_adjust=True (newer API version)
                    ticker_data = yf.download(
                        ticker, 
                        start=start_date, 
                        end=end_date, 
                        auto_adjust=True, 
                        progress=False
                    )
                    
                    if ticker_data.empty:
                        logger.warning(f"{ticker}: Empty data returned")
                        break
                    
                    # For newer API versions, adjusted prices are in 'Close'
                    if 'Close' in ticker_data.columns:
                        ticker_series = ticker_data['Close']
                        logger.info(f"Successfully downloaded {ticker} data: {len(ticker_data)} rows")
                        break
                        
                    # If that doesn't work, try with auto_adjust=False (older API version)
                    ticker_data = yf.download(
                        ticker, 
                        start=start_date, 
                        end=end_date, 
                        auto_adjust=False, 
                        progress=False
                    )
                    
                    if not ticker_data.empty and 'Adj Close' in ticker_data.columns:
                        ticker_series = ticker_data['Adj Close']
                        logger.info(f"Successfully downloaded {ticker} data (using Adj Close): {len(ticker_data)} rows")
                        break
                            
                except Exception as e:
                    if attempt < retry_count - 1:
                        logger.warning(f"Error downloading {ticker}, attempt {attempt+1}/{retry_count}: {str(e)}")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"Failed to download {ticker} after {retry_count} attempts: {str(e)}")
            
            # Add this ticker to our result DataFrame if we got data
            if ticker_series is not None and not ticker_series.empty:
                # If this is our first successful ticker, initialize the DataFrame with its index
                if result_df.empty:
                    result_df = pd.DataFrame(index=ticker_series.index)
                
                # Add this ticker as a column
                result_df[ticker] = ticker_series
        
        # Check if we have any data
        if result_df.empty:
            logger.error("No data was successfully downloaded")
            # Return empty DataFrame with a default date range index
            return pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq='B'))
        
        # Drop any tickers with insufficient data
        min_data_points = int(0.9 * len(result_df))  # Require 90% data coverage
        valid_columns = []
        
        for col in result_df.columns:
            valid_data_count = result_df[col].dropna().shape[0]
            if valid_data_count >= min_data_points:
                valid_columns.append(col)
            else:
                logger.warning(f"Dropping {col} due to insufficient data: {valid_data_count}/{min_data_points} valid points")
        
        if not valid_columns:
            logger.error("No tickers with sufficient data after filtering")
            return pd.DataFrame(index=result_df.index)  # Keep the original index
        
        filtered_data = result_df[valid_columns]
        
        logger.info(f"Successfully fetched data for {len(valid_columns)}/{len(tickers)} tickers")
        logger.info(f"Date range: {filtered_data.index[0]} to {filtered_data.index[-1]}")
        logger.info(f"Data shape: {filtered_data.shape}")
        
        return filtered_data
    
    def fetch_historical_data(self, tickers, start_date, end_date, fields=None):
        """
        Fetch historical data for the given tickers, possibly including multiple fields.
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        fields : list, optional
            List of fields to fetch (e.g., ['Open', 'High', 'Low', 'Close', 'Volume'])
            
        Returns:
        --------
        Dictionary of DataFrames with historical data for each field
        """
        if fields is None:
            fields = ['Close']
        
        # Initialize dictionary to store results for each field
        results = {field: pd.DataFrame() for field in fields}
        
        # Process one ticker at a time
        for ticker in tqdm(tickers, desc="Downloading historical data"):
            try:
                # Download data
                ticker_data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False
                )
                
                if ticker_data.empty:
                    logger.warning(f"{ticker}: Empty data returned")
                    continue
                
                # Process each requested field
                for field in fields:
                    if field in ticker_data.columns:
                        if results[field].empty:
                            results[field] = pd.DataFrame(index=ticker_data.index)
                        
                        results[field][ticker] = ticker_data[field]
                    else:
                        logger.warning(f"Field {field} not found for {ticker}")
                
                logger.info(f"Successfully downloaded {ticker} historical data: {len(ticker_data)} rows")
                
            except Exception as e:
                logger.error(f"Failed to download historical data for {ticker}: {str(e)}")
        
        # Check if we have any data
        empty_fields = [field for field, df in results.items() if df.empty]
        for field in empty_fields:
            logger.error(f"No data downloaded for field: {field}")
        
        return results
    
    def fetch_volume_data(self, tickers, start_date, end_date):
        """
        Fetch volume data for the given tickers.
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
            
        Returns:
        --------
        DataFrame with volume data
        """
        # Fetch historical data with Volume field
        historical_data = self.fetch_historical_data(
            tickers, start_date, end_date, fields=['Volume']
        )
        
        return historical_data.get('Volume', pd.DataFrame())
    
    def calculate_rolling_metrics(self, price_data, windows=None):
        """
        Calculate various rolling metrics for price data.
        
        Parameters:
        -----------
        price_data : DataFrame
            DataFrame with price series
        windows : list, optional
            List of rolling windows to use
            
        Returns:
        --------
        Dictionary of DataFrames with rolling metrics
        """
        if windows is None:
            windows = [5, 20, 60]
        
        # Calculate returns
        returns = price_data.pct_change()
        
        # Initialize results dictionary
        metrics = {
            'returns': returns,
            'rolling_mean': {},
            'rolling_std': {},
            'rolling_sharpe': {},
            'rolling_drawdown': {}
        }
        
        # Calculate rolling metrics for each window
        for window in windows:
            # Rolling mean (annualized)
            rolling_mean = returns.rolling(window).mean() * 252
            metrics['rolling_mean'][window] = rolling_mean
            
            # Rolling volatility (annualized)
            rolling_std = returns.rolling(window).std() * np.sqrt(252)
            metrics['rolling_std'][window] = rolling_std
            
            # Rolling Sharpe ratio
            metrics['rolling_sharpe'][window] = rolling_mean / rolling_std
            
            # Rolling maximum drawdown
            rolling_cumret = (1 + returns).rolling(window).apply(
                lambda x: (1 + x).prod() - 1, raw=True
            )
            
            # This is a simplification - true drawdown would require a more complex calculation
            metrics['rolling_drawdown'][window] = rolling_cumret.rolling(window).apply(
                lambda x: (x[-1] / x.max()) - 1 if x.max() > 0 else 0, raw=True
            )
        
        return metrics
    
    def calculate_technical_indicators(self, price_data):
        """
        Calculate technical indicators for price data.
        
        Parameters:
        -----------
        price_data : DataFrame
            DataFrame with price series
            
        Returns:
        --------
        Dictionary of DataFrames with technical indicators
        """
        # Initialize dictionary for indicators
        indicators = {}
        
        # Moving Averages
        indicators['ma_50'] = price_data.rolling(50).mean()
        indicators['ma_200'] = price_data.rolling(200).mean()
        
        # Moving Average Convergence Divergence (MACD)
        ema_12 = price_data.ewm(span=12, adjust=False).mean()
        ema_26 = price_data.ewm(span=26, adjust=False).mean()
        indicators['macd'] = ema_12 - ema_26
        indicators['macd_signal'] = indicators['macd'].ewm(span=9, adjust=False).mean()
        
        # Relative Strength Index (RSI)
        returns = price_data.pct_change()
        up = returns.clip(lower=0)
        down = -returns.clip(upper=0)
        
        avg_up = up.rolling(14).mean()
        avg_down = down.rolling(14).mean()
        
        rs = avg_up / avg_down
        indicators['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        indicators['bb_middle'] = price_data.rolling(20).mean()
        indicators['bb_std'] = price_data.rolling(20).std()
        indicators['bb_upper'] = indicators['bb_middle'] + 2 * indicators['bb_std']
        indicators['bb_lower'] = indicators['bb_middle'] - 2 * indicators['bb_std']
        
        return indicators
    
    def fetch_all_data(self, tickers, start_date, end_date):
        """
        Fetch all relevant data for the given tickers.
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
            
        Returns:
        --------
        Dictionary with all data
        """
        # Fetch price data
        price_data = self.fetch_price_data(tickers, start_date, end_date)
        
        # Fetch volume data
        volume_data = self.fetch_volume_data(tickers, start_date, end_date)
        
        # Calculate rolling metrics
        rolling_metrics = self.calculate_rolling_metrics(price_data)
        
        # Calculate technical indicators
        technical_indicators = self.calculate_technical_indicators(price_data)
        
        return {
            'price_data': price_data,
            'volume_data': volume_data,
            'rolling_metrics': rolling_metrics,
            'technical_indicators': technical_indicators
        }
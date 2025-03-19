"""
Cointegration testing and analysis for pairs trading.
"""
import logging
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from tqdm import tqdm

logger = logging.getLogger(__name__)


def find_cointegrated_pairs(price_data, lookback=252, pvalue_threshold=0.05):
    """
    Find pairs of assets that are cointegrated.
    
    Parameters:
    -----------
    price_data : DataFrame
        DataFrame with price series
    lookback : int
        Lookback period for cointegration testing
    pvalue_threshold : float
        P-value threshold for cointegration testing
            
    Returns:
    --------
    List of cointegrated pairs and their p-values
    """
    if price_data.empty:
        logger.error("Empty price data provided to find_cointegrated_pairs")
        return []
            
    n = price_data.shape[1]
    pvalue_matrix = np.ones((n, n))
    keys = price_data.keys()
    pairs = []
    
    pair_count = n * (n - 1) // 2
    logger.info(f"Testing {pair_count} potential pairs for cointegration...")
    
    if pair_count == 0:
        logger.warning("No pairs to test (insufficient tickers)")
        return []
    
    for i in tqdm(range(n), desc="Testing pairs for cointegration"):
        for j in range(i+1, n):
            # Skip if either series has NaN values
            if price_data[keys[i]].isnull().any() or price_data[keys[j]].isnull().any():
                logger.debug(f"Skipping {keys[i]}-{keys[j]} due to NaN values")
                continue
                
            # Make sure we have enough data points
            valid_data_i = price_data[keys[i]].dropna()
            valid_data_j = price_data[keys[j]].dropna()
            
            if len(valid_data_i) < lookback or len(valid_data_j) < lookback:
                logger.debug(f"Skipping {keys[i]}-{keys[j]} due to insufficient data points")
                continue
            
            # Perform cointegration test on recent data
            try:
                recent_data = price_data.iloc[-lookback:]
                _, pvalue, _ = coint(recent_data[keys[i]], recent_data[keys[j]])
                pvalue_matrix[i, j] = pvalue
                
                if pvalue < pvalue_threshold:
                    pairs.append(((keys[i], keys[j]), pvalue))
                    logger.info(f"Found cointegrated pair: {keys[i]}-{keys[j]} with p-value {pvalue:.6f}")
            except Exception as e:
                logger.error(f"Error testing cointegration for {keys[i]}-{keys[j]}: {str(e)}")
    
    # Sort by p-value
    pairs.sort(key=lambda x: x[1])
    
    logger.info(f"Found {len(pairs)} cointegrated pairs")
    return pairs


def test_johansen_cointegration(price_data, pair, lookback=252):
    """
    Test for cointegration using Johansen's test.
    
    Parameters:
    -----------
    price_data : DataFrame
        DataFrame with price series
    pair : tuple
        Tuple containing ticker symbols for the pair
    lookback : int
        Lookback period for testing
        
    Returns:
    --------
    Tuple containing test statistic, critical value, and whether cointegrated
    """
    stock1, stock2 = pair
    
    # Ensure both tickers are in the data
    if stock1 not in price_data.columns or stock2 not in price_data.columns:
        logger.error(f"Pair {stock1}-{stock2} not found in price data")
        return None, None, False
    
    # Use recent data for test
    recent_data = price_data.iloc[-lookback:][[stock1, stock2]].dropna()
    
    # Check if we have enough data
    if len(recent_data) < lookback * 0.9:
        logger.warning(f"Insufficient data for Johansen test: {len(recent_data)}/{lookback}")
        return None, None, False
    
    try:
        # Perform Johansen test
        result = coint_johansen(recent_data, 0, 1)
        
        # Extract trace statistic and critical value (5%)
        trace_stat = result.lr1[0]
        critical_value = result.cvt[0, 1]
        
        # Determine if cointegrated
        is_cointegrated = trace_stat > critical_value
        
        return trace_stat, critical_value, is_cointegrated
    
    except Exception as e:
        logger.error(f"Error performing Johansen test for {pair}: {str(e)}")
        return None, None, False


def calculate_hedge_ratio(price_data, pair, window=60):
    """
    Calculate the hedge ratio between two assets using rolling OLS.
    
    Parameters:
    -----------
    price_data : DataFrame
        DataFrame with price series
    pair : tuple
        Tuple containing the ticker symbols of the pair
    window : int
        Window size for the rolling regression
            
    Returns:
    --------
    Series with the hedge ratios
    """
    stock1, stock2 = pair
    
    # Check if pair exists in data
    if stock1 not in price_data.columns or stock2 not in price_data.columns:
        logger.error(f"Pair {stock1}-{stock2} not found in price data")
        return None
    
    # Calculate rolling hedge ratio using OLS
    hedge_ratios = pd.Series(index=price_data.index)
    
    # Ensure we have sufficient data
    if len(price_data) < window:
        logger.warning(f"Insufficient data for hedge ratio calculation: {len(price_data)} < {window}")
        return None
    
    for i in range(window, len(price_data)):
        train = price_data.iloc[i-window:i]
        
        # Skip if we have NaN values
        if train[stock1].isnull().any() or train[stock2].isnull().any():
            continue
            
        X = train[stock1].values.reshape(-1, 1)
        y = train[stock2].values
        
        try:
            # OLS regression
            beta = np.linalg.lstsq(X, y, rcond=None)[0][0]
            hedge_ratios.iloc[i] = beta
        except Exception as e:
            logger.warning(f"Error calculating hedge ratio at index {i}: {str(e)}")
            continue
    
    # Check if we have sufficient valid hedge ratios
    valid_ratios = hedge_ratios.dropna()
    if len(valid_ratios) < window / 2:
        logger.warning(f"Too few valid hedge ratios: {len(valid_ratios)}")
        return None
        
    logger.info(f"Calculated hedge ratios for {pair}: mean={valid_ratios.mean():.4f}, std={valid_ratios.std():.4f}")
    return hedge_ratios.dropna()


def calculate_spread(price_data, pair, hedge_ratios=None):
    """
    Calculate the spread between two assets, optionally using dynamic hedge ratios.
    
    Parameters:
    -----------
    price_data : DataFrame
        DataFrame with price series
    pair : tuple
        Tuple containing the ticker symbols of the pair
    hedge_ratios : Series, optional
        Series with hedge ratios for each date
            
    Returns:
    --------
    Series with the spread
    """
    stock1, stock2 = pair
    
    # Check if pair exists in data
    if stock1 not in price_data.columns or stock2 not in price_data.columns:
        logger.error(f"Pair {stock1}-{stock2} not found in price data")
        return None
    
    try:
        if hedge_ratios is None:
            # If no hedge ratios provided, use full-sample OLS
            valid_mask = ~price_data[stock1].isnull() & ~price_data[stock2].isnull()
            X = price_data.loc[valid_mask, stock1].values.reshape(-1, 1)
            y = price_data.loc[valid_mask, stock2].values
            
            if len(X) < 30:  # Minimum required observations
                logger.warning(f"Insufficient valid observations for hedge ratio calculation: {len(X)}")
                return None
            
            beta = np.linalg.lstsq(X, y, rcond=None)[0][0]
            spread = price_data[stock2] - beta * price_data[stock1]
            logger.info(f"Calculated spread using static hedge ratio: {beta:.4f}")
        else:
            # Use dynamic hedge ratios - align indices
            common_idx = price_data.index.intersection(hedge_ratios.index)
            if len(common_idx) < 30:
                logger.warning(f"Insufficient overlapping observations: {len(common_idx)}")
                return None
                
            price_subset = price_data.loc[common_idx]
            hedge_subset = hedge_ratios.loc[common_idx]
            
            # Calculate spread
            spread = price_subset[stock2] - hedge_subset * price_subset[stock1]
            logger.info(f"Calculated spread using dynamic hedge ratios")
        
        # Check for stationarity
        adf_result = adfuller(spread.dropna())
        logger.info(f"ADF test for {stock1}-{stock2} spread: statistic={adf_result[0]:.4f}, p-value={adf_result[1]:.4f}")
        
        return spread
        
    except Exception as e:
        logger.error(f"Error calculating spread for {stock1}-{stock2}: {str(e)}")
        return None


def calculate_half_life(spread):
    """
    Calculate the half-life of mean reversion for a spread.
    
    Parameters:
    -----------
    spread : Series
        Series with spread values
            
    Returns:
    --------
    Float with half-life in days
    """
    spread = spread.dropna()
    
    if len(spread) < 30:
        logger.warning(f"Insufficient data for half-life calculation: {len(spread)}")
        return None
    
    # Calculate lagged spread and delta
    lagged_spread = spread.shift(1)
    delta = spread - lagged_spread
    
    # Remove NaN values
    mask = ~(delta.isnull() | lagged_spread.isnull())
    lagged_spread = lagged_spread[mask]
    delta = delta[mask]
    
    # Perform regression
    X = lagged_spread.values.reshape(-1, 1)
    y = delta.values
    
    try:
        # OLS regression
        beta = np.linalg.lstsq(X, y, rcond=None)[0][0]
        
        # Calculate half-life
        if beta < 0:
            half_life = -np.log(2) / beta
            logger.info(f"Calculated half-life: {half_life:.4f} days")
            return half_life
        else:
            logger.warning(f"Non-mean-reverting series (beta={beta:.4f})")
            return None
    except Exception as e:
        logger.error(f"Error calculating half-life: {str(e)}")
        return None
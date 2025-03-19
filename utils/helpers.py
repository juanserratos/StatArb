"""
Helper functions for the statistical arbitrage strategy.
"""
import logging
import pandas as pd
import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)

def expand_universe(tickers, add_etfs=True, add_sectors=True):
    """
    Expand the universe of tickers to include related ETFs and sector indices.
    
    Parameters:
    -----------
    tickers : list
        List of ticker symbols
    add_etfs : bool
        Whether to add related ETFs
    add_sectors : bool
        Whether to add sector ETFs
        
    Returns:
    --------
    List of expanded tickers
    """
    expanded = set(tickers)
    
    # Add common ETFs
    if add_etfs:
        common_etfs = [
            'SPY', 'QQQ', 'IWM', 'DIA',  # Major indices
            'TLT', 'IEF', 'SHY',         # Treasury bonds
            'GLD', 'SLV', 'USO',         # Commodities
            'UUP', 'FXE', 'FXY',         # Currencies
            'VXX'                         # Volatility
        ]
        expanded.update(common_etfs)
    
    # Add sector ETFs
    if add_sectors:
        sector_etfs = [
            'XLF', 'XLK', 'XLV', 'XLI', 'XLP', 'XLY', 'XLE', 'XLU', 'XLB', 'XLRE',
            'KBE', 'KRE', 'XHB', 'XRT', 'XME', 'XOP', 'XBI', 'XPH'
        ]
        expanded.update(sector_etfs)
    
    logger.info(f"Expanded universe from {len(tickers)} to {len(expanded)} tickers")
    return list(expanded)

def filter_universe(price_data, min_price=5.0, min_volume=100000, min_history=252):
    """
    Filter universe based on liquidity, price, and data availability.
    
    Parameters:
    -----------
    price_data : DataFrame
        DataFrame with price data
    min_price : float
        Minimum price filter
    min_volume : float
        Minimum daily volume (dummy check - actual volume data not used here)
    min_history : int
        Minimum number of historical data points required
        
    Returns:
    --------
    List of filtered tickers
    """
    if price_data.empty:
        logger.error("Empty price data provided to filter_universe")
        return []
    
    filtered = []
    
    for ticker in price_data.columns:
        price_series = price_data[ticker].dropna()
        
        # Check for sufficient history
        if len(price_series) < min_history:
            logger.debug(f"Filtering out {ticker}: insufficient history ({len(price_series)}/{min_history})")
            continue
        
        # Check for minimum price
        if price_series.iloc[-1] < min_price:
            logger.debug(f"Filtering out {ticker}: price too low (${price_series.iloc[-1]:.2f}/${min_price:.2f})")
            continue
        
        # Add to filtered list
        filtered.append(ticker)
    
    logger.info(f"Filtered universe from {len(price_data.columns)} to {len(filtered)} tickers")
    return filtered

def calculate_half_life(spread):
    """
    Calculate half-life of mean reversion for a price series.
    
    Parameters:
    -----------
    spread : Series
        Price series to analyze
        
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
            logger.debug(f"Calculated half-life: {half_life:.4f} days")
            return half_life
        else:
            logger.warning(f"Non-mean-reverting series (beta={beta:.4f})")
            return None
    except Exception as e:
        logger.error(f"Error calculating half-life: {str(e)}")
        return None

def rank_pairs_by_criteria(pairs_data, weights=None):
    """
    Rank pairs based on multiple criteria with weights.
    
    Parameters:
    -----------
    pairs_data : list
        List of dictionaries with pair data
    weights : dict, optional
        Dictionary with criteria weights
        
    Returns:
    --------
    List of ranked pairs
    """
    if not pairs_data:
        return []
    
    # Default weights
    if weights is None:
        weights = {
            'pvalue': 0.3,          # Lower is better
            'half_life': 0.25,      # Mid-range is better (10-40 days)
            'correlation': 0.15,    # Negative is better
            'liquidity': 0.15,      # Higher is better
            'beta_neutral': 0.15    # Higher is better
        }
    
    # Calculate scores for each criterion
    for pair_data in pairs_data:
        score = 0
        
        # Cointegration p-value (lower is better)
        if 'pvalue' in pair_data:
            score += (1 - pair_data['pvalue']) * weights.get('pvalue', 0)
        
        # Half-life (mid-range is optimal)
        if 'half_life' in pair_data:
            half_life = pair_data['half_life']
            # Score peaks at 20 days
            half_life_score = 1 - min(abs(half_life - 20) / 30, 1)
            score += half_life_score * weights.get('half_life', 0)
        
        # Correlation (negative is better)
        if 'correlation' in pair_data:
            corr = pair_data['correlation']
            corr_score = min((-corr + 1) / 2, 1) if corr < 0 else 0
            score += corr_score * weights.get('correlation', 0)
        
        # Liquidity (higher is better)
        if 'liquidity' in pair_data:
            liq = min(pair_data['liquidity'] / 1e7, 1)
            score += liq * weights.get('liquidity', 0)
        
        # Beta neutrality (closer to 1 is better)
        if 'beta_neutral_score' in pair_data:
            score += pair_data['beta_neutral_score'] * weights.get('beta_neutral', 0)
        
        pair_data['score'] = score
    
    # Sort by score (higher is better)
    ranked_pairs = sorted(pairs_data, key=lambda x: x.get('score', 0), reverse=True)
    
    return ranked_pairs
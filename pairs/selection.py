"""
Enhanced pair selection methods for the statistical arbitrage strategy.
"""
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from tqdm import tqdm

from pairs.cointegration import test_johansen_cointegration, calculate_spread, calculate_half_life

logger = logging.getLogger(__name__)


def find_enhanced_pairs(price_data, lookback=252, min_half_life=5, max_half_life=100,
                       pvalue_threshold=0.05, correlation_filter=True, 
                       min_correlation=-0.5, cointegration_method='johansen',
                       sector_map=None, macro_data=None, beta_neutral=True):
    """
    Find pairs with enhanced selection criteria including half-life,
    correlation filters, and sector-based analysis.
    
    Parameters:
    -----------
    price_data : DataFrame
        DataFrame with price series
    lookback : int
        Lookback period for statistical tests
    min_half_life : int
        Minimum acceptable half-life for mean reversion in days
    max_half_life : int
        Maximum acceptable half-life for mean reversion in days
    pvalue_threshold : float
        P-value threshold for cointegration testing
    correlation_filter : bool
        Whether to filter pairs based on correlation
    min_correlation : float
        Minimum correlation threshold for pair selection
    cointegration_method : str
        Method for cointegration testing: 'engle_granger' or 'johansen'
    sector_map : dict, optional
        Dictionary mapping tickers to sectors
    macro_data : DataFrame, optional
        DataFrame with market data for beta calculation
    beta_neutral : bool
        Whether to prefer beta-neutral pairs
            
    Returns:
    --------
    List of selected pairs with additional metrics
    """
    if price_data.empty:
        logger.error("Empty price data provided")
        return []
            
    n = price_data.shape[1]
    keys = price_data.keys()
    all_pairs = []
    
    # Add counters for debugging
    total_pairs = n * (n - 1) // 2
    half_life_count = 0
    coint_count = 0
    corr_count = 0
    beta_count = 0
    stationary_count = 0
    
    pair_count = n * (n - 1) // 2
    logger.info(f"Testing {pair_count} potential pairs with enhanced criteria...")
    
    if pair_count == 0:
        logger.warning("No pairs to test (insufficient tickers)")
        return []
    
    # If beta neutrality is enabled and market data is provided
    market_betas = None
    if beta_neutral and macro_data is not None:
        if 'SPY' in macro_data.columns:
            market_returns = macro_data['SPY'].pct_change().dropna()
            market_betas = {}
            
            for ticker in keys:
                if ticker in price_data.columns:
                    ticker_returns = price_data[ticker].pct_change().dropna()
                    
                    # Align returns
                    common_idx = ticker_returns.index.intersection(market_returns.index)
                    if len(common_idx) > 60:  # Need sufficient data
                        aligned_ticker = ticker_returns.loc[common_idx]
                        aligned_market = market_returns.loc[common_idx]
                        
                        # Calculate beta using regression
                        X = aligned_market.values.reshape(-1, 1)
                        y = aligned_ticker.values
                        beta = np.linalg.lstsq(X, y, rcond=None)[0][0]
                        market_betas[ticker] = beta
    
    for i in tqdm(range(n), desc="Testing pairs with enhanced criteria"):
        for j in range(i+1, n):
            # Skip if either series has NaN values
            if price_data[keys[i]].isnull().any() or price_data[keys[j]].isnull().any():
                continue
                
            # Make sure we have enough data points
            valid_data_i = price_data[keys[i]].dropna()
            valid_data_j = price_data[keys[j]].dropna()
            
            if len(valid_data_i) < lookback or len(valid_data_j) < lookback:
                continue
            
            # Perform analysis
            try:
                recent_data = price_data.iloc[-lookback:]
                
                # Calculate correlation
                correlation = recent_data[keys[i]].corr(recent_data[keys[j]])
                
                # Check correlation filter
                pass_correlation = True
                if correlation_filter and correlation > min_correlation:
                    pass_correlation = False
                
                if pass_correlation or not correlation_filter:
                    corr_count += 1
                
                # Cointegration test
                is_cointegrated = False
                pvalue = 1.0
                
                if cointegration_method == 'johansen':
                    # Johansen test (more powerful for multiple time series)
                    trace_stat, critical_value, is_cointegrated = test_johansen_cointegration(
                        recent_data, (keys[i], keys[j]), lookback
                    )
                    
                    # Approximate p-value (johansen test doesn't return one directly)
                    if is_cointegrated:
                        pvalue = 0.01  # Strong evidence
                    else:
                        pvalue = 0.1   # Weak evidence
                else:
                    # Engle-Granger test
                    _, pvalue, _ = coint(recent_data[keys[i]], recent_data[keys[j]])
                    is_cointegrated = pvalue < pvalue_threshold
                
                if is_cointegrated:
                    coint_count += 1
                
                if not is_cointegrated:
                    continue
                
                # Calculate hedge ratio and spread
                X = recent_data[keys[i]].values.reshape(-1, 1)
                y = recent_data[keys[j]].values
                beta = np.linalg.lstsq(X, y, rcond=None)[0][0]
                
                spread = recent_data[keys[j]] - beta * recent_data[keys[i]]
                
                # Calculate half-life of mean reversion
                half_life = calculate_half_life(spread)
                
                if half_life is not None:
                    half_life_count += 1
                
                # Skip pairs with half-life outside desired range
                if half_life is None or half_life < min_half_life or half_life > max_half_life:
                    continue
                
                # ADF test for stationarity
                adf_result = adfuller(spread.dropna())
                adf_pvalue = adf_result[1]
                is_stationary = adf_pvalue < 0.05
                
                if is_stationary:
                    stationary_count += 1
                
                # Calculate volatility ratio
                vol_i = recent_data[keys[i]].pct_change().std()
                vol_j = recent_data[keys[j]].pct_change().std()
                vol_ratio = min(vol_i / vol_j, vol_j / vol_i)  # Between 0 and 1
                
                # Beta neutrality check if enabled
                beta_neutral_score = 1.0
                pass_beta_neutral = True
                
                if beta_neutral and market_betas is not None:
                    if keys[i] in market_betas and keys[j] in market_betas:
                        beta_i = market_betas[keys[i]]
                        beta_j = market_betas[keys[j]]
                        
                        # Calculate dollar-neutral position sizes
                        price_i = recent_data[keys[i]].iloc[-1]
                        price_j = recent_data[keys[j]].iloc[-1]
                        
                        # For $1 notional in each asset
                        notional_i = 1.0
                        notional_j = beta * notional_i
                        
                        # Calculate beta exposure
                        beta_exposure_i = notional_i * beta_i
                        beta_exposure_j = notional_j * beta_j
                        net_beta_exposure = abs(beta_exposure_i - beta_exposure_j)
                        
                        # Score based on beta neutrality (lower is better)
                        beta_neutral_score = 1.0 / (1.0 + net_beta_exposure)
                        
                        # Apply threshold for beta neutrality (arbitrary threshold of 0.3)
                        if beta_neutral_score < 0.3:
                            pass_beta_neutral = False
                    else:
                        pass_beta_neutral = False
                
                if pass_beta_neutral or not beta_neutral:
                    beta_count += 1
                
                # Sector relationship check
                same_sector = False
                sector_bonus = 1.0
                
                if sector_map is not None:
                    if (keys[i] in sector_map and keys[j] in sector_map and
                        sector_map[keys[i]].get('sector') == sector_map[keys[j]].get('sector')):
                        same_sector = True
                        sector_bonus = 1.5  # Bonus for same-sector pairs
                
                # Calculate an overall score (lower is better)
                # Only add pairs that pass all criteria
                if is_cointegrated and is_stationary and (pass_correlation or not correlation_filter) and (pass_beta_neutral or not beta_neutral):
                    # Factors: p-value, half-life, volatility ratio, correlation, beta neutrality
                    half_life_score = min(1.0, abs(half_life - 30) / 30)  # Prefer half-life around 30 days
                    vol_score = 1.0 - vol_ratio  # Prefer similar volatility
                    corr_score = min(1.0, abs(correlation + 0.7) / 0.7) if correlation_filter else 0.5  # Prefer correlation around -0.7
                    
                    # Combined score (lower is better)
                    score = (
                        pvalue * 0.3 +
                        half_life_score * 0.2 +
                        vol_score * 0.2 +
                        corr_score * 0.1 +
                        (1.0 - beta_neutral_score) * 0.2
                    ) / sector_bonus
                    
                    all_pairs.append({
                        'pair': (keys[i], keys[j]),
                        'score': score,
                        'pvalue': pvalue,
                        'half_life': half_life,
                        'correlation': correlation,
                        'hedge_ratio': beta,
                        'vol_ratio': vol_ratio,
                        'adf_pvalue': adf_pvalue,
                        'same_sector': same_sector,
                        'beta_neutral_score': beta_neutral_score
                    })
            except Exception as e:
                logger.debug(f"Error testing pair {keys[i]}-{keys[j]}: {str(e)}")
                continue
    
    # Sort by score (lower is better)
    all_pairs.sort(key=lambda x: x['score'])
    
    logger.info(f"Found {len(all_pairs)} valid pairs after enhanced filtering")
    
    # Add more debug info
    if len(all_pairs) == 0:
        logger.warning("No pairs passed all criteria. Debug information:")
        logger.warning(f"- Total pairs tested: {total_pairs}")
        logger.warning(f"- Pairs with valid half-life ({min_half_life}-{max_half_life} days): {half_life_count}")
        logger.warning(f"- Pairs with cointegration p-value < {pvalue_threshold}: {coint_count}")
        logger.warning(f"- Pairs with stationarity: {stationary_count}")
        if correlation_filter:
            logger.warning(f"- Pairs with correlation < {min_correlation}: {corr_count}")
        if beta_neutral:
            logger.warning(f"- Pairs with beta neutrality: {beta_count}")
    
    # Log top pairs
    for i, p in enumerate(all_pairs[:10]):
        logger.info(f"Pair {i+1}: {p['pair'][0]}-{p['pair'][1]}, " +
                   f"Score: {p['score']:.4f}, " +
                   f"Half-life: {p['half_life']:.1f}, " +
                   f"Correlation: {p['correlation']:.3f}, " +
                   f"Same Sector: {p['same_sector']}")
    
    # Return pairs and their data
    return all_pairs


def get_sector_mapping(tickers):
    """
    Creates a mapping of tickers to their sectors for better pair formation.
    
    Parameters:
    -----------
    tickers : list
        List of ticker symbols
        
    Returns:
    --------
    Dictionary mapping tickers to sectors
    """
    sector_map = {}
    
    for ticker in tqdm(tickers, desc="Getting sector information"):
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            
            sector_map[ticker] = {
                'sector': sector,
                'industry': industry
            }
            
        except Exception as e:
            logger.warning(f"Error getting sector data for {ticker}: {str(e)}")
            sector_map[ticker] = {
                'sector': 'Unknown',
                'industry': 'Unknown'
            }
    
    # Print sector distribution
    sector_counts = {}
    for ticker, info in sector_map.items():
        sector = info.get('sector', 'Unknown')
        if sector in sector_counts:
            sector_counts[sector] += 1
        else:
            sector_counts[sector] = 1
    
    logger.info("Sector distribution:")
    for sector, count in sorted(sector_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"{sector}: {count} tickers")
    
    return sector_map


def filter_pairs_by_liquidity(pairs, price_data, min_adv=1000000, max_spread_bps=20):
    """
    Filter pairs based on liquidity constraints.
    
    Parameters:
    -----------
    pairs : list
        List of pairs to filter
    price_data : DataFrame
        DataFrame with price data
    min_adv : float
        Minimum average daily volume in dollars
    max_spread_bps : float
        Maximum bid-ask spread in basis points
        
    Returns:
    --------
    List of filtered pairs
    """
    filtered_pairs = []
    
    # Get volume data for tickers
    all_tickers = set()
    for pair in pairs:
        all_tickers.add(pair[0][0])
        all_tickers.add(pair[0][1])
    
    # Fetch volume data
    volume_data = {}
    for ticker in tqdm(all_tickers, desc="Fetching volume data"):
        try:
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(period="1mo")
            
            if not hist.empty:
                # Calculate average daily volume in dollars
                avg_volume = hist['Volume'].mean()
                avg_price = hist['Close'].mean()
                adv = avg_volume * avg_price
                
                # Get latest bid-ask spread if available
                spread_bps = None
                
                # Store data
                volume_data[ticker] = {
                    'adv': adv,
                    'spread_bps': spread_bps
                }
        except Exception as e:
            logger.warning(f"Error fetching volume data for {ticker}: {str(e)}")
    
    # Filter pairs based on liquidity
    for pair_info in pairs:
        pair = pair_info[0]
        
        if pair[0] in volume_data and pair[1] in volume_data:
            adv1 = volume_data[pair[0]]['adv']
            adv2 = volume_data[pair[1]]['adv']
            
            # Check if both assets meet minimum ADV
            if adv1 >= min_adv and adv2 >= min_adv:
                # Check spread if available
                spread1 = volume_data[pair[0]]['spread_bps']
                spread2 = volume_data[pair[1]]['spread_bps']
                
                if (spread1 is None or spread1 <= max_spread_bps) and \
                   (spread2 is None or spread2 <= max_spread_bps):
                    filtered_pairs.append(pair_info)
        
    logger.info(f"Filtered {len(pairs)} pairs to {len(filtered_pairs)} based on liquidity")
    return filtered_pairs
"""
Macroeconomic indicators for market regime detection.
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)


def fetch_enhanced_macro_data(start_date, end_date, lookback_period=252, retry_count=3, 
                           retry_delay=10, market_data_handler=None):
    """
    Fetch an expanded set of macroeconomic indicators for better regime detection.
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    lookback_period : int
        Period used for rolling calculations
    retry_count : int
        Number of retries for failed downloads
    retry_delay : int
        Delay in seconds between retries
    market_data_handler : MarketDataHandler, optional
        Market data handler instance
            
    Returns:
    --------
    DataFrame with enhanced macroeconomic indicators
    """
    # Add buffer for lookback
    adj_start_date = (datetime.strptime(start_date, '%Y-%m-%d') - 
                     timedelta(days=int(lookback_period * 1.5))).strftime('%Y-%m-%d')
    
    # Expanded list of indicators to fetch
    indicators = {
        # Volatility indices
        '^VIX': 'VIX',                # S&P 500 volatility
        '^VVIX': 'VVIX',              # VIX volatility (volatility of volatility)
        
        # Interest rates and spreads
        '^TNX': 'TNX',                # 10-year Treasury yield
        '^TYX': 'TYX',                # 30-year Treasury yield
        '^IRX': 'IRX',                # 13-week Treasury bill yield
        '^FVX': 'FVX',                # 5-year Treasury yield
        
        # Credit spreads via ETFs
        'LQD': 'LQD',                 # Investment grade corporate bonds
        'HYG': 'HYG',                 # High yield corporate bonds
        
        # Equity indices
        'SPY': 'SPY',                 # S&P 500
        'QQQ': 'QQQ',                 # Nasdaq 100
        'IWM': 'IWM',                 # Russell 2000
        
        # Fixed income
        'TLT': 'TLT',                 # 20+ year Treasury bonds
        'IEF': 'IEF',                 # 7-10 year Treasury bonds
        'SHY': 'SHY',                 # 1-3 year Treasury bonds
        
        # Currencies
        'DX-Y.NYB': 'USD_Index',      # US Dollar index
        'UUP': 'UUP',                 # US Dollar Bullish ETF
        
        # Commodities
        'GLD': 'GLD',                 # Gold
        'USO': 'USO',                 # Oil
        
        # Risk sentiment indicators
        'BIL': 'BIL',                 # T-Bills (cash equivalent)
    }
    
    logger.info(f"Fetching enhanced macroeconomic data from {adj_start_date} to {end_date}")
    
    if market_data_handler is None:
        from data.market_data import MarketDataHandler
        market_data_handler = MarketDataHandler()
    
    # Use the market data handler to fetch the data
    macro_df = market_data_handler.fetch_price_data(
        list(indicators.keys()), adj_start_date, end_date, retry_count, retry_delay
    )
    
    # Rename columns according to the indicators mapping
    macro_df = macro_df.rename(columns=indicators)
    
    # Filter to requested date range
    if not macro_df.empty:
        mask = (macro_df.index >= start_date) & (macro_df.index <= end_date)
        macro_df = macro_df.loc[mask]
    
    # Calculate additional derived features
    try:
        # Market features
        if 'SPY' in macro_df.columns:
            # Multiple timeframe volatility
            for window in [5, 10, 20, 60]:
                macro_df[f'SPY_Vol_{window}d'] = macro_df['SPY'].pct_change().rolling(window).std() * np.sqrt(252)
            
            # Moving average crossovers (trend indicators)
            macro_df['SPY_MA_10'] = macro_df['SPY'].rolling(10).mean()
            macro_df['SPY_MA_50'] = macro_df['SPY'].rolling(50).mean()
            macro_df['SPY_MA_Cross'] = (macro_df['SPY_MA_10'] > macro_df['SPY_MA_50']).astype(int)
            
            # Rate of change indicators
            for window in [5, 20, 60]:
                macro_df[f'SPY_ROC_{window}d'] = macro_df['SPY'].pct_change(window)
        
        # Volatility features
        if 'VIX' in macro_df.columns:
            # VIX momentum and extremes
            macro_df['VIX_5d_ROC'] = macro_df['VIX'].pct_change(5)
            macro_df['VIX_Percentile'] = macro_df['VIX'].rolling(252).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan)
            
            # VIX term structure (if we have VVIX)
            if 'VVIX' in macro_df.columns:
                macro_df['VIX_VVIX_Ratio'] = macro_df['VIX'] / macro_df['VVIX']
        
        # Interest rate features
        if 'TNX' in macro_df.columns and 'IRX' in macro_df.columns:
            # Yield curve slope (10y - 3m)
            macro_df['Yield_Curve_Slope'] = macro_df['TNX'] - macro_df['IRX']
            
            # Changes at different horizons
            for window in [5, 20, 60]:
                macro_df[f'TNX_Change_{window}d'] = macro_df['TNX'].diff(window)
        
        # Credit spread features
        if 'LQD' in macro_df.columns and 'HYG' in macro_df.columns:
            # Simple credit spread indicator
            macro_df['Credit_Spread'] = macro_df['LQD'] / macro_df['HYG']
            macro_df['Credit_Spread_ROC'] = macro_df['Credit_Spread'].pct_change(10)
            
        # Dollar strength features
        if 'USD_Index' in macro_df.columns:
            macro_df['USD_20d_ROC'] = macro_df['USD_Index'].pct_change(20)
            macro_df['USD_Trend'] = (macro_df['USD_Index'].rolling(10).mean() > 
                                    macro_df['USD_Index'].rolling(50).mean()).astype(int)
            
        logger.info(f"Calculated enhanced macro features, shape: {macro_df.shape}")
            
    except Exception as e:
        logger.warning(f"Error calculating enhanced macro features: {str(e)}")
    
    return macro_df.dropna(how='all')


def detect_market_regime(macro_data, lookback=60):
    """
    Detect the current market regime to adapt strategy parameters.
    
    Parameters:
    -----------
    macro_data : DataFrame
        DataFrame with market indicators
    lookback : int
        Lookback period for regime detection
            
    Returns:
    --------
    String indicating the current market regime
    """
    if macro_data is None or macro_data.empty:
        logger.warning("No market data for regime detection")
        return 'bull_low_vol'  # Default regime
    
    if 'SPY' not in macro_data.columns or 'VIX' not in macro_data.columns:
        logger.warning("Missing required indicators (SPY, VIX) for market regime detection")
        return 'bull_low_vol'  # Default regime
    
    # Get recent data
    recent_data = macro_data.iloc[-lookback:]
    
    # Calculate market trend
    spy_price = recent_data['SPY']
    spy_ma50 = spy_price.rolling(50).mean()
    is_bull = spy_price.iloc[-1] > spy_ma50.iloc[-1]
    
    # Calculate volatility regime
    vix = recent_data['VIX']
    vix_ma20 = vix.rolling(20).mean()
    is_high_vol = vix.iloc[-1] > vix_ma20.iloc[-1]
    
    # Determine regime
    if is_bull:
        regime = 'bull_high_vol' if is_high_vol else 'bull_low_vol'
    else:
        regime = 'bear_high_vol' if is_high_vol else 'bear_low_vol'
    
    logger.info(f"Detected market regime: {regime}")
    return regime


def calculate_regime_features(macro_data):
    """
    Calculate features for regime classification.
    
    Parameters:
    -----------
    macro_data : DataFrame
        DataFrame with market indicators
            
    Returns:
    --------
    DataFrame with regime features
    """
    if macro_data is None or macro_data.empty:
        logger.warning("No market data for feature calculation")
        return None
    
    # Initialize feature DataFrame
    features = pd.DataFrame(index=macro_data.index)
    
    try:
        # Market trend features
        if 'SPY' in macro_data.columns:
            # Trend indicators
            features['SPY_Trend_50d'] = (macro_data['SPY'] > 
                                         macro_data['SPY'].rolling(50).mean()).astype(int)
            
            features['SPY_Trend_200d'] = (macro_data['SPY'] > 
                                          macro_data['SPY'].rolling(200).mean()).astype(int)
            
            # Relative strength
            features['SPY_RS_20d'] = macro_data['SPY'].pct_change(20)
            features['SPY_RS_60d'] = macro_data['SPY'].pct_change(60)
        
        # Volatility features
        if 'VIX' in macro_data.columns:
            features['VIX_Level'] = macro_data['VIX']
            features['VIX_Trend'] = (macro_data['VIX'] > 
                                     macro_data['VIX'].rolling(20).mean()).astype(int)
            
            # Volatility regime
            features['VIX_Regime_High'] = (macro_data['VIX'] > 25).astype(int)
            features['VIX_Regime_Low'] = (macro_data['VIX'] < 15).astype(int)
        
        # Yield curve features
        if 'Yield_Curve_Slope' in macro_data.columns:
            features['Yield_Curve_Positive'] = (macro_data['Yield_Curve_Slope'] > 0).astype(int)
            features['Yield_Curve_Steepening'] = (macro_data['Yield_Curve_Slope'].diff(10) > 0).astype(int)
        
        # Credit spread features
        if 'Credit_Spread' in macro_data.columns:
            features['Credit_Widening'] = (macro_data['Credit_Spread'].diff(10) > 0).astype(int)
            features['Credit_Extreme'] = (macro_data['Credit_Spread'] > 
                                          macro_data['Credit_Spread'].rolling(252).mean() + 
                                          2 * macro_data['Credit_Spread'].rolling(252).std()).astype(int)
        
        logger.info(f"Calculated {len(features.columns)} regime features")
        return features
        
    except Exception as e:
        logger.error(f"Error calculating regime features: {str(e)}")
        return None


def classify_regime_state(features, method='rule_based', n_regimes=4):
    """
    Classify the market regime state based on features.
    
    Parameters:
    -----------
    features : DataFrame
        DataFrame with regime features
    method : str
        Classification method: 'rule_based' or 'clustering'
    n_regimes : int
        Number of regimes for clustering method
            
    Returns:
    --------
    Series with regime classifications
    """
    if features is None or features.empty:
        logger.warning("No features for regime classification")
        return None
    
    # Initialize regime series
    regimes = pd.Series(index=features.index, dtype='object')
    
    try:
        if method == 'rule_based':
            # Rule-based classification
            for idx in features.index:
                # Extract key features
                trend_50d = features.loc[idx, 'SPY_Trend_50d'] if 'SPY_Trend_50d' in features.columns else 1
                vix_high = features.loc[idx, 'VIX_Regime_High'] if 'VIX_Regime_High' in features.columns else 0
                vix_low = features.loc[idx, 'VIX_Regime_Low'] if 'VIX_Regime_Low' in features.columns else 1
                credit_widening = features.loc[idx, 'Credit_Widening'] if 'Credit_Widening' in features.columns else 0
                
                # Determine regime
                if trend_50d == 1:  # Bull market
                    if vix_high == 1:
                        regimes.loc[idx] = 'bull_high_vol'
                    else:
                        regimes.loc[idx] = 'bull_low_vol'
                else:  # Bear market
                    if vix_high == 1 or credit_widening == 1:
                        regimes.loc[idx] = 'bear_high_vol'
                    else:
                        regimes.loc[idx] = 'bear_low_vol'
                
        elif method == 'clustering':
            # Use clustering for regime classification
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # Prepare data for clustering
            cluster_data = features.fillna(0)
            
            # Normalize features
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)
            
            # Apply KMeans clustering
            kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_data)
            
            # Assign cluster labels to original data
            for i, idx in enumerate(features.index):
                regimes.loc[idx] = f'regime_{cluster_labels[i]}'
            
            # Map clusters to interpretable labels
            if n_regimes == 4:
                # Calculate average feature values per cluster
                for cluster in range(n_regimes):
                    cluster_mask = cluster_labels == cluster
                    cluster_features = scaled_data[cluster_mask]
                    
                    # Skip if empty cluster
                    if len(cluster_features) == 0:
                        continue
                    
                    # Calculate average features
                    avg_features = np.mean(cluster_features, axis=0)
                    
                    # Determine if bull/bear based on trend features
                    trend_idx = [i for i, col in enumerate(cluster_data.columns) if 'Trend' in col]
                    avg_trend = np.mean(avg_features[trend_idx]) if trend_idx else 0
                    
                    # Determine if high/low vol based on volatility features
                    vol_idx = [i for i, col in enumerate(cluster_data.columns) if 'VIX' in col]
                    avg_vol = np.mean(avg_features[vol_idx]) if vol_idx else 0
                    
                    # Assign semantic label
                    if avg_trend > 0:  # Bull market
                        if avg_vol > 0:
                            regime_name = 'bull_high_vol'
                        else:
                            regime_name = 'bull_low_vol'
                    else:  # Bear market
                        if avg_vol > 0:
                            regime_name = 'bear_high_vol'
                        else:
                            regime_name = 'bear_low_vol'
                    
                    # Replace cluster numbers with names
                    regimes = regimes.replace(f'regime_{cluster}', regime_name)
        
        else:
            logger.error(f"Unknown classification method: {method}")
            return None
        
        # Check regime distribution
        regime_counts = regimes.value_counts()
        for regime, count in regime_counts.items():
            logger.info(f"{regime}: {count} observations ({count/len(regimes):.2%})")
        
        return regimes
        
    except Exception as e:
        logger.error(f"Error classifying regimes: {str(e)}")
        return None
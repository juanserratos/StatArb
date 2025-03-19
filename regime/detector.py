"""
Regime detection functionality for the statistical arbitrage strategy.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm

from regime.hmm_model import fit_hmm_model

logger = logging.getLogger(__name__)


def detect_regimes(spread, macro_data=None, n_regimes=2):
    """
    Fit a Hidden Markov Model to detect regimes in the spread.
    
    Parameters:
    -----------
    spread : Series
        Series with the spread
    macro_data : DataFrame, optional
        DataFrame with macroeconomic indicators
    n_regimes : int
        Number of regimes to detect
            
    Returns:
    --------
    Tuple with the model, hidden states, and scaler
    """
    if spread is None or spread.dropna().empty:
        logger.error("Empty spread provided to detect_regimes")
        return None, None, None, None
            
    # Prepare data for HMM
    try:
        if macro_data is not None and not macro_data.empty:
            # Align macro data with spread
            aligned_data = pd.concat([spread.to_frame('spread'), macro_data], axis=1)
            aligned_data = aligned_data.dropna(how='any')
            
            if aligned_data.empty:
                logger.warning("No overlapping data between spread and macro data")
                X = spread.dropna().values.reshape(-1, 1)
            else:
                # Select features based on availability
                features = ['spread']
                for col in ['VIX', 'SPY_Vol_20d', 'TNX_Change_20d', 'VIX_Vol_Gap']:
                    if col in aligned_data.columns:
                        features.append(col)
                
                X = aligned_data[features].values
                logger.info(f"Using features for regime detection: {features}")
        else:
            X = spread.dropna().values.reshape(-1, 1)
            logger.info("Using only spread for regime detection (no macro data)")
        
        # Check if we have enough data
        if len(X) < 30:
            logger.warning(f"Insufficient data for regime detection: {len(X)}")
            return None, None, None, None
        
        # Normalize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit HMM with multiple attempts
        best_model = None
        best_score = -np.inf
        max_attempts = 5
        
        for attempt in range(max_attempts):
            try:
                model = hmm.GaussianHMM(
                    n_components=n_regimes, 
                    covariance_type="full", 
                    n_iter=1000,
                    random_state=42 + attempt
                )
                
                model.fit(X_scaled)
                score = model.score(X_scaled)
                
                if best_model is None or score > best_score:
                    best_model = model
                    best_score = score
                    
            except Exception as e:
                logger.warning(f"HMM fitting attempt {attempt+1} failed: {str(e)}")
        
        if best_model is None:
            logger.error("All HMM fitting attempts failed")
            return None, None, None, None
            
        # Decode the states
        hidden_states = best_model.predict(X_scaled)
        
        # Log regime statistics
        regime_counts = np.bincount(hidden_states)
        regime_props = regime_counts / len(hidden_states)
        regime_stats = {i: {"count": c, "proportion": p} for i, (c, p) in enumerate(zip(regime_counts, regime_props))}
        logger.info(f"Regime statistics: {regime_stats}")
        
        return best_model, hidden_states, scaler, X_scaled
        
    except Exception as e:
        logger.error(f"Error detecting regimes: {str(e)}")
        return None, None, None, None


def detect_complex_regimes(spread, macro_data=None, n_regimes=None):
    """
    Enhanced regime detection using more advanced techniques and features.
    
    Parameters:
    -----------
    spread : Series
        Series with the spread
    macro_data : DataFrame, optional
        DataFrame with macroeconomic indicators
    n_regimes : int, optional
        Number of regimes, defaults to self.n_regimes
            
    Returns:
    --------
    Tuple with the model, hidden states, regime labels, and features
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    if spread is None or spread.dropna().empty:
        logger.error("Empty spread provided to detect_complex_regimes")
        return None, None, None, None
    
    if n_regimes is None:
        n_regimes = 2
    
    # Prepare data
    try:
        # Create base features from spread
        spread_features = pd.DataFrame({
            'spread': spread,
            'spread_ma5': spread.rolling(5).mean(),
            'spread_ma20': spread.rolling(20).mean(),
            'spread_vol5': spread.rolling(5).std(),
            'spread_vol20': spread.rolling(20).std(),
            'spread_z5': (spread - spread.rolling(5).mean()) / spread.rolling(5).std(),
            'spread_z20': (spread - spread.rolling(20).mean()) / spread.rolling(20).std(),
            'spread_mom5': spread.pct_change(5),
            'spread_mom20': spread.pct_change(20),
        })
        
        # Rate of mean reversion (half-life)
        spread_df = pd.DataFrame({'spread': spread})
        spread_df['lag_spread'] = spread_df['spread'].shift(1)
        spread_df['delta'] = spread_df['spread'] - spread_df['lag_spread']
        
        # Estimate mean reversion coefficient
        if len(spread_df.dropna()) > 30:  # Need sufficient data
            try:
                X = spread_df['lag_spread'].dropna().values.reshape(-1, 1)
                y = spread_df['delta'].dropna().values
                beta = np.linalg.lstsq(X, y, rcond=None)[0][0]
                half_life = -np.log(2) / beta if beta < 0 else np.nan
                spread_features['half_life'] = half_life
            except:
                pass
        
        # If we have macro data, add selected macro features and align dates
        feature_df = spread_features.copy()
        if macro_data is not None and not macro_data.empty:
            # Select key macro features
            key_macro_features = [
                'VIX', 'SPY_Vol_20d', 'Yield_Curve_Slope', 'Credit_Spread',
                'SPY_MA_Cross', 'USD_Trend', 'SPY_ROC_20d', 'TNX_Change_20d'
            ]
            
            available_features = [f for f in key_macro_features if f in macro_data.columns]
            
            if available_features:
                # Align dates with spread features
                macro_subset = macro_data[available_features]
                feature_df = pd.concat([feature_df, macro_subset], axis=1)
        
        # Drop rows with NaN values
        feature_df = feature_df.dropna()
        
        if len(feature_df) < 30:
            logger.warning(f"Insufficient data for regime detection: {len(feature_df)} rows")
            return None, None, None, None
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feature_df)
        
        # Apply PCA if we have many features to reduce dimensionality
        if X_scaled.shape[1] > 5:
            # Keep enough components to explain 95% of variance
            pca = PCA(n_components=0.95)
            X_pca = pca.fit_transform(X_scaled)
            logger.info(f"Reduced features from {X_scaled.shape[1]} to {X_pca.shape[1]} using PCA")
            X_model = X_pca
        else:
            X_model = X_scaled
        
        # Fit HMM model
        model, hidden_states, best_cov_type = fit_hmm_model(X_model, n_regimes)
        
        if model is None:
            logger.error("HMM model fitting failed")
            return None, None, None, None
        
        logger.info(f"Best HMM model: {n_regimes} regimes with {best_cov_type} covariance")
        
        # Create regime labels based on mean spread in each regime
        regime_spread_means = {}
        regime_volatilities = {}
        regime_labels = {}
        
        for state in range(n_regimes):
            mask = hidden_states == state
            regime_spread = feature_df.loc[mask, 'spread']
            regime_spread_means[state] = regime_spread.mean()
            regime_volatilities[state] = regime_spread.std()
        
        # Sort regimes by their mean spread
        sorted_states = sorted(range(n_regimes), key=lambda x: regime_spread_means[x])
        
        # Assign descriptive labels
        labels = {
            2: ['Low', 'High'],  # For 2 regimes
            3: ['Low', 'Medium', 'High'],  # For 3 regimes
            4: ['Very Low', 'Low', 'High', 'Very High']  # For 4 regimes
        }
        
        if n_regimes in labels:
            for i, state in enumerate(sorted_states):
                regime_labels[state] = labels[n_regimes][i]
        else:
            # Generic labeling
            for i, state in enumerate(sorted_states):
                regime_labels[state] = f"Regime {i+1}"
        
        # Log regime statistics
        for state in range(n_regimes):
            mask = hidden_states == state
            proportion = np.mean(mask)
            logger.info(f"{regime_labels[state]} ({state}): " +
                       f"mean={regime_spread_means[state]:.4f}, " +
                       f"std={regime_volatilities[state]:.4f}, " +
                       f"proportion={proportion:.2f}")
        
        return model, hidden_states, regime_labels, feature_df
        
    except Exception as e:
        logger.error(f"Error in complex regime detection: {str(e)}")
        return None, None, None, None


def train_regime_specific_parameters(spread, hidden_states, regime_labels=None):
    """
    Train regime-specific z-score thresholds and other parameters.
    
    Parameters:
    -----------
    spread : Series
        Series with the spread
    hidden_states : array
        Array with the hidden states
    regime_labels : dict, optional
        Dictionary mapping states to descriptive labels
            
    Returns:
    --------
    Dictionary with regime-specific parameters
    """
    if spread is None or hidden_states is None:
        logger.error("Spread or hidden states is None")
        return None
    
    if len(spread) != len(hidden_states):
        logger.error(f"Length mismatch: spread ({len(spread)}) vs hidden_states ({len(hidden_states)})")
        return None
    
    # Create spread df with regime info
    spread_df = pd.DataFrame({
        'spread': spread.values,
        'regime': hidden_states
    }, index=spread.index)
    
    regime_params = {}
    unique_regimes = np.unique(hidden_states)
    
    for regime in unique_regimes:
        regime_mask = spread_df['regime'] == regime
        regime_spread = spread_df.loc[regime_mask, 'spread']
        
        if len(regime_spread) < 20:  # Need sufficient data
            logger.warning(f"Insufficient data points for regime {regime}: {len(regime_spread)}")
            continue
        
        # Basic statistics
        mean = regime_spread.mean()
        std = regime_spread.std()
        
        # Calculate optimal z-score thresholds based on historical reversion
        z_scores = (regime_spread - mean) / std
        
        # Find points of reversal
        # This is a simple approach - find when z-score crosses certain thresholds and
        # calculate subsequent returns
        z_thresholds = np.arange(1.0, 3.1, 0.2)  # Test thresholds from 1.0 to 3.0
        
        best_entry_z = 2.0  # Default
        best_metric = 0
        
        for z_thresh in z_thresholds:
            # Long entries (z-score < -z_thresh)
            long_entries = z_scores < -z_thresh
            
            # Short entries (z-score > z_thresh)
            short_entries = z_scores > z_thresh
            
            # If we have entries, measure forward returns
            if np.sum(long_entries) > 5 and np.sum(short_entries) > 5:
                # Calculate forward mean reversion (5-day forward return for simplicity)
                # For longs, positive return = good, for shorts, negative return = good
                forward_returns = z_scores.shift(-5) - z_scores
                
                long_perf = forward_returns.loc[long_entries].mean()
                short_perf = -forward_returns.loc[short_entries].mean()
                
                # Combined metric (higher is better)
                metric = (long_perf + short_perf) / 2
                
                if metric > best_metric:
                    best_metric = metric
                    best_entry_z = z_thresh
        
        # Optimal exit is typically around 0 (mean reversion)
        # but we can allow some room for transaction costs
        best_exit_z = min(best_entry_z * 0.3, 0.5)
        
        regime_params[regime] = {
            'z_entry': best_entry_z,
            'z_exit': best_exit_z,
            'mean': mean,
            'std': std
        }
        
        # Use regime label if available
        if regime_labels and regime in regime_labels:
            name = regime_labels[regime]
            logger.info(f"Regime {name} ({regime}) parameters: " +
                       f"z_entry={best_entry_z:.2f}, z_exit={best_exit_z:.2f}")
        else:
            logger.info(f"Regime {regime} parameters: " +
                       f"z_entry={best_entry_z:.2f}, z_exit={best_exit_z:.2f}")
    
    return regime_params
"""
Signal generation for the statistical arbitrage strategy.
"""
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def generate_signals(spread, hidden_states, z_entry=2.0, z_exit=0.5, holding_period=20):
    """
    Generate trading signals based on the spread and regime state.
    
    Parameters:
    -----------
    spread : Series
        Series with the spread
    hidden_states : array
        Array with the hidden states
    z_entry : float
        Z-score threshold for entering positions
    z_exit : float
        Z-score threshold for exiting positions
    holding_period : int
        Maximum holding period in days
            
    Returns:
    --------
    Series with trading signals (1: long spread, -1: short spread, 0: no position)
    """
    if spread is None or hidden_states is None:
        logger.error("Spread or hidden states is None")
        return None
        
    if len(spread) != len(hidden_states):
        logger.error(f"Length mismatch: spread ({len(spread)}) vs hidden_states ({len(hidden_states)})")
        return None
    
    # Create spread df with regime info
    spread_with_states = pd.DataFrame({
        'spread': spread.values,
        'state': hidden_states
    }, index=spread.index)
    
    # Calculate z-score for each regime separately
    z_scores = pd.Series(index=spread.index, dtype=float)
    
    for state in np.unique(hidden_states):
        state_mask = spread_with_states['state'] == state
        if sum(state_mask) > 30:  # Ensure enough data points
            state_data = spread_with_states[state_mask]
            state_mean = state_data['spread'].mean()
            state_std = state_data['spread'].std()
            
            # Avoid division by zero
            if state_std > 0:
                # Calculate z-scores for this state
                z_scores[state_mask] = (spread[state_mask] - state_mean) / state_std
                logger.info(f"Regime {state}: mean={state_mean:.4f}, std={state_std:.4f}, n={sum(state_mask)}")
            else:
                logger.warning(f"Zero standard deviation for regime {state}")
        else:
            logger.warning(f"Insufficient data points for regime {state}: {sum(state_mask)}")
    
    # Generate signals
    signals = pd.Series(index=spread.index, data=0)
    
    # Only trade when we have valid z-scores
    valid_mask = ~z_scores.isna()
    
    # Short when spread is too high (z-score > entry_z)
    signals[valid_mask & (z_scores > z_entry)] = -1
    
    # Long when spread is too low (z-score < -entry_z)
    signals[valid_mask & (z_scores < -z_entry)] = 1
    
    # Exit positions when spread reverts (|z-score| < exit_z)
    exit_mask = valid_mask & (z_scores.abs() < z_exit)
    signals[exit_mask & (signals.shift(1) != 0)] = 0
    
    # Implement holding period constraint
    position_duration = pd.Series(0, index=signals.index)
    current_position = 0
    duration = 0
    
    for i in range(len(signals)):
        if signals.iloc[i] != current_position:
            # Position changed
            current_position = signals.iloc[i]
            duration = 0
        else:
            # Position unchanged
            duration += 1
            
        position_duration.iloc[i] = duration
        
        # Force exit if holding period exceeded
        if duration >= holding_period and current_position != 0:
            signals.iloc[i] = 0
            current_position = 0
            duration = 0
    
    # Log signal statistics
    signal_counts = signals.value_counts()
    logger.info(f"Generated signals: {signal_counts.to_dict()}")
    
    return signals


def generate_enhanced_signals(spread, regime_data, macro_data=None, volatility_scaling=True):
    """
    Generate trading signals with regime-specific parameters.
    
    Parameters:
    -----------
    spread : Series
        Series with the spread
    regime_data : tuple
        Tuple containing (model, hidden_states, regime_labels, feature_df)
    macro_data : DataFrame, optional
        DataFrame with macroeconomic indicators
    volatility_scaling : bool
        Whether to scale position sizes by volatility
            
    Returns:
    --------
    DataFrame with trading signals and position sizing
    """
    if spread is None or regime_data is None:
        logger.error("Spread or regime data is None")
        return None
    
    model, hidden_states, regime_labels, feature_df = regime_data
    
    if model is None or hidden_states is None:
        logger.error("Model or hidden states is None")
        return None
    
    # Align indices
    spread_idx = spread.index
    regime_idx = pd.Series(hidden_states, index=feature_df.index)
    
    # Make sure these are aligned
    common_idx = spread_idx.intersection(regime_idx.index)
    if len(common_idx) < 30:
        logger.error(f"Insufficient aligned data points between spread and regimes: {len(common_idx)}")
        return None
    
    # Create DataFrame for signals
    signal_df = pd.DataFrame(index=common_idx)
    signal_df['spread'] = spread.loc[common_idx]
    signal_df['regime'] = regime_idx.loc[common_idx]
    
    # Calculate z-scores for each regime
    signal_df['z_score'] = np.nan
    signal_df['signal'] = 0
    signal_df['position_size'] = 0.0
    
    # Train regime-specific parameters
    from regime.detector import train_regime_specific_parameters
    regime_params = train_regime_specific_parameters(
        signal_df['spread'], signal_df['regime'].values, regime_labels
    )
    
    if regime_params is None:
        logger.error("Failed to train regime parameters")
        return None
    
    # Calculate volatility for scaling if requested
    if volatility_scaling:
        # Calculate rolling volatility of the spread
        signal_df['volatility'] = signal_df['spread'].rolling(20).std()
        
        # Calculate volatility ratio for scaling (relative to historical average)
        historical_vol = signal_df['volatility'].mean()
        signal_df['vol_ratio'] = historical_vol / signal_df['volatility']
        
        # Cap the ratio to avoid extreme position sizes
        signal_df['vol_ratio'] = signal_df['vol_ratio'].clip(0.5, 2.0)
    else:
        signal_df['vol_ratio'] = 1.0
    
    # Apply regime-specific parameters
    for regime, params in regime_params.items():
        regime_mask = signal_df['regime'] == regime
        
        # Skip if no data for this regime
        if not any(regime_mask):
            continue
        
        # Calculate z-score for this regime
        signal_df.loc[regime_mask, 'z_score'] = (
            (signal_df.loc[regime_mask, 'spread'] - params['mean']) / params['std']
        )
        
        # Entry signals
        signal_df.loc[regime_mask & (signal_df['z_score'] > params['z_entry']), 'signal'] = -1
        signal_df.loc[regime_mask & (signal_df['z_score'] < -params['z_entry']), 'signal'] = 1
        
        # Exit signals - override entry signals
        signal_df.loc[regime_mask & (signal_df['z_score'].abs() < params['z_exit']), 'signal'] = 0
    
    # Implement position management logic
    for i in range(1, len(signal_df)):
        prev_signal = signal_df['signal'].iloc[i-1]
        curr_signal = signal_df['signal'].iloc[i]
        
        # Handle signal state transitions
        if prev_signal != 0 and curr_signal == 0:
            # Exit
            pass  # Signal is already 0
        elif prev_signal == 0 and curr_signal != 0:
            # New entry
            pass  # Signal is already set
        elif prev_signal != 0 and curr_signal != 0 and prev_signal != curr_signal:
            # Reversing position
            pass  # Signal is already changed
        elif prev_signal != 0 and curr_signal == prev_signal:
            # Maintain position
            pass
    
    # Calculate position sizes using volatility scaling
    signal_df['position_size'] = signal_df['signal'] * signal_df['vol_ratio']
    
    return signal_df


def filter_signals(signals, lookback=10, consistency_threshold=0.7, min_holding_days=3):
    """
    Filter signals to reduce noise and whipsaws.
    
    Parameters:
    -----------
    signals : Series
        Series with trading signals
    lookback : int
        Lookback period for filtering
    consistency_threshold : float
        Threshold for signal consistency
    min_holding_days : int
        Minimum holding period in days
            
    Returns:
    --------
    Series with filtered signals
    """
    if signals is None:
        logger.error("Signals is None")
        return None
    
    # Create a copy of the signals
    filtered_signals = signals.copy()
    
    # Track the current position and holding days
    current_position = 0
    holding_days = 0
    
    for i in range(len(filtered_signals)):
        # Get current signal
        current_signal = filtered_signals.iloc[i]
        
        # Check position change
        if current_signal != current_position:
            # Check if we've held the current position for minimum days
            if current_position != 0 and holding_days < min_holding_days:
                # Don't change position yet
                filtered_signals.iloc[i] = current_position
            else:
                # Check consistency of the new signal
                if i >= lookback:
                    # Get recent signals
                    recent_signals = filtered_signals.iloc[i-lookback:i]
                    
                    # Count occurrences of the new signal
                    signal_count = (recent_signals == current_signal).sum()
                    
                    # Calculate consistency
                    consistency = signal_count / lookback
                    
                    # If not consistent enough, maintain current position
                    if consistency < consistency_threshold:
                        filtered_signals.iloc[i] = current_position
                    else:
                        # Update position
                        current_position = current_signal
                        holding_days = 0
                else:
                    # Not enough history, accept the new position
                    current_position = current_signal
                    holding_days = 0
        else:
            # Position unchanged, increment holding days
            holding_days += 1
    
    # Log filtering statistics
    original_changes = (signals.diff() != 0).sum()
    filtered_changes = (filtered_signals.diff() != 0).sum()
    
    logger.info(f"Signal filtering: {original_changes} original changes, " +
               f"{filtered_changes} filtered changes, " +
               f"{(original_changes - filtered_changes) / original_changes:.2%} reduction")
    
    return filtered_signals


def calculate_kelly_position_size(spread, signals, lookback=60, max_size=1.0, 
                                target_proba=0.55, fractional_kelly=0.5):
    """
    Calculate position sizes using the Kelly criterion.
    
    Parameters:
    -----------
    spread : Series
        Series with the spread
    signals : Series
        Series with trading signals
    lookback : int
        Lookback period for estimating win rate and return
    max_size : float
        Maximum position size (1.0 = full)
    target_proba : float
        Target win probability for full sizing
    fractional_kelly : float
        Fraction of Kelly to use
            
    Returns:
    --------
    Series with position sizes
    """
    if spread is None or signals is None:
        logger.error("Spread or signals is None")
        return None
    
    # Align indices
    common_idx = spread.index.intersection(signals.index)
    spread = spread.loc[common_idx]
    signals = signals.loc[common_idx]
    
    # Calculate spread returns
    spread_returns = spread.pct_change()
    
    # Create DataFrame with signals and returns
    df = pd.DataFrame({
        'signal': signals,
        'spread_return': spread_returns
    })
    
    # Initialize position size series
    position_sizes = pd.Series(0.0, index=signals.index)
    
    # Calculate position sizes based on rolling statistics
    for i in range(lookback, len(df)):
        # Get lookback window
        window = df.iloc[i-lookback:i]
        
        # Separate long and short signals
        long_signals = window[window['signal'] == 1]
        short_signals = window[window['signal'] == -1]
        
        if len(long_signals) > 5 and len(short_signals) > 5:
            # Calculate win rate for each signal type
            long_win_rate = (long_signals['spread_return'] < 0).mean()  # For long signals, we want spread to decrease
            short_win_rate = (short_signals['spread_return'] > 0).mean()  # For short signals, we want spread to increase
            
            # Calculate average return for each signal type
            long_avg_return = -long_signals['spread_return'].mean()  # Negative of spread return
            short_avg_return = short_signals['spread_return'].mean()
            
            # Calculate Kelly fraction for the current signal
            current_signal = df['signal'].iloc[i]
            
            if current_signal == 1:  # Long spread
                win_rate = long_win_rate
                avg_win = long_avg_return if long_avg_return > 0 else 0.01
                avg_loss = -long_avg_return if long_avg_return < 0 else 0.01
            elif current_signal == -1:  # Short spread
                win_rate = short_win_rate
                avg_win = short_avg_return if short_avg_return > 0 else 0.01
                avg_loss = -short_avg_return if short_avg_return < 0 else 0.01
            else:  # No position
                position_sizes.iloc[i] = 0
                continue
            
            # Kelly formula: f* = (p*b - q) / b, where p = win_rate, q = (1-p), b = win/loss ratio
            win_loss_ratio = avg_win / avg_loss
            kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
            
            # Apply fractional Kelly
            kelly = kelly * fractional_kelly
            
            # Scale by win rate relative to target
            win_rate_scale = min(win_rate / target_proba, 1.0) if target_proba > 0 else 1.0
            
            # Calculate position size, cap at max_size
            position_size = min(kelly * win_rate_scale, max_size)
            
            # Make sure position size is not negative
            position_size = max(position_size, 0)
            
            # Apply position size with sign of signal
            position_sizes.iloc[i] = position_size * current_signal
        else:
            # Not enough history for this signal type, use signal directly
            position_sizes.iloc[i] = df['signal'].iloc[i] * 0.5  # Use half size
    
    logger.info(f"Calculated Kelly position sizes with average abs size: {position_sizes.abs().mean():.4f}")
    
    return position_sizes
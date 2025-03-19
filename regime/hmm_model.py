"""
Hidden Markov Model implementation for regime detection.
"""
import logging
import numpy as np
from hmmlearn import hmm

logger = logging.getLogger(__name__)


def fit_hmm_model(X, n_regimes=2, max_attempts=5):
    """
    Fit a Hidden Markov Model to the data with multiple attempts and covariance types.
    
    Parameters:
    -----------
    X : ndarray
        Input data, shape (n_samples, n_features)
    n_regimes : int
        Number of regimes to detect
    max_attempts : int
        Maximum number of fitting attempts per covariance type
            
    Returns:
    --------
    Tuple with the model, hidden states, and best covariance type
    """
    # Try different covariance types
    covariance_types = ['full', 'diag', 'tied']
    best_model = None
    best_score = -np.inf
    best_cov_type = None
    
    for cov_type in covariance_types:
        for attempt in range(max_attempts):
            try:
                model = hmm.GaussianHMM(
                    n_components=n_regimes,
                    covariance_type=cov_type,
                    n_iter=1000,
                    random_state=42 + attempt
                )
                
                model.fit(X)
                score = model.score(X)
                
                if best_model is None or score > best_score:
                    best_model = model
                    best_score = score
                    best_cov_type = cov_type
                    
            except Exception as e:
                logger.debug(f"HMM fitting failed with {cov_type} covariance, attempt {attempt+1}: {str(e)}")
    
    if best_model is None:
        logger.error("All HMM fitting attempts failed")
        return None, None, None
    
    # Predict hidden states
    hidden_states = best_model.predict(X)
    
    return best_model, hidden_states, best_cov_type


def analyze_regime_characteristics(X, hidden_states, feature_names=None):
    """
    Analyze the characteristics of each regime.
    
    Parameters:
    -----------
    X : ndarray
        Input data, shape (n_samples, n_features)
    hidden_states : ndarray
        Array of hidden states
    feature_names : list, optional
        Names of features for better logging
            
    Returns:
    --------
    Dictionary with regime characteristics
    """
    if X is None or hidden_states is None:
        logger.error("X or hidden_states is None")
        return None
    
    if len(X) != len(hidden_states):
        logger.error(f"Length mismatch: X ({len(X)}) vs hidden_states ({len(hidden_states)})")
        return None
    
    # Get unique regimes
    unique_regimes = np.unique(hidden_states)
    n_regimes = len(unique_regimes)
    n_features = X.shape[1]
    
    # If feature names not provided, create generic ones
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(n_features)]
    
    # Analyze each regime
    regime_stats = {}
    
    for regime in unique_regimes:
        # Get data for this regime
        regime_mask = hidden_states == regime
        regime_data = X[regime_mask]
        
        # Calculate basic statistics
        regime_mean = np.mean(regime_data, axis=0)
        regime_std = np.std(regime_data, axis=0)
        regime_count = len(regime_data)
        regime_prop = regime_count / len(X)
        
        # Store statistics
        regime_stats[regime] = {
            'count': regime_count,
            'proportion': regime_prop,
            'means': {feature_names[i]: regime_mean[i] for i in range(n_features)},
            'stds': {feature_names[i]: regime_std[i] for i in range(n_features)}
        }
        
        # Log regime statistics
        logger.info(f"Regime {regime}: {regime_count} samples ({regime_prop:.2%} of data)")
        for i in range(min(n_features, 5)):  # Log first 5 features
            logger.info(f"  {feature_names[i]}: mean={regime_mean[i]:.4f}, std={regime_std[i]:.4f}")
    
    # Analyze transitions
    transitions = []
    for i in range(1, len(hidden_states)):
        if hidden_states[i] != hidden_states[i-1]:
            transitions.append((hidden_states[i-1], hidden_states[i]))
    
    # Count transitions
    transition_counts = {}
    for from_state, to_state in transitions:
        key = (from_state, to_state)
        if key in transition_counts:
            transition_counts[key] += 1
        else:
            transition_counts[key] = 1
    
    # Log transitions
    for (from_state, to_state), count in transition_counts.items():
        logger.info(f"Transitions {from_state} -> {to_state}: {count}")
    
    # Calculate regime persistence
    persistence = {}
    
    for regime in unique_regimes:
        # Find runs of this regime
        in_run = False
        run_lengths = []
        current_run = 0
        
        for state in hidden_states:
            if state == regime:
                in_run = True
                current_run += 1
            elif in_run:
                run_lengths.append(current_run)
                in_run = False
                current_run = 0
        
        # Add last run if in progress
        if in_run:
            run_lengths.append(current_run)
        
        # Calculate persistence statistics
        if run_lengths:
            avg_persistence = np.mean(run_lengths)
            max_persistence = np.max(run_lengths)
            persistence[regime] = {
                'avg_persistence': avg_persistence,
                'max_persistence': max_persistence,
                'runs': len(run_lengths)
            }
            
            logger.info(f"Regime {regime} persistence: avg={avg_persistence:.1f} days, max={max_persistence} days")
    
    return {
        'regime_stats': regime_stats,
        'transitions': transition_counts,
        'persistence': persistence
    }


def detect_regime_change_probability(model, X_recent, n_forecasts=5):
    """
    Detect the probability of regime changes in the near future.
    
    Parameters:
    -----------
    model : GaussianHMM
        Fitted HMM model
    X_recent : ndarray
        Recent data points, shape (n_samples, n_features)
    n_forecasts : int
        Number of steps to forecast ahead
            
    Returns:
    --------
    Dictionary with forecast probabilities
    """
    if model is None or X_recent is None:
        logger.error("Model or X_recent is None")
        return None
    
    try:
        # Get current regime probabilities
        current_probs = model.predict_proba(X_recent[-1:])
        
        # Extract transition matrix
        trans_mat = model.transmat_
        
        # Calculate future probabilities
        future_probs = current_probs.copy()
        probs_over_time = [current_probs[0]]
        
        for i in range(n_forecasts):
            future_probs = np.dot(future_probs, trans_mat)
            probs_over_time.append(future_probs[0])
        
        # Convert to array for easier analysis
        probs_array = np.array(probs_over_time)
        
        # Calculate regime change probability
        current_regime = np.argmax(current_probs[0])
        regime_change_prob = 1 - probs_array[1:, current_regime]
        
        logger.info(f"Current regime: {current_regime}, change probabilities: {regime_change_prob}")
        
        return {
            'current_regime': current_regime,
            'current_probs': current_probs[0],
            'future_probs': probs_array[1:],
            'regime_change_prob': regime_change_prob
        }
    
    except Exception as e:
        logger.error(f"Error calculating regime change probabilities: {str(e)}")
        return None
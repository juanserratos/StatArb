"""
Portfolio allocation strategies for the statistical arbitrage strategy.
"""
import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def calculate_optimal_allocations(backtests, method='risk_parity', 
                             lookback_window=60, target_vol=0.1):
    """
    Calculate optimal allocations across pairs based on various allocation methods.
    
    Parameters:
    -----------
    backtests : dict
        Dictionary with backtest results for each pair
    method : str
        Allocation method: 'equal_weight', 'risk_parity', 'min_variance', 'max_sharpe'
    lookback_window : int
        Window for estimating covariance
    target_vol : float
        Target portfolio volatility (annualized)
        
    Returns:
    --------
    Dictionary with allocation weights for each pair
    """
    # Prepare returns data
    pair_returns = {}
    common_dates = None
    
    for pair_str, results in backtests.items():
        if pair_str == 'aggregate' or pair_str.startswith('error') or pair_str == 'portfolio':
            continue
            
        if 'returns' in results:
            returns_series = results['returns']
            
            if common_dates is None:
                common_dates = returns_series.index
            else:
                common_dates = common_dates.intersection(returns_series.index)
                
            pair_returns[pair_str] = returns_series
    
    if not pair_returns or common_dates is None or len(common_dates) == 0:
        logger.error("No valid return data for allocation")
        return None
    
    # Align all return series to common dates
    aligned_returns = pd.DataFrame({
        pair: returns.reindex(common_dates) for pair, returns in pair_returns.items()
    })
    
    # Fill any remaining NaNs with 0
    aligned_returns = aligned_returns.fillna(0)
    
    # Use recent data for allocation decisions
    if len(aligned_returns) > lookback_window:
        recent_returns = aligned_returns.iloc[-lookback_window:]
    else:
        recent_returns = aligned_returns
    
    # Calculate allocation weights based on the chosen method
    n_pairs = len(pair_returns)
    
    if method == 'equal_weight':
        # Simple equal weight allocation
        weights = {pair: 1.0 / n_pairs for pair in pair_returns.keys()}
        
    elif method == 'risk_parity':
        # Risk parity allocation
        cov_matrix = recent_returns.cov() * 252  # Annualized covariance
        
        # Calculate volatility (risk) for each strategy
        vols = np.sqrt(np.diag(cov_matrix))
        
        # Risk parity weights are inverse of volatility (normalized)
        raw_weights = 1.0 / vols
        weights = {pair: w / sum(raw_weights) for pair, w in zip(pair_returns.keys(), raw_weights)}
        
    elif method == 'min_variance':
        # Minimum variance optimization
        cov_matrix = recent_returns.cov() * 252  # Annualized covariance
        
        def portfolio_vol(w):
            # Convert w to array if it's not already
            w_arr = np.array(w)
            return np.sqrt(w_arr.T @ cov_matrix.values @ w_arr)
        
        # Constraints: weights sum to 1, and each weight >= 0
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},)
        bounds = tuple((0.0, 1.0) for _ in range(n_pairs))
        
        # Initial guess: equal weights
        initial_weights = np.array([1.0 / n_pairs] * n_pairs)
        
        # Optimize to minimize portfolio volatility
        result = minimize(portfolio_vol, initial_weights, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            opt_weights = result.x
            weights = {pair: weight for pair, weight in zip(pair_returns.keys(), opt_weights)}
        else:
            logger.warning(f"Optimization failed: {result.message}")
            weights = {pair: 1.0 / n_pairs for pair in pair_returns.keys()}
        
    elif method == 'max_sharpe':
        # Maximum Sharpe ratio optimization
        cov_matrix = recent_returns.cov() * 252  # Annualized covariance
        expected_returns = recent_returns.mean() * 252  # Annualized returns
        
        def neg_sharpe_ratio(w):
            w_arr = np.array(w)
            port_return = np.sum(w_arr * expected_returns.values)
            port_vol = np.sqrt(w_arr.T @ cov_matrix.values @ w_arr)
            return -port_return / port_vol if port_vol > 0 else 0
        
        # Constraints: weights sum to 1, and each weight >= 0
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},)
        bounds = tuple((0.0, 1.0) for _ in range(n_pairs))
        
        # Initial guess: equal weights
        initial_weights = np.array([1.0 / n_pairs] * n_pairs)
        
        # Optimize to maximize Sharpe ratio
        result = minimize(neg_sharpe_ratio, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            opt_weights = result.x
            weights = {pair: weight for pair, weight in zip(pair_returns.keys(), opt_weights)}
        else:
            logger.warning(f"Optimization failed: {result.message}")
            weights = {pair: 1.0 / n_pairs for pair in pair_returns.keys()}
    
    else:
        logger.warning(f"Unknown allocation method: {method}, using equal weight")
        weights = {pair: 1.0 / n_pairs for pair in pair_returns.keys()}
    
    # Scale weights to target volatility
    if target_vol is not None:
        try:
            portfolio_vol = np.sqrt(
                sum(weights[p1] * weights[p2] * cov_matrix.loc[p1, p2]
                    for p1 in weights for p2 in weights)
            )
            
            scaling_factor = target_vol / portfolio_vol if portfolio_vol > 0 else 1.0
            weights = {pair: w * scaling_factor for pair, w in weights.items()}
            
            logger.info(f"Scaled weights to target vol {target_vol:.2%}, scaling factor: {scaling_factor:.2f}")
        except Exception as e:
            logger.warning(f"Error scaling weights to target vol: {str(e)}")
    
    # Log allocation results
    logger.info(f"Allocation method: {method}")
    for pair, weight in sorted(weights.items(), key=lambda x: -x[1]):
        logger.info(f"  {pair}: {weight:.4f}")
    
    return weights


def hierarchical_risk_parity(returns, risk_measure='variance'):
    """
    Implement hierarchical risk parity for more robust allocation.
    
    Parameters:
    -----------
    returns : DataFrame
        DataFrame with strategy returns
    risk_measure : str
        Risk measure to use: 'variance', 'mad' (mean absolute deviation), 
        'cvar' (conditional value at risk)
        
    Returns:
    --------
    Dictionary with allocation weights
    """
    import scipy.cluster.hierarchy as sch
    from scipy.spatial.distance import squareform
    
    # Calculate correlation matrix
    corr = returns.corr()
    
    # Convert correlation to distance
    dist = np.sqrt(0.5 * (1 - corr))
    
    # Hierarchical clustering
    link = sch.linkage(squareform(dist), 'single')
    
    # Extract quasi-diagonalization order
    sortIx = sch.leaves_list(link)
    sortIx = corr.index[sortIx].tolist()
    
    # Calculate risk based on chosen measure
    if risk_measure == 'variance':
        # Standard volatility
        risk = returns.std() * np.sqrt(252)
    elif risk_measure == 'mad':
        # Mean absolute deviation
        risk = returns.apply(lambda x: np.mean(np.abs(x - np.mean(x)))) * np.sqrt(252)
    elif risk_measure == 'cvar':
        # Conditional Value at Risk (Expected Shortfall)
        alpha = 0.05  # 95% CVaR
        risk = returns.apply(lambda x: -np.mean(np.sort(x)[:int(len(x)*alpha)]))
    else:
        logger.warning(f"Unknown risk measure: {risk_measure}, using variance")
        risk = returns.std() * np.sqrt(252)
    
    # Calculate inverse risk
    inv_risk = 1 / risk
    
    # Normalize weights
    weights = inv_risk / inv_risk.sum()
    
    # Sort weights based on clustering
    sorted_weights = {ticker: weights[ticker] for ticker in sortIx}
    
    return sorted_weights


def dynamic_allocation_model(backtests, lookback_window=60, rebalance_freq=20,
                           base_allocation='risk_parity'):
    """
    Dynamic allocation that adapts to changing market conditions.
    
    Parameters:
    -----------
    backtests : dict
        Dictionary with backtest results for each pair
    lookback_window : int
        Window for estimating performance metrics
    rebalance_freq : int
        Frequency of rebalancing in days
    base_allocation : str
        Base allocation method
        
    Returns:
    --------
    DataFrame with allocation weights over time
    """
    # Extract return series
    pair_returns = {}
    common_dates = None
    
    for pair_str, results in backtests.items():
        if pair_str == 'aggregate' or pair_str.startswith('error') or pair_str == 'portfolio':
            continue
            
        if 'returns' in results:
            returns_series = results['returns']
            
            if common_dates is None:
                common_dates = returns_series.index
            else:
                common_dates = common_dates.intersection(returns_series.index)
                
            pair_returns[pair_str] = returns_series
    
    if not pair_returns or common_dates is None or len(common_dates) == 0:
        logger.error("No valid return data for dynamic allocation")
        return None
    
    # Create DataFrame with aligned returns
    aligned_returns = pd.DataFrame({
        pair: returns.reindex(common_dates) for pair, returns in pair_returns.items()
    }).fillna(0)
    
    # Determine rebalance dates
    rebalance_dates = [common_dates[i] for i in range(0, len(common_dates), rebalance_freq)]
    if common_dates[-1] not in rebalance_dates:
        rebalance_dates.append(common_dates[-1])
    
    # Create DataFrame for dynamic allocations
    dynamic_weights = pd.DataFrame(index=common_dates, columns=pair_returns.keys())
    
    # Calculate allocations at each rebalance date
    for i, rebalance_date in enumerate(rebalance_dates):
        # Skip first rebalance if we don't have enough history
        if i == 0 and common_dates.get_loc(rebalance_date) < lookback_window:
            continue
        
        # Get data up to rebalance date
        date_idx = common_dates.get_loc(rebalance_date)
        start_idx = max(0, date_idx - lookback_window)
        
        # Use recent data for allocation
        recent_returns = aligned_returns.iloc[start_idx:date_idx]
        
        if len(recent_returns) < 30:  # Need sufficient data
            continue
        
        try:
            # Calculate performance metrics for each pair
            metrics = {}
            
            for pair in pair_returns.keys():
                pair_rets = recent_returns[pair]
                
                sharpe = pair_rets.mean() / pair_rets.std() * np.sqrt(252) if pair_rets.std() > 0 else 0
                vol = pair_rets.std() * np.sqrt(252)
                
                # Calculate drawdown
                cum_rets = (1 + pair_rets).cumprod()
                running_max = cum_rets.cummax()
                drawdown = (cum_rets / running_max - 1).min()
                
                metrics[pair] = {
                    'sharpe': sharpe,
                    'vol': vol,
                    'drawdown': drawdown
                }
            
            # Calculate allocation weights based on metrics
            if base_allocation == 'risk_parity':
                # Inverse volatility weighting
                inv_vols = {pair: 1.0 / max(metrics[pair]['vol'], 1e-8) for pair in metrics}
                total_inv_vol = sum(inv_vols.values())
                raw_weights = {pair: w / total_inv_vol for pair, w in inv_vols.items()}
                
                # Adjust weights based on Sharpe ratio
                sharpe_score = {pair: max(metrics[pair]['sharpe'], 0) for pair in metrics}
                total_sharpe = sum(sharpe_score.values())
                
                # Final weights blend inverse volatility and Sharpe
                weights = {}
                for pair in metrics:
                    # Base weight from inverse volatility
                    base_weight = raw_weights[pair]
                    
                    # Sharpe adjustment
                    sharpe_adj = 0.5 * (sharpe_score[pair] / total_sharpe if total_sharpe > 0 else 0)
                    
                    # Drawdown penalty
                    dd_penalty = min(0, metrics[pair]['drawdown']) * 0.1
                    
                    # Combined weight
                    weights[pair] = max(0, base_weight + sharpe_adj + dd_penalty)
            else:
                # Default to equal weights
                weights = {pair: 1.0 / len(metrics) for pair in metrics}
            
            # Normalize weights
            total_weight = sum(weights.values())
            weights = {pair: w / total_weight for pair, w in weights.items()} if total_weight > 0 else {pair: 1.0 / len(metrics) for pair in metrics}
            
            # Store weights
            next_rebalance = rebalance_dates[i+1] if i+1 < len(rebalance_dates) else common_dates[-1]
            rebalance_period = common_dates[common_dates.get_loc(rebalance_date):common_dates.get_loc(next_rebalance)+1]
            
            for date in rebalance_period:
                for pair, weight in weights.items():
                    dynamic_weights.loc[date, pair] = weight
        
        except Exception as e:
            logger.error(f"Error calculating dynamic weights for {rebalance_date}: {str(e)}")
            continue
    
    # Fill any missing weights with the last valid ones
    dynamic_weights = dynamic_weights.fillna(method='ffill')
    
    # If still missing, use equal weights
    if dynamic_weights.isnull().any().any():
        equal_weight = 1.0 / len(pair_returns)
        dynamic_weights = dynamic_weights.fillna(equal_weight)
    
    return dynamic_weights
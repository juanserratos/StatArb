"""
Portfolio optimization and rebalancing logic.
"""
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def backtest_portfolio(backtests, allocations=None, initial_capital=1000000,
                      transaction_costs=0.0005, rebalance_freq='M'):
    """
    Backtest a portfolio of pairs with custom allocations.
    
    Parameters:
    -----------
    backtests : dict
        Dictionary with backtest results for each pair
    allocations : dict, optional
        Dictionary with allocation weights for each pair
    initial_capital : float
        Initial capital
    transaction_costs : float
        Transaction cost as a fraction of trade value
    rebalance_freq : str
        Rebalancing frequency: 'D' (daily), 'W' (weekly), 'M' (monthly)
        
    Returns:
    --------
    Dictionary with portfolio backtest results
    """
    # Validate inputs
    valid_pairs = []
    
    for pair_str, results in backtests.items():
        if pair_str != 'aggregate' and not pair_str.startswith('error') and pair_str != 'portfolio':
            if 'portfolio' in results and 'returns' in results:
                valid_pairs.append(pair_str)
    
    if not valid_pairs:
        logger.error("No valid pairs with backtest results")
        return None
    
    # Use equal weights if no allocations provided
    if allocations is None:
        allocations = {pair: 1.0 / len(valid_pairs) for pair in valid_pairs}
    
    # Filter allocations to valid pairs
    allocations = {p: w for p, w in allocations.items() if p in valid_pairs}
    
    if not allocations:
        logger.error("No valid allocations")
        return None
    
    # Normalize allocations to sum to 1
    allocation_sum = sum(allocations.values())
    normalized_allocs = {p: w / allocation_sum for p, w in allocations.items()}
    
    # Collect portfolio values for each pair
    pair_portfolios = {}
    all_dates = set()
    
    for pair in valid_pairs:
        if pair in normalized_allocs:
            pair_portfolios[pair] = backtests[pair]['portfolio']
            all_dates.update(pair_portfolios[pair].index)
    
    # Create a common date range
    all_dates = sorted(all_dates)
    date_range = pd.DatetimeIndex(all_dates)
    
    # Reindex all portfolios to the common date range and handle NaN values
    for pair in pair_portfolios:
        pair_portfolios[pair] = pair_portfolios[pair].reindex(date_range).fillna(method='ffill')
        # Fill any remaining NaNs with initial value
        pair_portfolios[pair] = pair_portfolios[pair].fillna(initial_capital)
    
    # Initialize portfolio tracking
    portfolio_df = pd.DataFrame(index=date_range)
    portfolio_df['total_value'] = 0.0
    
    # Add individual pair allocations
    for pair, alloc in normalized_allocs.items():
        if pair in pair_portfolios:
            col_name = f"{pair}_value"
            target_col = f"{pair}_target"
            actual_col = f"{pair}_actual"
            
            portfolio_df[col_name] = pair_portfolios[pair]
            portfolio_df[target_col] = initial_capital * alloc
            
            # Calculate target vs actual allocation ratio
            portfolio_df[actual_col] = 0.0  # Will be filled during simulation
    
    # Implement portfolio rebalancing simulation
    portfolio_df['cash'] = initial_capital
    last_rebalance = None
    
    # Determine rebalancing dates based on frequency
    if rebalance_freq == 'D':
        rebalance_dates = date_range
    elif rebalance_freq == 'W':
        rebalance_dates = pd.date_range(
            start=date_range[0], end=date_range[-1], freq='W-FRI'
        )
    elif rebalance_freq == 'M':
        rebalance_dates = pd.date_range(
            start=date_range[0], end=date_range[-1], freq='BM'
        )
    else:
        logger.warning(f"Invalid rebalance frequency: {rebalance_freq}, using monthly")
        rebalance_dates = pd.date_range(
            start=date_range[0], end=date_range[-1], freq='BM'
        )
    
    # Convert to actual dates in our range
    rebalance_dates = [d for d in rebalance_dates if d in date_range]
    
    # Add the first date as a rebalance date
    if date_range[0] not in rebalance_dates:
        rebalance_dates = [date_range[0]] + rebalance_dates
    
    # Simulate portfolio over time
    for i, date in enumerate(date_range):
        # On first day, invest according to allocations
        if i == 0:
            for pair, alloc in normalized_allocs.items():
                if pair in pair_portfolios:
                    # Allocate capital
                    pair_amount = initial_capital * alloc
                    portfolio_df.at[date, f"{pair}_actual"] = pair_amount
                    portfolio_df.at[date, 'cash'] -= pair_amount
        else:
            prev_date = date_range[i-1]
            
            # Update values based on pair returns
            for pair in normalized_allocs:
                if pair in pair_portfolios:
                    # Calculate pair return for this day
                    if prev_date in pair_portfolios[pair].index and date in pair_portfolios[pair].index:
                        pair_return = (pair_portfolios[pair][date] / pair_portfolios[pair][prev_date]) - 1
                    else:
                        pair_return = 0.0
                    
                    # Update allocation
                    prev_allocation = portfolio_df.at[prev_date, f"{pair}_actual"]
                    portfolio_df.at[date, f"{pair}_actual"] = prev_allocation * (1 + pair_return)
        
        # Rebalance if needed
        if date in rebalance_dates:
            # Calculate current total portfolio value
            current_value = portfolio_df.at[date, 'cash'] + sum(
                portfolio_df.at[date, f"{pair}_actual"] for pair in normalized_allocs 
                if pair in pair_portfolios
            )
            
            # Calculate target allocations
            for pair, alloc in normalized_allocs.items():
                if pair in pair_portfolios:
                    portfolio_df.at[date, f"{pair}_target"] = current_value * alloc
            
            # Execute rebalancing trades
            total_trade_value = 0.0
            
            for pair, alloc in normalized_allocs.items():
                if pair in pair_portfolios:
                    current_allocation = portfolio_df.at[date, f"{pair}_actual"]
                    target_allocation = portfolio_df.at[date, f"{pair}_target"]
                    
                    trade_value = abs(target_allocation - current_allocation)
                    total_trade_value += trade_value
                    
                    # Update allocation
                    portfolio_df.at[date, f"{pair}_actual"] = target_allocation
            
            # Apply transaction costs
            transaction_cost = total_trade_value * transaction_costs
            portfolio_df.at[date, 'cash'] -= transaction_cost
            
            last_rebalance = date
            logger.info(f"Rebalanced portfolio on {date}, transaction cost: ${transaction_cost:.2f}")
        else:
            # Carry forward cash
            portfolio_df.at[date, 'cash'] = portfolio_df.at[prev_date, 'cash']
        
        # Calculate total portfolio value
        portfolio_df.at[date, 'total_value'] = portfolio_df.at[date, 'cash'] + sum(
            portfolio_df.at[date, f"{pair}_actual"] for pair in normalized_allocs 
            if pair in pair_portfolios
        )
    
    # Calculate portfolio returns and metrics
    portfolio_returns = portfolio_df['total_value'].pct_change().fillna(0)
    
    # Annualized metrics
    total_return = (portfolio_df['total_value'].iloc[-1] / initial_capital) - 1
    n_years = (date_range[-1] - date_range[0]).days / 365.25
    annual_return = ((1 + total_return) ** (1 / n_years)) - 1 if n_years > 0 else 0
    
    annual_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
    
    # Drawdown analysis
    rolling_max = portfolio_df['total_value'].cummax()
    drawdown = portfolio_df['total_value'] / rolling_max - 1
    max_drawdown = drawdown.min()
    
    # Calculate additional risk metrics
    daily_returns = portfolio_returns.values
    positive_days = sum(1 for r in daily_returns if r > 0)
    negative_days = sum(1 for r in daily_returns if r < 0)
    
    win_rate = positive_days / len(daily_returns) if len(daily_returns) > 0 else 0
    
    positive_returns = sum(r for r in daily_returns if r > 0)
    negative_returns = sum(r for r in daily_returns if r < 0)
    
    profit_factor = abs(positive_returns / negative_returns) if negative_returns != 0 else float('inf')
    
    # Monthly returns analysis
    monthly_returns = portfolio_df['total_value'].resample('M').last().pct_change().fillna(0)
    positive_months = sum(1 for r in monthly_returns if r > 0)
    total_months = len(monthly_returns)
    
    # Log portfolio performance
    logger.info("\nPortfolio Performance Summary:")
    logger.info(f"Total Return: {total_return:.2%}")
    logger.info(f"Annual Return: {annual_return:.2%}")
    logger.info(f"Annual Volatility: {annual_vol:.2%}")
    logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    logger.info(f"Max Drawdown: {max_drawdown:.2%}")
    logger.info(f"Win Rate (daily): {win_rate:.2%}")
    logger.info(f"Profit Factor: {profit_factor:.2f}")
    logger.info(f"Positive Months: {positive_months}/{total_months} ({positive_months/total_months:.2%})")
    
    # Return results
    return {
        'portfolio': portfolio_df['total_value'],
        'returns': portfolio_returns,
        'metrics': {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'positive_months_pct': positive_months / total_months if total_months > 0 else 0,
            'calmar_ratio': -annual_return / max_drawdown if max_drawdown != 0 else float('inf')
        },
        'allocations': normalized_allocs,
        'rebalance_dates': rebalance_dates,
        'detailed_portfolio': portfolio_df
    }


def optimize_portfolio_weights(returns, objective='sharpe', 
                              constraints=None, bounds=None):
    """
    Optimize portfolio weights using various objectives.
    
    Parameters:
    -----------
    returns : DataFrame
        DataFrame with returns for each component
    objective : str
        Optimization objective: 'sharpe', 'min_var', 'max_return', 'min_cvar'
    constraints : list, optional
        List of constraints for optimization
    bounds : tuple, optional
        Tuple of bounds for weights
        
    Returns:
    --------
    Dictionary with optimized weights
    """
    from scipy.optimize import minimize
    
    # Prepare data
    n_assets = returns.shape[1]
    mean_returns = returns.mean() * 252  # Annualized
    cov_matrix = returns.cov() * 252  # Annualized
    
    # Default bounds: weights between 0 and 1
    if bounds is None:
        bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Default constraints: sum of weights = 1
    if constraints is None:
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    # Initial guess: equal weights
    initial_weights = np.ones(n_assets) / n_assets
    
    # Define objective functions
    if objective == 'sharpe':
        # Negative Sharpe ratio (for minimization)
        def obj_function(weights):
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
            
    elif objective == 'min_var':
        # Portfolio variance
        def obj_function(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
            
    elif objective == 'max_return':
        # Negative portfolio return (for minimization)
        def obj_function(weights):
            return -np.sum(mean_returns * weights)
            
    elif objective == 'min_cvar':
        # Conditional Value at Risk (Expected Shortfall)
        alpha = 0.05  # 95% confidence level
        
        def obj_function(weights):
            # Generate scenario returns
            portfolio_returns = returns.dot(weights)
            
            # Sort returns and find worst alpha percentile
            sorted_returns = np.sort(portfolio_returns)
            indices = int(alpha * len(sorted_returns))
            
            # Calculate CVaR
            cvar = -np.mean(sorted_returns[:indices])
            return cvar
    else:
        logger.error(f"Unknown objective function: {objective}")
        return None
    
    # Run optimization
    try:
        result = minimize(
            obj_function,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimized_weights = result.x
            
            # Create dictionary of weights
            weights_dict = {asset: weight for asset, weight 
                           in zip(returns.columns, optimized_weights)}
            
            # Calculate performance metrics
            portfolio_return = np.sum(mean_returns * optimized_weights)
            portfolio_volatility = np.sqrt(np.dot(optimized_weights.T, 
                                                np.dot(cov_matrix, optimized_weights)))
            sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
            
            logger.info(f"Optimization successful with {objective} objective:")
            logger.info(f"Expected annual return: {portfolio_return:.2%}")
            logger.info(f"Expected annual volatility: {portfolio_volatility:.2%}")
            logger.info(f"Expected Sharpe ratio: {sharpe_ratio:.2f}")
            
            return {
                'weights': weights_dict,
                'metrics': {
                    'return': portfolio_return,
                    'volatility': portfolio_volatility,
                    'sharpe': sharpe_ratio
                }
            }
        else:
            logger.error(f"Optimization failed: {result.message}")
            return None
    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}")
        return None


def calculate_efficient_frontier(returns, num_portfolios=50, risk_free_rate=0):
    """
    Calculate the efficient frontier of portfolios.
    
    Parameters:
    -----------
    returns : DataFrame
        DataFrame with returns for each component
    num_portfolios : int
        Number of portfolios to generate
    risk_free_rate : float
        Risk-free rate for Sharpe ratio calculation
        
    Returns:
    --------
    DataFrame with portfolio weights and metrics for the efficient frontier
    """
    from scipy.optimize import minimize
    
    # Prepare data
    n_assets = returns.shape[1]
    mean_returns = returns.mean() * 252  # Annualized
    cov_matrix = returns.cov() * 252  # Annualized
    
    # Create array for storing results
    results = np.zeros((num_portfolios, n_assets + 3))  # +3 for return, volatility, sharpe
    
    # Find minimum volatility portfolio
    bounds = tuple((0, 1) for _ in range(n_assets))
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    def min_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    min_vol_result = minimize(
        min_volatility,
        np.ones(n_assets) / n_assets,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    min_vol_weights = min_vol_result.x
    min_vol_ret = np.sum(mean_returns * min_vol_weights)
    min_vol_vol = min_volatility(min_vol_weights)
    
    # Find maximum return portfolio
    def neg_return(weights):
        return -np.sum(mean_returns * weights)
    
    max_ret_result = minimize(
        neg_return,
        np.ones(n_assets) / n_assets,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    max_ret_weights = max_ret_result.x
    max_ret = np.sum(mean_returns * max_ret_weights)
    
    # Generate portfolios along the efficient frontier
    target_returns = np.linspace(min_vol_ret, max_ret, num_portfolios)
    
    for i, target_ret in enumerate(target_returns):
        # Add return constraint
        return_constraint = {
            'type': 'eq',
            'fun': lambda x: np.sum(mean_returns * x) - target_ret
        }
        
        constraints_i = constraints + [return_constraint]
        
        result = minimize(
            min_volatility,
            np.ones(n_assets) / n_assets,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_i
        )
        
        weights = result.x
        vol = min_volatility(weights)
        sharpe = (target_ret - risk_free_rate) / vol if vol > 0 else 0
        
        # Store results
        results[i, :n_assets] = weights
        results[i, n_assets] = target_ret
        results[i, n_assets + 1] = vol
        results[i, n_assets + 2] = sharpe
    
    # Convert to DataFrame
    columns = list(returns.columns) + ['Return', 'Volatility', 'Sharpe']
    efficient_frontier = pd.DataFrame(results, columns=columns)
    
    return efficient_frontier


def rebalance_portfolio(current_allocations, target_allocations, 
                       min_trade_size=0.01, max_deviation=0.05):
    """
    Generate rebalancing trades given current and target allocations.
    
    Parameters:
    -----------
    current_allocations : dict
        Dictionary with current allocation weights
    target_allocations : dict
        Dictionary with target allocation weights
    min_trade_size : float
        Minimum trade size as a percentage of portfolio
    max_deviation : float
        Maximum allowed deviation from target before rebalancing
        
    Returns:
    --------
    Dictionary with trades required for rebalancing
    """
    trades = {}
    
    # Check for missing keys in either allocation
    all_assets = set(current_allocations.keys()) | set(target_allocations.keys())
    
    for asset in all_assets:
        # Get current and target allocations, default to 0 if not present
        current = current_allocations.get(asset, 0)
        target = target_allocations.get(asset, 0)
        
        # Calculate deviation
        deviation = target - current
        
        # Check if rebalancing is required
        if abs(deviation) > max_deviation or (asset in target_allocations and asset not in current_allocations):
            # Trade size exceeds minimum and deviation threshold
            if abs(deviation) >= min_trade_size:
                trades[asset] = deviation
    
    return trades
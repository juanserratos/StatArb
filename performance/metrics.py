"""
Performance metrics calculation for the statistical arbitrage strategy.
"""
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def calculate_performance_metrics(portfolio_equity, risk_free_rate=0.02):
    """
    Calculate key performance metrics for a portfolio.
    
    Parameters:
    -----------
    portfolio_equity : Series
        Series with portfolio equity values over time
    risk_free_rate : float
        Annual risk-free rate
        
    Returns:
    --------
    Dictionary with performance metrics
    """
    # Calculate returns
    returns = portfolio_equity.pct_change().dropna()
    
    # Basic metrics
    total_return = (portfolio_equity.iloc[-1] / portfolio_equity.iloc[0]) - 1
    n_days = len(portfolio_equity)
    annual_return = ((1 + total_return) ** (252 / n_days)) - 1 if n_days > 0 else 0
    daily_volatility = returns.std()
    annual_volatility = daily_volatility * np.sqrt(252)
    
    # Sharpe Ratio
    excess_return = annual_return - risk_free_rate
    sharpe_ratio = excess_return / annual_volatility if annual_volatility > 0 else 0
    
    # Sortino Ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252)
    sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0
    
    # Maximum Drawdown
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max) - 1
    max_drawdown = drawdown.min()
    
    # Calmar Ratio
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')
    
    # Win Rate
    win_rate = (returns > 0).mean()
    
    # Profit Factor
    gross_profits = returns[returns > 0].sum()
    gross_losses = abs(returns[returns < 0].sum())
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
    
    # Average Win/Loss
    avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
    avg_loss = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    
    # Return/Risk Statistics
    returns_annualized = returns.mean() * 252
    volatility_annualized = returns.std() * np.sqrt(252)
    
    # Downside Deviation
    target_return = 0
    downside_diff = np.minimum(returns - target_return / 252, 0)
    downside_deviation = np.sqrt(np.mean(downside_diff ** 2)) * np.sqrt(252)
    
    # Information Ratio (assuming benchmark return of risk-free rate for simplicity)
    benchmark_return = risk_free_rate / 252  # Daily risk-free rate
    tracking_error = (returns - benchmark_return).std() * np.sqrt(252)
    information_ratio = (returns_annualized - risk_free_rate) / tracking_error if tracking_error > 0 else 0
    
    # CAGR (Compound Annual Growth Rate)
    years = n_days / 252
    cagr = (portfolio_equity.iloc[-1] / portfolio_equity.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
    
    # Metrics dictionary
    metrics = {
        'total_return': total_return,
        'annual_return': annual_return,
        'cagr': cagr,
        'daily_volatility': daily_volatility,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'win_loss_ratio': win_loss_ratio,
        'information_ratio': information_ratio,
        'downside_deviation': downside_deviation,
        'tracking_error': tracking_error
    }
    
    return metrics


def calculate_drawdowns(returns, top_n=5):
    """
    Calculate drawdown statistics.
    
    Parameters:
    -----------
    returns : Series
        Series with returns
    top_n : int
        Number of largest drawdowns to return
        
    Returns:
    --------
    DataFrame with drawdown statistics
    """
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max) - 1
    
    # Find drawdown periods
    is_in_drawdown = drawdown < 0
    
    # Initialize variables for tracking drawdown periods
    drawdown_periods = []
    current_drawdown_start = None
    
    # Identify drawdown periods
    for date, in_dd in is_in_drawdown.items():
        if in_dd and current_drawdown_start is None:
            current_drawdown_start = date
        elif not in_dd and current_drawdown_start is not None:
            # Drawdown just ended, calculate metrics
            dd_end = date
            dd_data = drawdown[current_drawdown_start:dd_end]
            max_dd = dd_data.min()
            max_dd_date = dd_data.idxmin()
            
            # Only include if significant drawdown
            if max_dd < -0.01:  # At least 1% drawdown
                drawdown_periods.append({
                    'start_date': current_drawdown_start,
                    'max_drawdown_date': max_dd_date,
                    'end_date': dd_end,
                    'max_drawdown': max_dd,
                    'duration': (dd_end - current_drawdown_start).days,
                    'recovery': (dd_end - max_dd_date).days
                })
            
            # Reset for next drawdown
            current_drawdown_start = None
    
    # Check if we're still in a drawdown at the end of the series
    if current_drawdown_start is not None:
        dd_data = drawdown[current_drawdown_start:]
        max_dd = dd_data.min()
        max_dd_date = dd_data.idxmin()
        
        drawdown_periods.append({
            'start_date': current_drawdown_start,
            'max_drawdown_date': max_dd_date,
            'end_date': None,  # Still in drawdown
            'max_drawdown': max_dd,
            'duration': (dd_data.index[-1] - current_drawdown_start).days,
            'recovery': None  # Still in drawdown
        })
    
    # Convert to DataFrame and sort by max drawdown
    if drawdown_periods:
        dd_df = pd.DataFrame(drawdown_periods)
        dd_df.sort_values('max_drawdown', inplace=True)
        
        # Return top N drawdowns
        return dd_df.head(top_n)
    else:
        return pd.DataFrame()


def calculate_rolling_metrics(returns, window=252):
    """
    Calculate rolling performance metrics.
    
    Parameters:
    -----------
    returns : Series
        Series with returns
    window : int
        Rolling window size in days
        
    Returns:
    --------
    DataFrame with rolling metrics
    """
    # Initialize DataFrame for rolling metrics
    rolling_metrics = pd.DataFrame(index=returns.index)
    
    # Rolling annualized return
    rolling_metrics['return'] = returns.rolling(window).mean() * 252
    
    # Rolling annualized volatility
    rolling_metrics['volatility'] = returns.rolling(window).std() * np.sqrt(252)
    
    # Rolling Sharpe ratio
    rolling_metrics['sharpe'] = rolling_metrics['return'] / rolling_metrics['volatility']
    
    # Rolling drawdown
    rolling_cum_returns = (1 + returns).rolling(window).apply(
        lambda x: (1 + x).prod() - 1, raw=True
    )
    
    # This is an approximation - true rolling drawdown is more complex
    rolling_max = rolling_cum_returns.rolling(window).max()
    rolling_metrics['drawdown'] = rolling_cum_returns / rolling_max - 1
    
    # Rolling win rate
    rolling_metrics['win_rate'] = returns.rolling(window).apply(
        lambda x: (x > 0).mean(), raw=True
    )
    
    # Rolling Sortino ratio
    def sortino_ratio(x):
        downside = x[x < 0]
        if len(downside) == 0 or downside.std() == 0:
            return np.nan
        return (x.mean() * 252) / (downside.std() * np.sqrt(252))
    
    rolling_metrics['sortino'] = returns.rolling(window).apply(
        sortino_ratio, raw=True
    )
    
    return rolling_metrics


def calculate_monthly_returns(returns):
    """
    Calculate monthly returns and statistics.
    
    Parameters:
    -----------
    returns : Series
        Series with daily returns
        
    Returns:
    --------
    DataFrame with monthly returns and statistics
    """
    # Resample to get monthly returns
    monthly_returns = (1 + returns).resample('M').prod() - 1
    
    # Create a pivot table of month vs year
    monthly_table = pd.DataFrame({
        'year': monthly_returns.index.year,
        'month': monthly_returns.index.month,
        'return': monthly_returns.values
    })
    
    pivot = monthly_table.pivot(index='year', columns='month', values='return')
    
    # Add year total
    pivot['Year'] = (1 + pivot.fillna(0)).prod(axis=1) - 1
    
    # Calculate monthly statistics
    monthly_stats = pd.DataFrame({
        'mean': monthly_returns.mean(),
        'std': monthly_returns.std(),
        'min': monthly_returns.min(),
        'max': monthly_returns.max(),
        'positive': (monthly_returns > 0).mean()
    }).T
    
    return {
        'monthly_returns': monthly_returns,
        'monthly_table': pivot,
        'monthly_stats': monthly_stats
    }


def calculate_return_distribution(returns):
    """
    Calculate return distribution statistics.
    
    Parameters:
    -----------
    returns : Series
        Series with returns
        
    Returns:
    --------
    Dictionary with distribution statistics
    """
    import scipy.stats as stats
    
    # Basic statistics
    mean = returns.mean()
    median = returns.median()
    std = returns.std()
    skew = stats.skew(returns.dropna())
    kurtosis = stats.kurtosis(returns.dropna())
    
    # Calculate percentiles
    percentiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    percentile_values = {}
    
    for p in percentiles:
        percentile_values[f'p{int(p*100)}'] = returns.quantile(p)
    
    # Value at Risk (VaR)
    var_95 = -returns.quantile(0.05)
    var_99 = -returns.quantile(0.01)
    
    # Expected Shortfall (Conditional VaR)
    cvar_95 = -returns[returns <= returns.quantile(0.05)].mean()
    cvar_99 = -returns[returns <= returns.quantile(0.01)].mean()
    
    # Return distribution statistics
    distribution_stats = {
        'mean': mean,
        'median': median,
        'std': std,
        'skew': skew,
        'kurtosis': kurtosis,
        'percentiles': percentile_values,
        'var_95': var_95,
        'var_99': var_99,
        'cvar_95': cvar_95,
        'cvar_99': cvar_99
    }
    
    return distribution_stats


def calculate_trade_statistics(trades):
    """
    Calculate trade-specific statistics.
    
    Parameters:
    -----------
    trades : list
        List of trade dictionaries
        
    Returns:
    --------
    Dictionary with trade statistics
    """
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'avg_profit': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'avg_trade': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'avg_duration': 0
        }
    
    # Convert trades to DataFrame for easier analysis
    trades_df = pd.DataFrame(trades)
    
    # Filter to relevant trades
    if 'action' in trades_df.columns:
        open_trades = trades_df[trades_df['action'] == 'open']
        close_trades = trades_df[trades_df['action'] == 'close']
    else:
        # Assume all trades are open-close pairs
        open_trades = trades_df
        close_trades = pd.DataFrame()
    
    # Calculate trade statistics
    total_trades = len(open_trades)
    
    # If we have PnL information
    if 'pnl' in open_trades.columns:
        winning_trades = open_trades[open_trades['pnl'] > 0]
        losing_trades = open_trades[open_trades['pnl'] <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_profit = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if losing_trades['pnl'].sum() != 0 else float('inf')
        avg_trade = open_trades['pnl'].mean()
        largest_win = winning_trades['pnl'].max() if len(winning_trades) > 0 else 0
        largest_loss = losing_trades['pnl'].min() if len(losing_trades) > 0 else 0
    else:
        # Cannot calculate these without PnL information
        win_rate = 0
        avg_profit = 0
        avg_loss = 0
        profit_factor = 0
        avg_trade = 0
        largest_win = 0
        largest_loss = 0
    
    # Calculate average trade duration if dates are available
    if 'date' in open_trades.columns and 'date' in close_trades.columns and len(close_trades) > 0:
        # Match open and close trades
        matched_trades = pd.merge(
            open_trades, close_trades, 
            left_index=True, right_index=True,
            suffixes=('_open', '_close')
        )
        
        if 'date_open' in matched_trades.columns and 'date_close' in matched_trades.columns:
            matched_trades['duration'] = (matched_trades['date_close'] - matched_trades['date_open']).dt.days
            avg_duration = matched_trades['duration'].mean()
        else:
            avg_duration = 0
    else:
        avg_duration = 0
    
    # Calculate transaction costs
    total_costs = trades_df['cost'].sum() if 'cost' in trades_df.columns else 0
    avg_cost = trades_df['cost'].mean() if 'cost' in trades_df.columns else 0
    
    # Return trade statistics
    trade_stats = {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_profit': avg_profit,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'avg_trade': avg_trade,
        'largest_win': largest_win,
        'largest_loss': largest_loss,
        'avg_duration': avg_duration,
        'total_costs': total_costs,
        'avg_cost': avg_cost
    }
    
    return trade_stats
"""
Execution simulation for the statistical arbitrage strategy.
"""
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def backtest_pairs(price_data, pair, signals, hedge_ratios=None, 
                 transaction_cost=0.0005, initial_capital=1000000):
    """
    Backtest a pair trade with historical data.
    
    Parameters:
    -----------
    price_data : DataFrame
        DataFrame with price data
    pair : tuple
        Tuple containing the ticker symbols of the pair
    signals : Series
        Series with trading signals
    hedge_ratios : Series, optional
        Series with hedge ratios for each date
    transaction_cost : float
        Transaction cost as a fraction of trade value
    initial_capital : float
        Initial capital for the backtest
        
    Returns:
    --------
    Dictionary with backtest results
    """
    if price_data is None or signals is None:
        logger.error("Price data or signals is None")
        return None
        
    stock1, stock2 = pair
    
    # Align data
    try:
        aligned_data = pd.concat([
            price_data[[stock1, stock2]],
            signals.to_frame('signal')
        ], axis=1)
        
        aligned_data = aligned_data.dropna()
        
        if aligned_data.empty:
            logger.error("No overlapping data after alignment")
            return None
            
        if hedge_ratios is not None:
            aligned_data = pd.concat([aligned_data, hedge_ratios.to_frame('hedge_ratio')], axis=1)
            aligned_data = aligned_data.dropna()
            
            if aligned_data.empty:
                logger.error("No overlapping data after adding hedge ratios")
                return None
        
        # Initialize portfolio and positions
        portfolio = pd.Series(index=aligned_data.index, data=initial_capital)
        positions = pd.DataFrame(index=aligned_data.index, columns=[stock1, stock2], data=0.0)
        cash = pd.Series(index=aligned_data.index, data=initial_capital)
        trades = []
        
        # Implement trading logic
        prev_signal = 0
        
        for i in range(1, len(aligned_data)):
            current_date = aligned_data.index[i]
            prev_date = aligned_data.index[i-1]
            
            current_signal = aligned_data['signal'].iloc[i]
            
            # Determine hedge ratio
            if hedge_ratios is not None and 'hedge_ratio' in aligned_data.columns:
                beta = aligned_data['hedge_ratio'].iloc[i]
            else:
                # Use a simple OLS regression for the hedge ratio
                window = min(60, i)
                train_data = aligned_data.iloc[i-window:i]
                X = train_data[stock1].values.reshape(-1, 1)
                y = train_data[stock2].values
                beta = np.linalg.lstsq(X, y, rcond=None)[0][0]
            
            # Position sizes (in dollars)
            stock1_value = aligned_data[stock1].iloc[i-1] * positions.iloc[i-1][stock1]
            stock2_value = aligned_data[stock2].iloc[i-1] * positions.iloc[i-1][stock2]
            
            # Update portfolio value from previous positions
            stock1_return = aligned_data[stock1].iloc[i] / aligned_data[stock1].iloc[i-1] - 1
            stock2_return = aligned_data[stock2].iloc[i] / aligned_data[stock2].iloc[i-1] - 1
            
            portfolio.iloc[i] = portfolio.iloc[i-1] + stock1_value * stock1_return + stock2_value * stock2_return
            cash.iloc[i] = cash.iloc[i-1]
            
            # Carry forward positions if no signal change
            positions.iloc[i] = positions.iloc[i-1]
            
            # Execute trades on signal change
            if current_signal != prev_signal:
                # Close existing position if any
                if prev_signal != 0:
                    # Transaction costs for closing
                    close_cost = (abs(stock1_value) + abs(stock2_value)) * transaction_cost
                    portfolio.iloc[i] -= close_cost
                    cash.iloc[i] += stock1_value + stock2_value - close_cost
                    
                    trades.append({
                        'date': current_date,
                        'action': 'close',
                        'stock1_qty': -positions.iloc[i-1][stock1],
                        'stock2_qty': -positions.iloc[i-1][stock2],
                        'stock1_price': aligned_data[stock1].iloc[i],
                        'stock2_price': aligned_data[stock2].iloc[i],
                        'cost': close_cost
                    })
                    
                    positions.iloc[i] = 0
                
                # Open new position if signaled
                if current_signal != 0:
                    # Determine position sizes
                    # Allocate 80% of current portfolio value
                    allocation = portfolio.iloc[i] * 0.8
                    
                    if current_signal == 1:  # Long spread (short stock1, long stock2)
                        # Maintain dollar-neutral positions based on hedge ratio
                        total_allocation = allocation / (1 + beta)
                        stock1_allocation = -total_allocation
                        stock2_allocation = total_allocation * beta
                    else:  # Short spread (long stock1, short stock2)
                        total_allocation = allocation / (1 + beta)
                        stock1_allocation = total_allocation
                        stock2_allocation = -total_allocation * beta
                    
                    # Calculate quantities
                    stock1_qty = stock1_allocation / aligned_data[stock1].iloc[i]
                    stock2_qty = stock2_allocation / aligned_data[stock2].iloc[i]
                    
                    # Transaction costs for opening
                    open_cost = (abs(stock1_allocation) + abs(stock2_allocation)) * transaction_cost
                    portfolio.iloc[i] -= open_cost
                    cash.iloc[i] -= stock1_allocation + stock2_allocation + open_cost
                    
                    positions.iloc[i][stock1] = stock1_qty
                    positions.iloc[i][stock2] = stock2_qty
                    
                    trades.append({
                        'date': current_date,
                        'action': 'open',
                        'signal': current_signal,
                        'stock1_qty': stock1_qty,
                        'stock2_qty': stock2_qty,
                        'stock1_price': aligned_data[stock1].iloc[i],
                        'stock2_price': aligned_data[stock2].iloc[i],
                        'cost': open_cost
                    })
            
            prev_signal = current_signal
        
        # Calculate returns
        returns = portfolio.pct_change().fillna(0)
        
        # Calculate metrics
        total_return = (portfolio.iloc[-1] / portfolio.iloc[0]) - 1
        annual_return = ((1 + total_return) ** (252 / len(portfolio))) - 1
        daily_returns = returns[1:]  # Skip first day
        annual_volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
        
        # Maximum drawdown
        running_max = portfolio.cummax()
        drawdown = (portfolio / running_max) - 1
        max_drawdown = drawdown.min()
        
        # Trade statistics
        trade_stats = {
            'total_trades': sum(t['action'] == 'open' for t in trades),
            'avg_holding_period': len(portfolio) / (sum(t['action'] == 'open' for t in trades) + 1e-10),
            'transaction_costs': sum(t.get('cost', 0) for t in trades)
        }
        
        # Regime statistics
        regime_returns = {}
        if len(trades) > 0:
            trade_df = pd.DataFrame(trades)
            if 'signal' in trade_df.columns:
                for signal in trade_df['signal'].unique():
                    if pd.notna(signal):
                        signal_trades = trade_df[trade_df['signal'] == signal]
                        regime_returns[int(signal)] = signal_trades['pnl'].mean() if 'pnl' in signal_trades.columns else None
        
        # Results
        results = {
            'pair': pair,
            'portfolio': portfolio,
            'positions': positions,
            'returns': returns,
            'trades': trades,
            'metrics': {
                'total_return': total_return,
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': -annual_return / max_drawdown if max_drawdown != 0 else 0,
                'profit_factor': sum(r for r in daily_returns if r > 0) / abs(sum(r for r in daily_returns if r < 0)) if sum(r for r in daily_returns if r < 0) != 0 else float('inf'),
                'win_rate': sum(r > 0 for r in daily_returns) / len(daily_returns)
            },
            'trade_stats': trade_stats,
            'regime_returns': regime_returns
        }
        
        logger.info(f"Backtest results for {pair}: " +
                   f"Return={total_return:.4f}, " +
                   f"Sharpe={sharpe_ratio:.4f}, " +
                   f"Drawdown={max_drawdown:.4f}, " +
                   f"Trades={trade_stats['total_trades']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during backtesting for {pair}: {str(e)}")
        return None


def backtest_with_realistic_execution(pair, signals, price_data, hedge_ratios=None,
                                     initial_capital=1000000, transaction_cost_model='advanced',
                                     execution_model='twap'):
    """
    Backtest with realistic execution modeling.
    
    Parameters:
    -----------
    pair : tuple
        Tuple containing the ticker symbols of the pair
    signals : Series
        Series with trading signals
    price_data : DataFrame
        DataFrame with price series
    hedge_ratios : Series, optional
        Series with hedge ratios
    initial_capital : float
        Initial capital
    transaction_cost_model : str
        'simple', 'intermediate', or 'advanced'
    execution_model : str
        'immediate', 'twap', or 'participation'
        
    Returns:
    --------
    Dictionary with backtest results
    """
    import pandas as pd
    import numpy as np
    
    stock1, stock2 = pair
    
    # Align data
    try:
        aligned_data = pd.concat([
            price_data[[stock1, stock2]],
            signals.to_frame('signal')
        ], axis=1)
        
        aligned_data = aligned_data.dropna()
        
        if aligned_data.empty:
            logger.error("No overlapping data after alignment")
            return None
            
        if hedge_ratios is not None:
            aligned_data = pd.concat([aligned_data, hedge_ratios.to_frame('hedge_ratio')], axis=1)
            aligned_data = aligned_data.dropna()
            
            if aligned_data.empty:
                logger.error("No overlapping data after adding hedge ratios")
                return None
        
        # Initialize portfolio and positions
        portfolio_df = pd.Series(index=aligned_data.index, data=initial_capital)
        positions = pd.DataFrame(index=aligned_data.index, columns=[stock1, stock2], data=0.0)
        cash = pd.Series(index=aligned_data.index, data=initial_capital)
        trades = []
        
        # Implement trading logic with execution model
        prev_signal = 0
        pending_executions = pd.DataFrame()  # For TWAP/VWAP simulation
        
        for i in range(1, len(aligned_data)):
            current_date = aligned_data.index[i]
            prev_date = aligned_data.index[i-1]
            
            current_signal = aligned_data['signal'].iloc[i]
            
            # Determine hedge ratio
            if hedge_ratios is not None and 'hedge_ratio' in aligned_data.columns:
                beta = aligned_data['hedge_ratio'].iloc[i]
            else:
                # Use a simple OLS regression for the hedge ratio
                window = min(60, i)
                train_data = aligned_data.iloc[i-window:i]
                X = train_data[stock1].values.reshape(-1, 1)
                y = train_data[stock2].values
                beta = np.linalg.lstsq(X, y, rcond=None)[0][0]
            
            # Position sizes (in dollars)
            stock1_value = aligned_data[stock1].iloc[i-1] * positions.iloc[i-1][stock1]
            stock2_value = aligned_data[stock2].iloc[i-1] * positions.iloc[i-1][stock2]
            
            # Update portfolio value from previous positions
            stock1_return = aligned_data[stock1].iloc[i] / aligned_data[stock1].iloc[i-1] - 1
            stock2_return = aligned_data[stock2].iloc[i] / aligned_data[stock2].iloc[i-1] - 1
            
            portfolio_df.iloc[i] = portfolio_df.iloc[i-1] + stock1_value * stock1_return + stock2_value * stock2_return
            cash.iloc[i] = cash.iloc[i-1]
            
            # Carry forward positions if no signal change
            positions.iloc[i] = positions.iloc[i-1]
            
            # Process any pending executions (TWAP/VWAP)
            if execution_model in ['twap', 'participation'] and not pending_executions.empty:
                # Identify executions for today
                today_execs = pending_executions[pending_executions['execution_date'] == current_date]
                
                if not today_execs.empty:
                    for _, exec_row in today_execs.iterrows():
                        # Apply execution
                        stock1_qty = exec_row['stock1_qty']
                        stock2_qty = exec_row['stock2_qty']
                        stock1_price = exec_row['stock1_price']
                        stock2_price = exec_row['stock2_price']
                        execution_cost = exec_row['cost']
                        
                        # Update positions
                        positions.iloc[i][stock1] += stock1_qty
                        positions.iloc[i][stock2] += stock2_qty
                        
                        # Update cash (deduct trade value and costs)
                        trade_value = stock1_qty * stock1_price + stock2_qty * stock2_price
                        cash.iloc[i] -= trade_value + execution_cost
                        portfolio_df.iloc[i] -= execution_cost
                        
                        trades.append({
                            'date': current_date,
                            'action': 'execution',
                            'stock1_qty': stock1_qty,
                            'stock2_qty': stock2_qty,
                            'stock1_price': stock1_price,
                            'stock2_price': stock2_price,
                            'cost': execution_cost
                        })
                    
                    # Remove processed executions
                    pending_executions = pending_executions[pending_executions['execution_date'] != current_date]
            
            # Execute trades on signal change
            if current_signal != prev_signal:
                # Calculate new position sizes
                if current_signal != 0:
                    # Determine position sizes
                    # Allocate 80% of current portfolio value
                    allocation = portfolio_df.iloc[i] * 0.8
                    
                    if current_signal == 1:  # Long spread (short stock1, long stock2)
                        # Maintain dollar-neutral positions based on hedge ratio
                        total_allocation = allocation / (1 + beta)
                        stock1_allocation = -total_allocation
                        stock2_allocation = total_allocation * beta
                    else:  # Short spread (long stock1, short stock2)
                        total_allocation = allocation / (1 + beta)
                        stock1_allocation = total_allocation
                        stock2_allocation = -total_allocation * beta
                    
                    # Calculate quantities
                    stock1_qty = stock1_allocation / aligned_data[stock1].iloc[i]
                    stock2_qty = stock2_allocation / aligned_data[stock2].iloc[i]
                else:
                    # Exit positions
                    stock1_qty = -positions.iloc[i][stock1]
                    stock2_qty = -positions.iloc[i][stock2]
                
                # Handle execution based on the chosen model
                if execution_model == 'immediate':
                    # Immediate execution at current price
                    if transaction_cost_model == 'simple':
                        # Simple fixed cost model
                        trade_value = abs(stock1_qty * aligned_data[stock1].iloc[i]) + abs(stock2_qty * aligned_data[stock2].iloc[i])
                        execution_cost = trade_value * 0.0005  # 5 bps
                    elif transaction_cost_model == 'intermediate':
                        # Intermediate cost model with spread and commission
                        trade_value1 = abs(stock1_qty * aligned_data[stock1].iloc[i])
                        trade_value2 = abs(stock2_qty * aligned_data[stock2].iloc[i])
                        
                        # Spread cost (2 bps) + commission (1 bp)
                        spread_cost = (trade_value1 + trade_value2) * 0.0002
                        commission = (trade_value1 + trade_value2) * 0.0001
                        
                        execution_cost = spread_cost + commission
                    else:  # 'advanced'
                        # Advanced cost model with market impact
                        trade_value1 = abs(stock1_qty * aligned_data[stock1].iloc[i])
                        trade_value2 = abs(stock2_qty * aligned_data[stock2].iloc[i])
                        
                        # Spread cost (2 bps) + commission (1 bp) + market impact
                        spread_cost = (trade_value1 + trade_value2) * 0.0002
                        commission = (trade_value1 + trade_value2) * 0.0001
                        
                        # Market impact (simplified square root model)
                        # Assuming ADV (average daily volume) is roughly 1% of market cap
                        market_cap1 = aligned_data[stock1].iloc[i] * 1e9  # Simplified
                        market_cap2 = aligned_data[stock2].iloc[i] * 1e9
                        
                        adv1 = market_cap1 * 0.01
                        adv2 = market_cap2 * 0.01
                        
                        impact1 = trade_value1 * 0.1 * np.sqrt(trade_value1 / adv1)
                        impact2 = trade_value2 * 0.1 * np.sqrt(trade_value2 / adv2)
                        
                        execution_cost = spread_cost + commission + impact1 + impact2
                    
                    # Update positions
                    positions.iloc[i][stock1] += stock1_qty
                    positions.iloc[i][stock2] += stock2_qty
                    
                    # Update cash (deduct trade value and costs)
                    trade_value = stock1_qty * aligned_data[stock1].iloc[i] + stock2_qty * aligned_data[stock2].iloc[i]
                    cash.iloc[i] -= trade_value + execution_cost
                    portfolio_df.iloc[i] -= execution_cost
                    
                    trades.append({
                        'date': current_date,
                        'action': 'immediate',
                        'signal': current_signal,
                        'stock1_qty': stock1_qty,
                        'stock2_qty': stock2_qty,
                        'stock1_price': aligned_data[stock1].iloc[i],
                        'stock2_price': aligned_data[stock2].iloc[i],
                        'cost': execution_cost
                    })
                
                elif execution_model == 'twap':
                    # TWAP execution: spread over 3 days
                    execution_days = 3
                    daily_ratio = 1.0 / execution_days
                    
                    # List to store execution rows
                    new_execs = []
                    
                    for day in range(execution_days):
                        execution_date = aligned_data.index[min(i + day, len(aligned_data) - 1)]
                        
                        # Fraction of order for this day
                        day_stock1_qty = stock1_qty * daily_ratio
                        day_stock2_qty = stock2_qty * daily_ratio
                        
                        # Estimate execution price (with a slight penalty)
                        if i + day < len(aligned_data):
                            stock1_price = aligned_data[stock1].iloc[i + day]
                            stock2_price = aligned_data[stock2].iloc[i + day]
                        else:
                            # Use last available price for future dates
                            stock1_price = aligned_data[stock1].iloc[-1]
                            stock2_price = aligned_data[stock2].iloc[-1]
                        
                        # Add slippage (1 bp penalty)
                        if day_stock1_qty > 0:
                            stock1_price *= 1.0001
                        elif day_stock1_qty < 0:
                            stock1_price *= 0.9999
                            
                        if day_stock2_qty > 0:
                            stock2_price *= 1.0001
                        elif day_stock2_qty < 0:
                            stock2_price *= 0.9999
                        
                        # Calculate costs for this execution
                        if transaction_cost_model == 'simple':
                            day_trade_value = abs(day_stock1_qty * stock1_price) + abs(day_stock2_qty * stock2_price)
                            day_cost = day_trade_value * 0.0005
                        elif transaction_cost_model == 'intermediate':
                            day_trade_value1 = abs(day_stock1_qty * stock1_price)
                            day_trade_value2 = abs(day_stock2_qty * stock2_price)
                            
                            day_spread_cost = (day_trade_value1 + day_trade_value2) * 0.0002
                            day_commission = (day_trade_value1 + day_trade_value2) * 0.0001
                            
                            day_cost = day_spread_cost + day_commission
                        else:  # 'advanced'
                            day_trade_value1 = abs(day_stock1_qty * stock1_price)
                            day_trade_value2 = abs(day_stock2_qty * stock2_price)
                            
                            day_spread_cost = (day_trade_value1 + day_trade_value2) * 0.0002
                            day_commission = (day_trade_value1 + day_trade_value2) * 0.0001
                            
                            # Market impact (simplified square root model)
                            market_cap1 = stock1_price * 1e9
                            market_cap2 = stock2_price * 1e9
                            
                            adv1 = market_cap1 * 0.01
                            adv2 = market_cap2 * 0.01
                            
                            day_impact1 = day_trade_value1 * 0.1 * np.sqrt(day_trade_value1 / adv1)
                            day_impact2 = day_trade_value2 * 0.1 * np.sqrt(day_trade_value2 / adv2)
                            
                            day_cost = day_spread_cost + day_commission + day_impact1 + day_impact2
                        
                        # Add to new executions list
                        new_execs.append({
                            'execution_date': execution_date,
                            'stock1_qty': day_stock1_qty,
                            'stock2_qty': day_stock2_qty,
                            'stock1_price': stock1_price,
                            'stock2_price': stock2_price,
                            'cost': day_cost
                        })
                    
                    # Add new executions to pending executions
                    if pending_executions.empty:
                        pending_executions = pd.DataFrame(new_execs)
                    else:
                        # Use concat instead of append
                        pending_executions = pd.concat([pending_executions, pd.DataFrame(new_execs)], ignore_index=True)
                
                elif execution_model == 'participation':
                    # Similar to TWAP but with market-adaptive execution speed
                    # In practice, would use market volume profile
                    # For simplicity, use a fixed 5-day execution
                    execution_days = 5
                    
                    # Non-uniform distribution (front-loaded)
                    day_weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # 30% day 1, 25% day 2, etc.
                    
                    # List to store execution rows
                    new_execs = []
                    
                    for day in range(execution_days):
                        execution_date = aligned_data.index[min(i + day, len(aligned_data) - 1)]
                        
                        # Weighted fraction of order for this day
                        day_stock1_qty = stock1_qty * day_weights[day]
                        day_stock2_qty = stock2_qty * day_weights[day]
                        
                        # For brevity, reuse TWAP logic but with different weights
                        if i + day < len(aligned_data):
                            stock1_price = aligned_data[stock1].iloc[i + day]
                            stock2_price = aligned_data[stock2].iloc[i + day]
                        else:
                            stock1_price = aligned_data[stock1].iloc[-1]
                            stock2_price = aligned_data[stock2].iloc[-1]
                        
                        # Add slippage (1 bp penalty)
                        if day_stock1_qty > 0:
                            stock1_price *= 1.0001
                        elif day_stock1_qty < 0:
                            stock1_price *= 0.9999
                            
                        if day_stock2_qty > 0:
                            stock2_price *= 1.0001
                        elif day_stock2_qty < 0:
                            stock2_price *= 0.9999
                        
                        # Simple cost model for participation strategy
                        day_trade_value = abs(day_stock1_qty * stock1_price) + abs(day_stock2_qty * stock2_price)
                        day_cost = day_trade_value * 0.0006  # Slightly higher cost for participation
                        
                        # Add to new executions list
                        new_execs.append({
                            'execution_date': execution_date,
                            'stock1_qty': day_stock1_qty,
                            'stock2_qty': day_stock2_qty,
                            'stock1_price': stock1_price,
                            'stock2_price': stock2_price,
                            'cost': day_cost
                        })
                    
                    # Add new executions to pending executions
                    if pending_executions.empty:
                        pending_executions = pd.DataFrame(new_execs)
                    else:
                        # Use concat instead of append
                        pending_executions = pd.concat([pending_executions, pd.DataFrame(new_execs)], ignore_index=True)
            
            prev_signal = current_signal
        
        # Calculate returns
        returns = portfolio_df.pct_change().fillna(0)
        
        # Calculate metrics
        total_return = (portfolio_df.iloc[-1] / initial_capital) - 1
        annual_return = ((1 + total_return) ** (252 / len(portfolio_df))) - 1
        daily_returns = returns[1:]  # Skip first day
        annual_volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
        
        # Maximum drawdown
        running_max = portfolio_df.cummax()
        drawdown = (portfolio_df / running_max) - 1
        max_drawdown = drawdown.min()
        
        # Trade statistics
        trade_stats = {
            'total_trades': sum(t['action'] == 'immediate' for t in trades),
            'avg_holding_period': len(portfolio_df) / (sum(t['action'] == 'immediate' for t in trades) + 1e-10),
            'transaction_costs': sum(t.get('cost', 0) for t in trades)
        }
        
        # Results
        results = {
            'pair': pair,
            'portfolio': portfolio_df,
            'positions': positions,
            'returns': returns,
            'trades': trades,
            'metrics': {
                'total_return': total_return,
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': -annual_return / max_drawdown if max_drawdown != 0 else 0,
                'profit_factor': sum(r for r in daily_returns if r > 0) / abs(sum(r for r in daily_returns if r < 0)) if sum(r for r in daily_returns if r < 0) != 0 else float('inf'),
                'win_rate': sum(r > 0 for r in daily_returns) / len(daily_returns)
            },
            'trade_stats': trade_stats,
            'execution_model': execution_model,
            'transaction_cost_model': transaction_cost_model
        }
        
        logger.info(f"Backtest results with {execution_model} execution and {transaction_cost_model} costs for {pair}: " +
                   f"Return={total_return:.4f}, " +
                   f"Sharpe={sharpe_ratio:.4f}, " +
                   f"Drawdown={max_drawdown:.4f}, " +
                   f"Transaction Costs=${trade_stats['transaction_costs']:.2f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during realistic execution backtest for {pair}: {str(e)}")
        return None


def simulate_optimal_execution(pair, signals, price_data, hedge_ratio=None,
                              target_participation_rate=0.1, max_execution_time=3,
                              spread_model='constant', vol_model='garch'):
    """
    Simulate optimal execution with TWAP/VWAP and participation rate constraints.
    
    Parameters:
    -----------
    pair : tuple
        Tuple containing the ticker symbols of the pair
    signals : Series
        Series with trading signals
    price_data : DataFrame
        DataFrame with price series
    hedge_ratio : float, optional
        Fixed hedge ratio, if None uses rolling hedge ratio
    target_participation_rate : float
        Target participation rate as fraction of volume
    max_execution_time : int
        Maximum execution time in days
    spread_model : str
        Spread modeling approach: 'constant', 'proportional', 'stochastic'
    vol_model : str
        Volatility modeling approach: 'historical', 'garch'
        
    Returns:
    --------
    DataFrame with execution details
    """
    try:
        import pandas as pd
        import numpy as np
        from arch import arch_model
    except ImportError:
        logger.error("Required packages not available for simulation")
        return None
    
    stock1, stock2 = pair
    
    # Align data
    common_idx = signals.index.intersection(price_data.index)
    if len(common_idx) < 30:  # Need sufficient data
        logger.error(f"Insufficient aligned data for execution simulation")
        return None
    
    # Prepare execution dataframe
    signals = signals.reindex(common_idx)
    prices = price_data.reindex(common_idx)
    
    # Calculate dollar position sizes based on signals
    if hedge_ratio is None:
        # Use a 60-day rolling window for hedge ratio
        X = prices[stock1].rolling(60).apply(
            lambda x: np.linalg.lstsq(
                np.vstack([x.values]).T, 
                prices[stock2].iloc[x.index], 
                rcond=None
            )[0][0] if len(x) == 60 else np.nan
        )
        hedge_ratios = X
    else:
        # Use fixed hedge ratio
        hedge_ratios = pd.Series(hedge_ratio, index=common_idx)
    
    # Target position size in dollars (assuming $1M per pair)
    target_position_size = 1000000
    
    # Calculate target positions
    target_positions = pd.DataFrame(index=common_idx)
    target_positions['signal'] = signals
    target_positions['hedge_ratio'] = hedge_ratios
    
    # Calculate stock positions
    target_positions['stock1_target'] = 0.0
    target_positions['stock2_target'] = 0.0
    
    # For long spread positions (signal = 1): short stock1, long stock2
    # For short spread positions (signal = -1): long stock1, short stock2
    long_mask = target_positions['signal'] == 1
    short_mask = target_positions['signal'] == -1
    
    # Calculate dollar position sizes
    hr = target_positions['hedge_ratio']
    
    # For long spread positions
    target_positions.loc[long_mask, 'stock1_target'] = -target_position_size / (1 + hr[long_mask])
    target_positions.loc[long_mask, 'stock2_target'] = target_position_size - target_positions.loc[long_mask, 'stock1_target'].abs()
    
    # For short spread positions
    target_positions.loc[short_mask, 'stock1_target'] = target_position_size / (1 + hr[short_mask])
    target_positions.loc[short_mask, 'stock2_target'] = -target_position_size + target_positions.loc[short_mask, 'stock1_target']
    
    # Model volatility
    try:
        if vol_model == 'garch':
            # Fit GARCH models for both stocks
            returns1 = prices[stock1].pct_change().dropna()
            returns2 = prices[stock2].pct_change().dropna()
            
            garch_model1 = arch_model(returns1, vol='Garch', p=1, q=1)
            garch_result1 = garch_model1.fit(disp='off')
            vol_forecast1 = garch_result1.forecast(horizon=max_execution_time)
            
            garch_model2 = arch_model(returns2, vol='Garch', p=1, q=1)
            garch_result2 = garch_model2.fit(disp='off')
            vol_forecast2 = garch_result2.forecast(horizon=max_execution_time)
            
            # Extract volatility forecasts
            std_forecasts1 = np.sqrt(vol_forecast1.variance.values[-1])
            std_forecasts2 = np.sqrt(vol_forecast2.variance.values[-1])
        else:
            # Use historical volatility
            std_forecasts1 = [prices[stock1].pct_change().rolling(20).std().iloc[-1]] * max_execution_time
            std_forecasts2 = [prices[stock2].pct_change().rolling(20).std().iloc[-1]] * max_execution_time
    except Exception as e:
        logger.warning(f"Error modeling volatility: {e}, using historical approach")
        std_forecasts1 = [prices[stock1].pct_change().rolling(20).std().iloc[-1]] * max_execution_time
        std_forecasts2 = [prices[stock2].pct_change().rolling(20).std().iloc[-1]] * max_execution_time
    
    # Simulate execution
    execution_data = []
    current_position1 = 0.0
    current_position2 = 0.0
    
    for i in range(1, len(target_positions)):
        curr_date = target_positions.index[i]
        prev_date = target_positions.index[i-1]
        
        prev_signal = target_positions['signal'].iloc[i-1]
        curr_signal = target_positions['signal'].iloc[i]
        
        # Check if position change is needed
        if curr_signal != prev_signal:
            # Calculate target positions
            target1 = target_positions['stock1_target'].iloc[i]
            target2 = target_positions['stock2_target'].iloc[i]
            
            # Position change required
            position_change1 = target1 - current_position1
            position_change2 = target2 - current_position2
            
            # Skip if no change
            if abs(position_change1) < 1e-6 and abs(position_change2) < 1e-6:
                continue
            
            # Estimate execution time based on participation rate
            # Highly simplified - in reality would use market volume data
            price1 = prices[stock1].iloc[i]
            price2 = prices[stock2].iloc[i]
            
            # Estimate daily volume in dollars (very simplified)
            if stock1 in ['SPY', 'QQQ']:
                daily_volume1 = 1e10  # $10B for very liquid ETFs
            elif stock1.startswith('XL'):
                daily_volume1 = 5e8   # $500M for sector ETFs
            else:
                daily_volume1 = 1e8   # $100M for other ETFs
            
            if stock2 in ['SPY', 'QQQ']:
                daily_volume2 = 1e10
            elif stock2.startswith('XL'):
                daily_volume2 = 5e8
            else:
                daily_volume2 = 1e8
            
            # Calculate execution days needed
            days_needed1 = abs(position_change1) / (daily_volume1 * target_participation_rate)
            days_needed2 = abs(position_change2) / (daily_volume2 * target_participation_rate)
            
            execution_days = min(max(days_needed1, days_needed2), max_execution_time)
            execution_days = max(1, execution_days)  # At least 1 day
            
            # Round to nearest day
            execution_days = round(execution_days)
            
            # Model execution price impact
            # Simulate execution over multiple days
            for day in range(execution_days):
                # Calculate daily execution size
                day_execution1 = position_change1 / execution_days
                day_execution2 = position_change2 / execution_days
                
                # Calculate market impact based on participation rate
                # Square root model for market impact
                impact1 = 0.1 * np.sqrt(abs(day_execution1) / daily_volume1) * np.sign(day_execution1)
                impact2 = 0.1 * np.sqrt(abs(day_execution2) / daily_volume2) * np.sign(day_execution2)
                
                # Calculate execution price
                if day < len(std_forecasts1):
                    # Use volatility forecasts to model price uncertainty
                    price_uncertainty1 = price1 * std_forecasts1[day]
                    price_uncertainty2 = price2 * std_forecasts2[day]
                else:
                    price_uncertainty1 = price1 * std_forecasts1[-1]
                    price_uncertainty2 = price2 * std_forecasts2[-1]
                
                # Simulate execution price with noise
                execution_price1 = price1 * (1 + impact1 + np.random.normal(0, price_uncertainty1))
                execution_price2 = price2 * (1 + impact2 + np.random.normal(0, price_uncertainty2))
                
                # Add to execution data
                execution_data.append({
                    'date': curr_date,
                    'execution_day': day + 1,
                    'signal': curr_signal,
                    'stock1': stock1,
                    'stock2': stock2,
                    'stock1_size': day_execution1,
                    'stock2_size': day_execution2,
                    'stock1_price': execution_price1,
                    'stock2_price': execution_price2,
                    'stock1_impact_bps': impact1 * 10000,
                    'stock2_impact_bps': impact2 * 10000
                })
                
                # Update current positions
                current_position1 += day_execution1
                current_position2 += day_execution2
    
    # Convert to DataFrame
    execution_df = pd.DataFrame(execution_data)
    
    if execution_df.empty:
        logger.warning("No executions simulated")
        return pd.DataFrame()
    
    # Calculate summary statistics
    avg_impact1 = execution_df['stock1_impact_bps'].abs().mean()
    avg_impact2 = execution_df['stock2_impact_bps'].abs().mean()
    
    logger.info(f"Average execution impact for {stock1}: {avg_impact1:.2f} bps")
    logger.info(f"Average execution impact for {stock2}: {avg_impact2:.2f} bps")
    logger.info(f"Average execution days: {execution_df['execution_day'].mean():.2f}")
    
    return execution_df
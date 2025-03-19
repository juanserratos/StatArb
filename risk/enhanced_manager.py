"""
Enhanced risk management functionality for production-ready stat arb.
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class EnhancedRiskManager:
    """
    Comprehensive risk management framework for statistical arbitrage strategies.
    
    Features:
    - Position sizing based on volatility and expected risk
    - Dynamic stop losses and profit targets
    - Exposure monitoring across sectors and asset classes
    - Correlation-based risk estimates
    - Drawdown controls and circuit breakers
    - Concentration limits
    """
    
    def __init__(self, max_portfolio_var=0.02, max_position_var=0.005,
                max_sector_exposure=0.3, max_drawdown=-0.15,
                circuit_breaker_loss=-0.05, max_correlation=0.7,
                max_concentration=0.2, target_leverage=1.0):
        """
        Initialize the risk manager.
        
        Parameters:
        -----------
        max_portfolio_var : float
            Maximum portfolio daily VaR (95%)
        max_position_var : float
            Maximum single position VaR contribution
        max_sector_exposure : float
            Maximum exposure to any sector
        max_drawdown : float
            Maximum allowable strategy drawdown
        circuit_breaker_loss : float
            Daily loss that triggers circuit breaker
        max_correlation : float
            Maximum average correlation between positions
        max_concentration : float
            Maximum allocation to any single position
        target_leverage : float
            Target leverage for the strategy
        """
        self.max_portfolio_var = max_portfolio_var
        self.max_position_var = max_position_var
        self.max_sector_exposure = max_sector_exposure
        self.max_drawdown = max_drawdown
        self.circuit_breaker_loss = circuit_breaker_loss
        self.max_correlation = max_correlation
        self.max_concentration = max_concentration
        self.target_leverage = target_leverage
        
        # Track exposures and positions
        self.positions = {}
        self.sector_exposures = {}
        self.portfolio_drawdown = 0.0
        self.portfolio_var = 0.0
        self.active_circuit_breaker = False
        
        # Historical tracking
        self.historical_var = []
        self.historical_exposures = []
    
    def size_position(self, pair, prices, volatilities, correlation, hedge_ratio,
                     max_sizing_factor=3.0, min_sizing_factor=0.2,
                     base_position=0.1, sector_map=None):
        """
        Determine optimal position size based on risk.
        
        Parameters:
        -----------
        pair : tuple
            Tuple containing ticker symbols
        prices : dict
            Dictionary with current prices
        volatilities : dict
            Dictionary with volatility estimates
        correlation : float
            Correlation between the two assets
        hedge_ratio : float
            Current hedge ratio
        max_sizing_factor : float
            Maximum multiplier for position sizing
        min_sizing_factor : float
            Minimum multiplier for position sizing
        base_position : float
            Base position size as fraction of portfolio
        sector_map : dict, optional
            Dictionary mapping tickers to sectors
            
        Returns:
        --------
        Dictionary with position sizes and risk metrics
        """
        stock1, stock2 = pair
        
        # Default sizes
        pair_value = base_position
        stock1_size = 0.0
        stock2_size = 0.0
        
        # Check if we have necessary data
        if not all(x in prices for x in pair) or not all(x in volatilities for x in pair):
            logger.warning(f"Missing price or volatility data for {pair}")
            return {
                'pair_value': 0.0,
                'stock1_size': 0.0,
                'stock2_size': 0.0,
                'pair_var': 0.0,
                'position_allowed': False,
                'reason': "Missing data"
            }
        
        # Get prices and volatilities
        price1 = prices[stock1]
        price2 = prices[stock2]
        vol1 = volatilities[stock1]
        vol2 = volatilities[stock2]
        
        # Calculate combined volatility with correlation effect
        combined_vol = np.sqrt(vol1**2 + vol2**2 * hedge_ratio**2 - 
                              2 * correlation * vol1 * vol2 * hedge_ratio)
        
        # Scale by inverse of volatility, bounded by min/max factors
        vol_scaling = min(max(1.0 / combined_vol, min_sizing_factor), max_sizing_factor)
        
        # Adjust base position by volatility scaling
        pair_value = base_position * vol_scaling
        
        # Calculate stock-specific sizes
        total_pair_value = pair_value
        
        # Stock1 position (long)
        stock1_value = total_pair_value / (1 + hedge_ratio)
        stock1_size = stock1_value / price1
        
        # Stock2 position (short, hedge ratio adjusted)
        stock2_value = -stock1_value * hedge_ratio  # Negative for short
        stock2_size = stock2_value / price2
        
        # Calculate pair VaR (95% daily)
        pair_var = 1.65 * combined_vol * total_pair_value
        
        # Check sector exposure limits if sector map provided
        sector_limit_exceeded = False
        if sector_map:
            # Get sectors
            sector1 = sector_map.get(stock1, {}).get('sector', 'Unknown')
            sector2 = sector_map.get(stock2, {}).get('sector', 'Unknown')
            
            # Calculate current sector exposures
            current_sector_exp = self.sector_exposures.copy()
            
            # Add new position's contribution
            current_sector_exp[sector1] = current_sector_exp.get(sector1, 0) + abs(stock1_value)
            current_sector_exp[sector2] = current_sector_exp.get(sector2, 0) + abs(stock2_value)
            
            # Check if any sector exceeds limit
            total_exposure = sum(abs(v) for v in self.positions.values())
            total_new_exposure = total_exposure + abs(stock1_value) + abs(stock2_value)
            
            for sector, exposure in current_sector_exp.items():
                if exposure / total_new_exposure > self.max_sector_exposure:
                    sector_limit_exceeded = True
                    logger.warning(f"Sector limit exceeded for {sector}: "
                                 f"{exposure/total_new_exposure:.2%} > {self.max_sector_exposure:.2%}")
                    break
        
        # Calculate total portfolio VaR with this position
        current_total_var = self.portfolio_var
        new_total_var = np.sqrt(current_total_var**2 + pair_var**2)  # Simple approximation
        
        # Determine if position is allowed based on risk limits
        position_allowed = (
            (pair_var <= self.max_position_var) and  # Position VaR limit
            (new_total_var <= self.max_portfolio_var) and  # Portfolio VaR limit
            (not sector_limit_exceeded) and  # Sector exposure limit
            (not self.active_circuit_breaker)  # No active circuit breaker
        )
        
        reason = ""
        if pair_var > self.max_position_var:
            reason = "Position VaR limit exceeded"
        elif new_total_var > self.max_portfolio_var:
            reason = "Portfolio VaR limit exceeded"
        elif sector_limit_exceeded:
            reason = "Sector exposure limit exceeded"
        elif self.active_circuit_breaker:
            reason = "Circuit breaker active"
            
        # Track potential new portfolio VaR
        if position_allowed:
            potential_var = new_total_var
        else:
            potential_var = current_total_var
        
        return {
            'pair_value': pair_value,
            'stock1_size': stock1_size,
            'stock2_size': stock2_size,
            'pair_var': pair_var,
            'portfolio_var': potential_var,
            'position_allowed': position_allowed,
            'reason': reason
        }
    
    def update_portfolio_risk(self, current_positions, prices, returns, sector_map=None,
                           correlation_matrix=None):
        """
        Update portfolio risk metrics based on current positions.
        
        Parameters:
        -----------
        current_positions : dict
            Dictionary with current positions (ticker -> quantity)
        prices : dict
            Dictionary with current prices
        returns : DataFrame
            DataFrame with historical returns
        sector_map : dict, optional
            Dictionary mapping tickers to sectors
        correlation_matrix : DataFrame, optional
            Pre-calculated correlation matrix
            
        Returns:
        --------
        Dictionary with updated risk metrics
        """
        # Store current positions
        self.positions = current_positions.copy()
        
        # Calculate position values
        position_values = {}
        for ticker, quantity in current_positions.items():
            if ticker in prices:
                position_values[ticker] = quantity * prices[ticker]
        
        total_gross_exposure = sum(abs(v) for v in position_values.values())
        total_net_exposure = sum(v for v in position_values.values())
        
        # Calculate sector exposures if sector map provided
        if sector_map:
            self.sector_exposures = {}
            
            for ticker, value in position_values.items():
                sector = sector_map.get(ticker, {}).get('sector', 'Unknown')
                self.sector_exposures[sector] = self.sector_exposures.get(sector, 0) + abs(value)
            
            # Calculate sector exposure percentages
            sector_percentages = {s: e/total_gross_exposure for s, e in self.sector_exposures.items()}
            max_sector = max(sector_percentages.items(), key=lambda x: x[1]) if sector_percentages else ('None', 0)
            
            logger.info(f"Sector exposures: {sector_percentages}")
            logger.info(f"Largest sector: {max_sector[0]} at {max_sector[1]:.2%}")
        else:
            sector_percentages = {}
            max_sector = ('Unknown', 0)
        
        # Calculate concentration
        tickers = list(position_values.keys())
        concentration = {}
        
        for ticker, value in position_values.items():
            concentration[ticker] = abs(value) / total_gross_exposure if total_gross_exposure > 0 else 0
            
        max_concentration = max(concentration.values()) if concentration else 0
        
        # Calculate portfolio VaR
        if returns is not None and len(tickers) > 0:
            # Get returns for positions
            position_returns = returns[tickers].fillna(0)
            
            # Calculate correlation matrix if not provided
            if correlation_matrix is None:
                correlation_matrix = position_returns.corr()
            
            # Calculate volatilities
            volatilities = position_returns.std() * np.sqrt(252)
            
            # Create weight vector (adjusted for shorts)
            weights = np.array([position_values.get(t, 0) / total_gross_exposure 
                              if total_gross_exposure > 0 else 0 
                              for t in tickers])
            
            # Calculate portfolio volatility
            port_variance = 0.0
            
            for i in range(len(tickers)):
                for j in range(len(tickers)):
                    if tickers[i] in volatilities.index and tickers[j] in volatilities.index:
                        port_variance += (weights[i] * weights[j] * 
                                         volatilities[tickers[i]] * volatilities[tickers[j]] * 
                                         correlation_matrix.loc[tickers[i], tickers[j]])
            
            portfolio_vol = np.sqrt(port_variance) if port_variance > 0 else 0
            
            # VaR calculation (95% daily)
            self.portfolio_var = 1.65 * portfolio_vol * total_gross_exposure
            
            # Average correlation
            avg_corr = correlation_matrix.values.sum() / (len(tickers)**2) if len(tickers) > 0 else 0
            
            logger.info(f"Portfolio vol: {portfolio_vol:.2%}, VaR: {self.portfolio_var:.2f}, "
                       f"Avg correlation: {avg_corr:.2f}")
        else:
            self.portfolio_var = 0
            avg_corr = 0
        
        # Update historical tracking
        self.historical_var.append(self.portfolio_var)
        
        # Return current risk metrics
        return {
            'gross_exposure': total_gross_exposure,
            'net_exposure': total_net_exposure,
            'net_leverage': total_net_exposure / total_gross_exposure if total_gross_exposure > 0 else 0,
            'gross_leverage': total_gross_exposure / total_gross_exposure if total_gross_exposure > 0 else 1,
            'sector_exposures': sector_percentages,
            'max_sector_exposure': max_sector,
            'concentration': concentration,
            'max_concentration': max_concentration,
            'portfolio_var': self.portfolio_var,
            'average_correlation': avg_corr
        }
    
    def check_circuit_breakers(self, daily_pnl, equity_curve):
        """
        Check if any circuit breakers should be triggered.
        
        Parameters:
        -----------
        daily_pnl : float
            Today's P&L as a percentage
        equity_curve : Series
            Equity curve of the strategy
            
        Returns:
        --------
        Dictionary with circuit breaker status
        """
        # Check daily loss circuit breaker
        daily_breaker_triggered = daily_pnl <= self.circuit_breaker_loss
        
        # Calculate current drawdown
        peak = equity_curve.cummax()
        current_dd = (equity_curve.iloc[-1] / peak.iloc[-1]) - 1
        
        # Check drawdown circuit breaker
        dd_breaker_triggered = current_dd <= self.max_drawdown
        
        # Update circuit breaker status
        self.active_circuit_breaker = daily_breaker_triggered or dd_breaker_triggered
        self.portfolio_drawdown = current_dd
        
        # Define recovery threshold (when to disable circuit breaker)
        recovery_threshold = 0.5  # Only disable after recovering 50% of drawdown
        
        if self.active_circuit_breaker and current_dd > self.max_drawdown * recovery_threshold:
            self.active_circuit_breaker = False
            logger.info(f"Circuit breaker deactivated after partial recovery: "
                       f"current DD {current_dd:.2%} vs. threshold {self.max_drawdown:.2%}")
        
        if self.active_circuit_breaker:
            reason = "Daily loss limit" if daily_breaker_triggered else "Drawdown limit"
            logger.warning(f"Circuit breaker triggered: {reason}. "
                          f"Daily PnL: {daily_pnl:.2%}, Drawdown: {current_dd:.2%}")
        
        return {
            'daily_breaker': daily_breaker_triggered,
            'drawdown_breaker': dd_breaker_triggered,
            'active_breaker': self.active_circuit_breaker,
            'current_drawdown': current_dd,
            'daily_pnl': daily_pnl
        }
    
    def apply_risk_controls(self, signal_df, price_data, pair, hedge_ratios=None,
                          volatilities=None, correlation=None, sector_map=None):
        """
        Apply risk controls to signals.
        
        Parameters:
        -----------
        signal_df : DataFrame
            DataFrame with trading signals
        price_data : DataFrame
            DataFrame with price data
        pair : tuple
            Tuple containing ticker symbols
        hedge_ratios : Series, optional
            Series with hedge ratios
        volatilities : dict, optional
            Dictionary with volatility estimates
        correlation : float, optional
            Correlation between the two assets
        sector_map : dict, optional
            Dictionary mapping tickers to sectors
            
        Returns:
        --------
        DataFrame with risk-adjusted signals
        """
        stock1, stock2 = pair
        
        # Make sure we have required columns
        required_cols = ['signal']
        if not all(col in signal_df.columns for col in required_cols):
            logger.error(f"Signal DataFrame missing required columns.")
            return signal_df
        
        # Create a copy to avoid modifying the original
        risk_df = signal_df.copy()
        
        # Add columns for risk management
        risk_df['risk_signal'] = risk_df['signal']
        risk_df['position_allowed'] = True
        risk_df['position_size'] = 0.0
        
        # Calculate default volatilities if not provided
        if volatilities is None:
            volatilities = {
                stock1: price_data[stock1].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252),
                stock2: price_data[stock2].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
            }
        
        # Calculate default correlation if not provided
        if correlation is None:
            correlation = price_data[stock1].pct_change().corr(price_data[stock2].pct_change())
        
        # Apply risk controls to each day
        for i in range(1, len(risk_df)):
            curr_idx = risk_df.index[i]
            prev_idx = risk_df.index[i-1]
            
            prev_signal = risk_df['risk_signal'].iloc[i-1]
            curr_signal = risk_df['signal'].iloc[i]  # Raw signal
            
            # Get current prices
            prices = {
                stock1: price_data[stock1].loc[curr_idx],
                stock2: price_data[stock2].loc[curr_idx]
            }
            
            # Get current hedge ratio
            if hedge_ratios is not None and curr_idx in hedge_ratios.index:
                hedge_ratio = hedge_ratios.loc[curr_idx]
            else:
                # Use a default ratio
                hedge_ratio = 1.0
            
            # Only check risk limits when opening a new position or changing direction
            if curr_signal != 0 and (prev_signal == 0 or prev_signal != curr_signal):
                # Size the position based on risk
                sizing = self.size_position(
                    pair, prices, volatilities, correlation, hedge_ratio,
                    max_sizing_factor=3.0, min_sizing_factor=0.2,
                    base_position=0.1, sector_map=sector_map
                )
                
                # Store sizing results
                risk_df.loc[curr_idx, 'position_allowed'] = sizing['position_allowed']
                
                if sizing['position_allowed']:
                    # Allow the position
                    risk_df.loc[curr_idx, 'risk_signal'] = curr_signal
                    risk_df.loc[curr_idx, 'position_size'] = sizing['pair_value']
                else:
                    # Block the position
                    risk_df.loc[curr_idx, 'risk_signal'] = 0
                    risk_df.loc[curr_idx, 'position_size'] = 0.0
                    logger.warning(f"Position blocked at {curr_idx}: {sizing['reason']}")
            
            elif curr_signal == 0 and prev_signal != 0:
                # Exit signal
                risk_df.loc[curr_idx, 'risk_signal'] = 0
                risk_df.loc[curr_idx, 'position_size'] = 0.0
            
            elif prev_signal != 0:
                # Maintain position
                risk_df.loc[curr_idx, 'risk_signal'] = prev_signal
                risk_df.loc[curr_idx, 'position_size'] = risk_df.loc[prev_idx, 'position_size']
        
        return risk_df


def enhance_risk_management(strategy_results, max_drawdown=-0.15, max_concentration=0.2,
                         max_sector_exposure=0.3, max_leverage=1.5):
    """
    Apply enhanced risk management to strategy results.
    
    Parameters:
    -----------
    strategy_results : dict
        Dictionary with strategy results
    max_drawdown : float
        Maximum allowable drawdown
    max_concentration : float
        Maximum allocation to any single position
    max_sector_exposure : float
        Maximum exposure to any single sector
    max_leverage : float
        Maximum gross leverage
        
    Returns:
    --------
    Updated strategy results
    """
    logger.info(f"Applying enhanced risk management: "
               f"max_drawdown={max_drawdown:.2%}, "
               f"max_concentration={max_concentration:.2%}, "
               f"max_sector_exposure={max_sector_exposure:.2%}, "
               f"max_leverage={max_leverage:.1f}x")
    
    # Track active positions and exposures
    active_positions = {}
    sector_exposures = {}
    peak_capital = 1.0
    
    # Initialize enhanced results
    enhanced_results = {}
    
    # Process each pair's results
    for pair_str, results in strategy_results.items():
        if pair_str == 'aggregate' or pair_str == 'portfolio':
            continue
            
        pair = eval(pair_str)  # Convert string tuple to actual tuple
        
        # Skip if missing required data
        if 'portfolio' not in results or 'trades' not in results:
            enhanced_results[pair_str] = results
            continue
        
        # Create enhanced result entry
        enhanced_results[pair_str] = {
            'original_portfolio': results['portfolio'].copy(),
            'trades': []
        }
        
        # Initialize enhanced portfolio
        enhanced_portfolio = pd.Series(index=results['portfolio'].index, data=results['portfolio'].iloc[0])
        
        # Track pair position
        position_active = False
        entry_size = 0.0
        
        # Process trades with risk constraints
        for trade in results['trades']:
            # Skip if not a properly formatted trade
            if 'action' not in trade or 'date' not in trade:
                continue
                
            # Get basic trade info
            date = trade['date']
            action = trade['action']
            
            # Create a copy of the trade for enhanced results
            enhanced_trade = trade.copy()
            enhanced_trade['blocked'] = False
            enhanced_trade['reason'] = ""
            
            # For trades that open positions
            if action == 'open' or action == 'immediate':
                # Get position details
                stock1 = pair[0]
                stock2 = pair[1]
                signal = trade.get('signal', 0)
                stock1_qty = trade.get('stock1_qty', 0)
                stock2_qty = trade.get('stock2_qty', 0)
                stock1_price = trade.get('stock1_price', 0)
                stock2_price = trade.get('stock2_price', 0)
                
                # Calculate position values
                stock1_value = stock1_qty * stock1_price
                stock2_value = stock2_qty * stock2_price
                position_value = abs(stock1_value) + abs(stock2_value)
                
                # Check current equity and drawdown
                current_idx = enhanced_portfolio.index.get_loc(date)
                current_equity = enhanced_portfolio.iloc[current_idx]
                peak_capital = max(peak_capital, current_equity)
                current_drawdown = (current_equity / peak_capital) - 1
                
                # Block trades during excessive drawdowns
                if current_drawdown <= max_drawdown:
                    enhanced_trade['blocked'] = True
                    enhanced_trade['reason'] = f"Excessive drawdown: {current_drawdown:.2%}"
                    logger.warning(f"Blocked trade at {date} due to drawdown: {current_drawdown:.2%}")
                    continue
                
                # Calculate current exposure
                total_exposure = sum(abs(v) for v in active_positions.values())
                
                # Check concentration
                new_concentration = position_value / (total_exposure + position_value) if signal != 0 else 0
                
                if new_concentration > max_concentration:
                    enhanced_trade['blocked'] = True
                    enhanced_trade['reason'] = f"Concentration limit: {new_concentration:.2%}"
                    logger.warning(f"Blocked trade at {date} due to concentration: {new_concentration:.2%}")
                    continue
                
                # Check leverage
                new_leverage = (total_exposure + position_value) / current_equity if current_equity > 0 else 0
                
                if new_leverage > max_leverage:
                    enhanced_trade['blocked'] = True
                    enhanced_trade['reason'] = f"Leverage limit: {new_leverage:.2f}x"
                    logger.warning(f"Blocked trade at {date} due to leverage: {new_leverage:.2f}x")
                    continue
                
                # Allow the trade
                if not enhanced_trade['blocked']:
                    # Update active positions
                    active_positions[stock1] = stock1_value
                    active_positions[stock2] = stock2_value
                    
                    # Record entry size for calculating P&L
                    position_active = True
                    entry_size = position_value
            
            # For trades that close positions
            elif (action == 'close' or action == 'execution') and position_active:
                # Clear position tracking
                active_positions = {}
                position_active = False
                
                # Calculate P&L if possible
                if 'pnl' not in enhanced_trade and entry_size > 0:
                    # Simple P&L calculation based on original trade
                    exit_idx = enhanced_portfolio.index.get_loc(date)
                    entry_idx = exit_idx - 1  # Approximate
                    
                    if entry_idx >= 0:
                        pnl_pct = (results['portfolio'].iloc[exit_idx] / results['portfolio'].iloc[entry_idx]) - 1
                        enhanced_trade['pnl'] = pnl_pct * entry_size
            
            # Add trade to enhanced results
            enhanced_results[pair_str]['trades'].append(enhanced_trade)
            
            # Update portfolio values based on whether trade was blocked
            idx = enhanced_portfolio.index.get_loc(date)
            
            if enhanced_trade['blocked']:
                # If blocked, just carry forward previous value
                if idx > 0:
                    enhanced_portfolio.iloc[idx] = enhanced_portfolio.iloc[idx-1]
            else:
                # Otherwise, update with returns from original portfolio
                if idx > 0:
                    original_return = results['portfolio'].iloc[idx] / results['portfolio'].iloc[idx-1] - 1
                    enhanced_portfolio.iloc[idx] = enhanced_portfolio.iloc[idx-1] * (1 + original_return)
        
        # Store enhanced portfolio
        enhanced_results[pair_str]['portfolio'] = enhanced_portfolio
        
        # Calculate metrics
        returns = enhanced_portfolio.pct_change().fillna(0)
        total_return = (enhanced_portfolio.iloc[-1] / enhanced_portfolio.iloc[0]) - 1
        annual_return = ((1 + total_return) ** (252 / len(enhanced_portfolio))) - 1
        annual_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Calculate drawdown
        running_max = enhanced_portfolio.cummax()
        drawdown = (enhanced_portfolio / running_max) - 1
        max_dd = drawdown.min()
        
        # Store metrics
        enhanced_results[pair_str]['metrics'] = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'calmar_ratio': -annual_return / max_dd if max_dd != 0 else float('inf'),
            'win_rate': (returns > 0).mean()
        }
        
        # Log comparison
        logger.info(f"Pair {pair_str} with risk management: "
                   f"Return {total_return:.2%} vs original {results['metrics']['total_return']:.2%}, "
                   f"MaxDD {max_dd:.2%} vs original {results['metrics']['max_drawdown']:.2%}")
    
    # Calculate aggregate results
    aggregate_portfolio = None
    
    for pair_str, results in enhanced_results.items():
        if pair_str != 'aggregate' and pair_str != 'portfolio' and 'portfolio' in results:
            weight = 1.0 / (len(enhanced_results) - 1)  # Equal weight, excluding aggregate
            
            if aggregate_portfolio is None:
                aggregate_portfolio = results['portfolio'] * weight
            else:
                # Align dates
                common_idx = aggregate_portfolio.index.intersection(results['portfolio'].index)
                aggregate_portfolio = aggregate_portfolio.loc[common_idx]
                pair_contribution = results['portfolio'].loc[common_idx] * weight
                
                aggregate_portfolio = aggregate_portfolio + pair_contribution
    
    if aggregate_portfolio is not None:
        # Calculate aggregate metrics
        agg_returns = aggregate_portfolio.pct_change().fillna(0)
        total_return = (aggregate_portfolio.iloc[-1] / aggregate_portfolio.iloc[0]) - 1
        annual_return = ((1 + total_return) ** (252 / len(aggregate_portfolio))) - 1
        annual_volatility = agg_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Calculate drawdown
        running_max = aggregate_portfolio.cummax()
        drawdown = (aggregate_portfolio / running_max) - 1
        max_dd = drawdown.min()
        
        # Store aggregate results
        enhanced_results['aggregate'] = {
            'portfolio': aggregate_portfolio,
            'returns': agg_returns,
            'metrics': {
                'total_return': total_return,
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_dd,
                'calmar_ratio': -annual_return / max_dd if max_dd != 0 else float('inf'),
                'win_rate': (agg_returns > 0).mean()
            }
        }
        
        # Log comparison to original
        original_metrics = strategy_results.get('aggregate', {}).get('metrics', {})
        original_return = original_metrics.get('total_return', 0)
        original_dd = original_metrics.get('max_drawdown', 0)
        
        logger.info(f"Aggregate with risk management: "
                   f"Return {total_return:.2%} vs original {original_return:.2%}, "
                   f"MaxDD {max_dd:.2%} vs original {original_dd:.2%}")
    
    return enhanced_results
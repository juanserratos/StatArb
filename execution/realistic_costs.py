"""
Enhanced, realistic transaction cost modeling for statistical arbitrage.
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RealisticCostModel:
    """
    A realistic transaction cost model that incorporates:
    - Bid-ask spread scaling with volatility
    - Market impact that scales with order size and ADV
    - Opportunity cost for multi-day execution
    - Borrow costs for short positions
    - Trading restrictions for hard-to-borrow securities
    """

    def __init__(self, base_spread_bps=2.0, base_commission_bps=1.0,
                impact_factor=0.1, opportunity_cost_factor=0.05,
                borrow_cost_bps=15.0, min_adv_participation=0.05,
                max_adv_participation=0.2, etf_liquidity_boost=5.0):
        """
        Initialize the realistic cost model.

        Parameters:
        -----------
        base_spread_bps : float
            Base bid-ask spread in basis points, before volatility scaling
        base_commission_bps : float
            Base commission in basis points
        impact_factor : float
            Market impact factor (higher for less liquid securities)
        opportunity_cost_factor : float
            Opportunity cost factor for multi-day execution
        borrow_cost_bps : float
            Daily borrow cost in basis points for short positions
        min_adv_participation : float
            Minimum participation rate as fraction of daily volume
        max_adv_participation : float
            Maximum participation rate as fraction of daily volume
        etf_liquidity_boost : float
            Multiplier for ETF liquidity vs individual stocks
        """
        self.base_spread_bps = base_spread_bps
        self.base_commission_bps = base_commission_bps
        self.impact_factor = impact_factor
        self.opportunity_cost_factor = opportunity_cost_factor
        self.borrow_cost_bps = borrow_cost_bps
        self.min_adv_participation = min_adv_participation
        self.max_adv_participation = max_adv_participation
        self.etf_liquidity_boost = etf_liquidity_boost
        self.htb_list = set()  # Hard-to-borrow securities
        self.adv_estimates = {}  # Cache for ADV estimates

    def estimate_adv(self, ticker, price, volatility, is_etf=False):
        """
        Estimate average daily volume in dollars.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
        price : float
            Current price
        volatility : float
            Annualized volatility
        is_etf : bool
            Whether the security is an ETF
            
        Returns:
        --------
        Estimated ADV in dollars
        """
        # Use cached value if available
        if ticker in self.adv_estimates:
            return self.adv_estimates[ticker]
            
        # In production, this would use actual volume data
        # For now, use a model based on price and volatility
        
        # Base ADV estimate: higher price and volatility generally means higher dollar volume
        base_adv = price * 1e6 * (max(volatility, 0.05) / 0.20)
        
        # ETFs generally have higher liquidity
        if is_etf or ticker.startswith('X') or ticker in ['SPY', 'QQQ', 'IWM', 'DIA']:
            base_adv *= self.etf_liquidity_boost
            
        # Add some realistic constraints
        min_adv = 1e6  # Minimum $1M daily volume
        max_adv = 1e9  # Maximum $1B daily volume
        
        estimated_adv = max(min(base_adv, max_adv), min_adv)
        
        # Cache the result
        self.adv_estimates[ticker] = estimated_adv
        
        return estimated_adv
    
    def calculate_spread_cost(self, ticker, price, volatility, trade_size, is_etf=False):
        """
        Calculate spread costs that scale with volatility.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
        price : float
            Current price
        volatility : float
            Annualized volatility
        trade_size : float
            Trade size in dollars
        is_etf : bool
            Whether the security is an ETF
            
        Returns:
        --------
        Spread cost in dollars
        """
        # Scale spread with volatility - higher vol means wider spreads
        # Normalize around 20% annualized vol
        vol_ratio = volatility / 0.20
        
        # ETFs typically have tighter spreads
        spread_factor = 1.0
        if is_etf or ticker.startswith('X') or ticker in ['SPY', 'QQQ', 'IWM', 'DIA']:
            spread_factor = 0.5
            
        # Calculate spread in basis points, with a floor
        spread_bps = max(self.base_spread_bps * vol_ratio * spread_factor, 1.0)
        
        # Calculate cost
        spread_cost = trade_size * (spread_bps / 10000.0)
        
        return spread_cost
    
    def calculate_market_impact(self, ticker, price, volatility, trade_size, is_etf=False):
        """
        Calculate market impact costs using square root model.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
        price : float
            Current price
        volatility : float
            Annualized volatility
        trade_size : float
            Trade size in dollars
        is_etf : bool
            Whether the security is an ETF
            
        Returns:
        --------
        Market impact cost in dollars
        """
        # Estimate ADV
        adv = self.estimate_adv(ticker, price, volatility, is_etf)
        
        # Calculate participation rate
        participation = trade_size / adv
        
        # If participation exceeds limit, assume multi-day execution
        execution_days = 1
        if participation > self.max_adv_participation:
            execution_days = int(np.ceil(participation / self.max_adv_participation))
            participation = participation / execution_days
        
        # Calculate market impact using square root model
        # Higher participation and volatility lead to higher impact
        impact_bps = self.impact_factor * volatility * 100 * np.sqrt(participation)
        
        # ETFs typically have lower impact
        if is_etf:
            impact_bps *= 0.7
            
        # Calculate cost
        impact_cost = trade_size * (impact_bps / 10000.0)
        
        # If multi-day execution, add opportunity cost
        if execution_days > 1:
            opp_cost = trade_size * volatility * self.opportunity_cost_factor * np.sqrt(execution_days / 252.0)
            impact_cost += opp_cost
            
        return impact_cost, execution_days
    
    def calculate_borrowing_cost(self, ticker, trade_size, is_short, holding_days=1):
        """
        Calculate borrowing costs for short positions.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
        trade_size : float
            Trade size in dollars (absolute value)
        is_short : bool
            Whether the position is short
        holding_days : int
            Expected holding period in days
            
        Returns:
        --------
        Borrowing cost in dollars
        """
        if not is_short:
            return 0.0
            
        # Hard-to-borrow securities have higher costs
        borrow_multiple = 5.0 if ticker in self.htb_list else 1.0
        
        # Daily cost
        daily_cost = trade_size * (self.borrow_cost_bps * borrow_multiple / 10000.0) / 252.0
        
        # Total cost for holding period
        total_cost = daily_cost * holding_days
        
        return total_cost
    
    def calculate_total_cost(self, ticker, price, volatility, trade_size, is_short=False, 
                            holding_days=10, is_etf=False):
        """
        Calculate total transaction costs.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
        price : float
            Current price
        volatility : float
            Annualized volatility
        trade_size : float
            Trade size in dollars (absolute value)
        is_short : bool
            Whether the position is short
        holding_days : int
            Expected holding period in days
        is_etf : bool
            Whether the security is an ETF
            
        Returns:
        --------
        Dictionary with cost components
        """
        # Calculate commission
        commission = trade_size * (self.base_commission_bps / 10000.0)
        
        # Calculate spread cost
        spread_cost = self.calculate_spread_cost(ticker, price, volatility, trade_size, is_etf)
        
        # Calculate market impact
        impact_cost, execution_days = self.calculate_market_impact(ticker, price, volatility, 
                                                                  trade_size, is_etf)
        
        # Calculate borrowing cost
        borrow_cost = self.calculate_borrowing_cost(ticker, trade_size, is_short, holding_days)
        
        # Total cost
        total_cost = commission + spread_cost + impact_cost + borrow_cost
        
        # Cost as percentage of trade value
        cost_bps = (total_cost / trade_size) * 10000.0 if trade_size > 0 else 0.0
        
        return {
            'commission': commission,
            'spread_cost': spread_cost,
            'impact_cost': impact_cost,
            'borrow_cost': borrow_cost,
            'total_cost': total_cost,
            'cost_bps': cost_bps,
            'execution_days': execution_days
        }


def apply_realistic_costs_to_backtest(backtest_results, cost_model=None):
    """
    Apply realistic costs to backtest results.
    
    Parameters:
    -----------
    backtest_results : dict
        Dictionary with backtest results
    cost_model : RealisticCostModel, optional
        Custom cost model to use
        
    Returns:
    --------
    Updated backtest results
    """
    if cost_model is None:
        cost_model = RealisticCostModel()
        
    # Process each pair's results
    for pair_str, results in backtest_results.items():
        if pair_str == 'aggregate' or pair_str == 'portfolio':
            continue
            
        # Get pair details
        pair = eval(pair_str)  # Convert string tuple to actual tuple
        
        # Extract necessary data
        if 'trades' not in results or 'portfolio' not in results:
            continue
            
        trades = results['trades']
        portfolio = results['portfolio'].copy()
        
        # Track total costs
        total_costs = 0.0
        original_costs = 0.0
        
        # Apply realistic costs to each trade
        for trade in trades:
            # Skip if not a trade
            if 'action' not in trade or 'stock1_qty' not in trade:
                continue
                
            # Get trade details
            date = trade['date']
            action = trade['action']
            stock1 = pair[0]
            stock2 = pair[1]
            stock1_qty = trade['stock1_qty']
            stock2_qty = trade['stock2_qty']
            stock1_price = trade['stock1_price']
            stock2_price = trade['stock2_price']
            
            # Determine if ETFs
            is_etf1 = stock1.startswith('X') or stock1 in ['SPY', 'QQQ', 'IWM', 'DIA']
            is_etf2 = stock2.startswith('X') or stock2 in ['SPY', 'QQQ', 'IWM', 'DIA']
            
            # Calculate trade sizes
            stock1_value = abs(stock1_qty * stock1_price)
            stock2_value = abs(stock2_qty * stock2_price)
            
            # Estimate volatilities (in production, would use actual data)
            # Just use fixed values here for simplicity
            stock1_vol = 0.25  # 25% annualized vol
            stock2_vol = 0.25  # 25% annualized vol
            
            # Determine if short positions
            is_short1 = stock1_qty < 0
            is_short2 = stock2_qty < 0
            
            # Store original cost
            original_cost = trade.get('cost', 0.0)
            original_costs += original_cost
            
            # Calculate realistic costs
            stock1_costs = cost_model.calculate_total_cost(
                stock1, stock1_price, stock1_vol, stock1_value, 
                is_short1, 10, is_etf1
            )
            
            stock2_costs = cost_model.calculate_total_cost(
                stock2, stock2_price, stock2_vol, stock2_value, 
                is_short2, 10, is_etf2
            )
            
            # Total realistic costs
            realistic_cost = stock1_costs['total_cost'] + stock2_costs['total_cost']
            
            # Update trade with new cost
            trade['original_cost'] = original_cost
            trade['realistic_cost'] = realistic_cost
            trade['cost'] = realistic_cost
            
            # Track total costs
            total_costs += realistic_cost
        
        # Calculate cost ratio
        cost_ratio = total_costs / original_costs if original_costs > 0 else 1.0
        
        logger.info(f"Realistic costs for {pair_str}: " +
                   f"Original=${original_costs:.2f}, Realistic=${total_costs:.2f}, " +
                   f"Ratio={cost_ratio:.2f}x")
        
        # Adjust portfolio values to account for increased costs
        # This is a simplified approach - in reality would need to recalculate trades
        if cost_ratio > 1.0:
            # Adjust returns by the cost ratio difference
            returns = portfolio.pct_change().fillna(0)
            scaled_returns = returns - (returns / cost_ratio)
            
            # Reconstruct portfolio values
            adjusted_portfolio = [portfolio.iloc[0]]
            for i in range(1, len(portfolio)):
                new_value = adjusted_portfolio[-1] * (1 + scaled_returns.iloc[i])
                adjusted_portfolio.append(new_value)
                
            results['portfolio'] = pd.Series(adjusted_portfolio, index=portfolio.index)
            
            # Update metrics
            if 'metrics' in results:
                total_return = (results['portfolio'].iloc[-1] / results['portfolio'].iloc[0]) - 1
                annual_return = ((1 + total_return) ** (252 / len(results['portfolio']))) - 1
                
                results['metrics']['total_return'] = total_return
                results['metrics']['annual_return'] = annual_return
                
                # Recalculate other metrics as needed
    
    # Recalculate aggregate results if needed
    if 'aggregate' in backtest_results:
        aggregate_portfolio = None
        
        for pair_str, results in backtest_results.items():
            if pair_str != 'aggregate' and pair_str != 'portfolio' and 'portfolio' in results:
                weight = 1.0 / (len(backtest_results) - 1)  # Equal weight, excluding aggregate
                
                if aggregate_portfolio is None:
                    aggregate_portfolio = results['portfolio'] * weight
                else:
                    # Align dates
                    common_idx = aggregate_portfolio.index.intersection(results['portfolio'].index)
                    aggregate_portfolio = aggregate_portfolio.loc[common_idx]
                    pair_contribution = results['portfolio'].loc[common_idx] * weight
                    
                    aggregate_portfolio = aggregate_portfolio + pair_contribution
        
        if aggregate_portfolio is not None:
            backtest_results['aggregate']['portfolio'] = aggregate_portfolio
            
            # Update metrics
            returns = aggregate_portfolio.pct_change().fillna(0)
            total_return = (aggregate_portfolio.iloc[-1] / aggregate_portfolio.iloc[0]) - 1
            annual_return = ((1 + total_return) ** (252 / len(aggregate_portfolio))) - 1
            annual_volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
            
            # Calculate drawdown
            running_max = aggregate_portfolio.cummax()
            drawdown = (aggregate_portfolio / running_max) - 1
            max_drawdown = drawdown.min()
            
            # Update metrics
            backtest_results['aggregate']['metrics'] = {
                'total_return': total_return,
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': -annual_return / max_drawdown if max_drawdown != 0 else float('inf'),
                'win_rate': (returns > 0).mean()
            }
    
    return backtest_results
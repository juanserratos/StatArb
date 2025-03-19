"""
Transaction cost modeling for the statistical arbitrage strategy.
"""
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def model_transaction_costs(pair, position_sizes, price_data, 
                          spread_bps=2, commission_bps=1, market_impact_factor=0.1,
                          min_commission=1.0, strategy_aum=10000000):
    """
    Model realistic transaction costs including spread, commission, and market impact.
    
    Parameters:
    -----------
    pair : tuple
        Tuple containing the ticker symbols of the pair
    position_sizes : DataFrame
        DataFrame with position sizes 
    price_data : DataFrame
        DataFrame with price series
    spread_bps : float
        Bid-ask spread in basis points
    commission_bps : float
        Commission in basis points
    market_impact_factor : float
        Market impact factor (higher for less liquid securities)
    min_commission : float
        Minimum commission per trade
    strategy_aum : float
        Assets under management for the overall strategy
        
    Returns:
    --------
    Series with transaction costs for each trade
    """
    stock1, stock2 = pair
    
    # Align indices
    common_idx = position_sizes.index.intersection(price_data.index)
    if len(common_idx) < 2:
        logger.error(f"Insufficient aligned data points for cost modeling")
        return None
    
    # Get position data aligned with prices
    positions = position_sizes.reindex(common_idx)
    prices = price_data.reindex(common_idx)
    
    # Calculate absolute position changes
    position_diff1 = positions[stock1].diff().abs()
    position_diff2 = positions[stock2].diff().abs()
    
    # Calculate trade values
    trade_value1 = position_diff1 * prices[stock1]
    trade_value2 = position_diff2 * prices[stock2]
    total_trade_value = trade_value1 + trade_value2
    
    # Calculate market impact based on trade size relative to ADV
    # First, estimate ADV (average daily volume) in dollars
    # This would ideally come from market data, but we'll use a simplified approach
    # Assume ADV is proportional to price and inversely proportional to volatility
    price_volatility1 = prices[stock1].pct_change().rolling(20).std().fillna(0.01)
    price_volatility2 = prices[stock2].pct_change().rolling(20).std().fillna(0.01)
    
    # Estimate ADV (this is a simplified model)
    estimated_adv1 = prices[stock1] * 1000000 * (0.05 / price_volatility1)
    estimated_adv2 = prices[stock2] * 1000000 * (0.05 / price_volatility2)
    
    # Cap ADV estimates to reasonable values
    estimated_adv1 = estimated_adv1.clip(1000000, 100000000)
    estimated_adv2 = estimated_adv2.clip(1000000, 100000000)
    
    # Calculate market impact costs
    # Market impact often scales with square root of trade size relative to ADV
    trade_fraction1 = trade_value1 / estimated_adv1
    trade_fraction2 = trade_value2 / estimated_adv2
    
    market_impact1 = trade_value1 * market_impact_factor * np.sqrt(trade_fraction1)
    market_impact2 = trade_value2 * market_impact_factor * np.sqrt(trade_fraction2)
    
    total_market_impact = market_impact1.fillna(0) + market_impact2.fillna(0)
    
    # Calculate spread costs
    spread_cost1 = trade_value1 * (spread_bps / 10000)
    spread_cost2 = trade_value2 * (spread_bps / 10000)
    total_spread_cost = spread_cost1 + spread_cost2
    
    # Calculate commission costs
    commission_cost1 = np.maximum(trade_value1 * (commission_bps / 10000), min_commission)
    commission_cost2 = np.maximum(trade_value2 * (commission_bps / 10000), min_commission)
    total_commission = commission_cost1 + commission_cost2
    
    # Total transaction costs
    total_costs = total_market_impact + total_spread_cost + total_commission
    
    # Calculate cost as a fraction of trade value
    cost_fraction = total_costs / total_trade_value.replace(0, np.nan).fillna(1)
    
    # Fill NaN values (no trades) with 0
    total_costs = total_costs.fillna(0)
    
    # Log average transaction costs
    avg_cost_bps = (cost_fraction.mean() * 10000) if not cost_fraction.empty else 0
    logger.info(f"Average transaction costs for {pair}: {avg_cost_bps:.2f} bps")
    logger.info(f"  - Spread: {(total_spread_cost.sum() / total_costs.sum() * 100):.1f}%")
    logger.info(f"  - Commission: {(total_commission.sum() / total_costs.sum() * 100):.1f}%")
    logger.info(f"  - Market Impact: {(total_market_impact.sum() / total_costs.sum() * 100):.1f}%")
    
    return pd.Series(total_costs, index=common_idx)


def estimate_implementation_shortfall(pair, price_data, trade_size, participation_rate=0.1,
                                    spread_bps=2, volatility_window=20):
    """
    Estimate implementation shortfall for a trade.
    
    Parameters:
    -----------
    pair : tuple
        Tuple containing the ticker symbols of the pair
    price_data : DataFrame
        DataFrame with price series
    trade_size : float
        Size of the trade in dollars
    participation_rate : float
        Target participation rate as fraction of volume
    spread_bps : float
        Bid-ask spread in basis points
    volatility_window : int
        Window for volatility calculation
        
    Returns:
    --------
    Dictionary with implementation shortfall estimates
    """
    stock1, stock2 = pair
    
    # Check if pair exists in data
    if stock1 not in price_data.columns or stock2 not in price_data.columns:
        logger.error(f"Pair {stock1}-{stock2} not found in price data")
        return None
    
    # Get latest prices
    latest_prices = price_data.iloc[-1]
    price1 = latest_prices[stock1]
    price2 = latest_prices[stock2]
    
    # Calculate volatility
    volatility1 = price_data[stock1].pct_change().rolling(volatility_window).std().iloc[-1] * np.sqrt(252)
    volatility2 = price_data[stock2].pct_change().rolling(volatility_window).std().iloc[-1] * np.sqrt(252)
    
    # Estimate ADV (average daily volume) in dollars
    # This is a simplified approach - in practice, would use actual volume data
    estimated_adv1 = price1 * 1000000 * (0.05 / volatility1)
    estimated_adv2 = price2 * 1000000 * (0.05 / volatility2)
    
    # Cap ADV estimates to reasonable values
    estimated_adv1 = min(max(estimated_adv1, 1000000), 100000000)
    estimated_adv2 = min(max(estimated_adv2, 1000000), 100000000)
    
    # Calculate execution time based on participation rate
    execution_days1 = (trade_size / 2) / (estimated_adv1 * participation_rate)
    execution_days2 = (trade_size / 2) / (estimated_adv2 * participation_rate)
    
    execution_days = max(execution_days1, execution_days2)
    
    # Calculate expected shortfall components
    
    # 1. Spread costs
    spread_cost = trade_size * (spread_bps / 10000)
    
    # 2. Market impact (square root model)
    # Assuming splitting the trade evenly between the two stocks
    impact_factor = 0.1
    impact1 = (trade_size / 2) * impact_factor * np.sqrt((trade_size / 2) / estimated_adv1)
    impact2 = (trade_size / 2) * impact_factor * np.sqrt((trade_size / 2) / estimated_adv2)
    market_impact = impact1 + impact2
    
    # 3. Delay cost (based on volatility and execution time)
    # Assuming normal distribution, approx 0.4 * volatility * sqrt(execution_days)
    delay_factor = 0.4
    delay_cost1 = (trade_size / 2) * volatility1 * delay_factor * np.sqrt(execution_days)
    delay_cost2 = (trade_size / 2) * volatility2 * delay_factor * np.sqrt(execution_days)
    delay_cost = delay_cost1 + delay_cost2
    
    # 4. Opportunity cost (simplified)
    opportunity_cost = market_impact * 0.3  # Assuming 30% of market impact
    
    # Total implementation shortfall
    total_shortfall = spread_cost + market_impact + delay_cost + opportunity_cost
    shortfall_bps = (total_shortfall / trade_size) * 10000
    
    return {
        'spread_cost': spread_cost,
        'market_impact': market_impact,
        'delay_cost': delay_cost,
        'opportunity_cost': opportunity_cost,
        'total_shortfall': total_shortfall,
        'shortfall_bps': shortfall_bps,
        'execution_days': execution_days
    }


def model_variable_costs_by_regime(pair, position_sizes, price_data, regimes,
                                 regime_cost_factors={'low_vol': 0.8, 'high_vol': 1.5},
                                 base_cost_bps=3):
    """
    Model transaction costs that vary by market regime.
    
    Parameters:
    -----------
    pair : tuple
        Tuple containing the ticker symbols of the pair
    position_sizes : DataFrame
        DataFrame with position sizes
    price_data : DataFrame
        DataFrame with price series
    regimes : Series
        Series with market regime classifications
    regime_cost_factors : dict
        Dictionary with cost factors for each regime type
    base_cost_bps : float
        Base transaction cost in basis points
        
    Returns:
    --------
    Series with regime-adjusted transaction costs
    """
    # Check inputs
    if position_sizes is None or price_data is None or regimes is None:
        logger.error("Missing required inputs")
        return None
    
    stock1, stock2 = pair
    
    # Align indices
    common_idx = position_sizes.index.intersection(price_data.index).intersection(regimes.index)
    if len(common_idx) < 2:
        logger.error(f"Insufficient aligned data points for regime cost modeling")
        return None
    
    # Get aligned data
    positions = position_sizes.reindex(common_idx)
    prices = price_data.reindex(common_idx)
    aligned_regimes = regimes.reindex(common_idx)
    
    # Calculate absolute position changes
    position_diff1 = positions[stock1].diff().abs()
    position_diff2 = positions[stock2].diff().abs()
    
    # Calculate trade values
    trade_value1 = position_diff1 * prices[stock1]
    trade_value2 = position_diff2 * prices[stock2]
    total_trade_value = trade_value1 + trade_value2
    
    # Apply regime-specific cost factors
    regime_costs = pd.Series(0.0, index=common_idx)
    
    for i, idx in enumerate(common_idx):
        regime = aligned_regimes.loc[idx]
        trade_value = total_trade_value.loc[idx]
        
        # Determine cost factor based on regime
        if 'high_vol' in regime:
            cost_factor = regime_cost_factors.get('high_vol', 1.5)
        elif 'low_vol' in regime:
            cost_factor = regime_cost_factors.get('low_vol', 0.8)
        else:
            cost_factor = 1.0
        
        # Calculate regime-adjusted cost
        cost_bps = base_cost_bps * cost_factor
        regime_costs.loc[idx] = trade_value * (cost_bps / 10000)
    
    # Log average costs by regime
    for regime_type in set(aligned_regimes):
        mask = aligned_regimes == regime_type
        avg_cost = 0
        
        if total_trade_value[mask].sum() > 0:
            avg_cost = (regime_costs[mask].sum() / total_trade_value[mask].sum()) * 10000
            
        logger.info(f"Average cost for {regime_type} regime: {avg_cost:.2f} bps")
    
    return regime_costs
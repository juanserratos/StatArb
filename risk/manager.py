"""
Risk management functionality for the statistical arbitrage strategy.
"""
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def add_risk_management(signal_df, price_data, pair, max_drawdown=-0.1, 
                       max_holding_days=30, profit_take_multiple=2.0):
    """
    Add risk management rules to trading signals.
    
    Parameters:
    -----------
    signal_df : DataFrame
        DataFrame with trading signals
    price_data : DataFrame
        DataFrame with price series
    pair : tuple
        Tuple containing the ticker symbols of the pair
    max_drawdown : float
        Maximum allowed drawdown for a trade
    max_holding_days : int
        Maximum holding period in days
    profit_take_multiple : float
        Multiple of entry threshold for profit-taking
        
    Returns:
    --------
    DataFrame with risk-managed signals
    """
    if signal_df is None or price_data is None:
        logger.error("Signal DataFrame or price data is None")
        return None
    
    stock1, stock2 = pair
    
    # Make sure we have the required columns
    required_cols = ['spread', 'z_score', 'signal', 'position_size']
    if not all(col in signal_df.columns for col in required_cols):
        logger.error(f"Signal DataFrame missing required columns. Have: {signal_df.columns}")
        return None
    
    # Create a copy to avoid modifying the original
    risk_df = signal_df.copy()
    
    # Add columns for risk management
    risk_df['risk_signal'] = risk_df['signal']
    risk_df['days_in_position'] = 0
    risk_df['entry_z_score'] = np.nan
    risk_df['entry_spread'] = np.nan
    risk_df['max_pos_pnl'] = 0.0
    risk_df['max_neg_pnl'] = 0.0
    risk_df['profit_target'] = np.nan
    risk_df['stop_loss'] = np.nan
    
    # Get pair prices aligned with our signal dates
    pair_prices = price_data[[stock1, stock2]].reindex(risk_df.index)
    
    if len(pair_prices) != len(risk_df):
        logger.error(f"Length mismatch after reindexing: signal_df ({len(risk_df)}) vs pair_prices ({len(pair_prices)})")
        return None
    
    # Calculate pair returns for PnL tracking
    pair_returns = pair_prices.pct_change()
    
    # Implement risk management logic
    in_position = False
    entry_date = None
    entry_prices = None
    
    for i in range(1, len(risk_df)):
        curr_idx = risk_df.index[i]
        prev_idx = risk_df.index[i-1]
        
        prev_signal = risk_df['risk_signal'].iloc[i-1]
        curr_signal = risk_df['signal'].iloc[i]  # Raw signal from original logic
        
        # Update days in position if we're in a position
        if in_position:
            risk_df['days_in_position'].iloc[i] = risk_df['days_in_position'].iloc[i-1] + 1
            
            # Calculate PnL for risk management
            if entry_prices is not None:
                # Simple PnL calculation - long stock1, short stock2 if signal is 1, vice versa if -1
                if prev_signal > 0:  # Long spread
                    pnl = pair_returns[stock2].iloc[i] - pair_returns[stock1].iloc[i]
                else:  # Short spread
                    pnl = pair_returns[stock1].iloc[i] - pair_returns[stock2].iloc[i]
                
                # Track maximum positive and negative PnL
                if pnl > 0:
                    risk_df['max_pos_pnl'].iloc[i] = max(risk_df['max_pos_pnl'].iloc[i-1], pnl)
                else:
                    risk_df['max_neg_pnl'].iloc[i] = min(risk_df['max_neg_pnl'].iloc[i-1], pnl)
                
                # Check stop loss (maximum drawdown)
                cumulative_pnl = risk_df['max_pos_pnl'].iloc[i] + risk_df['max_neg_pnl'].iloc[i]
                if cumulative_pnl < max_drawdown:
                    # Hit stop loss
                    risk_df['risk_signal'].iloc[i] = 0
                    in_position = False
                    entry_date = None
                    entry_prices = None
                    logger.info(f"Stop loss triggered at {curr_idx}: cumulative PnL {cumulative_pnl:.4f}")
                    continue
                
                # Check profit target
                if risk_df['max_pos_pnl'].iloc[i] >= risk_df['profit_target'].iloc[i-1]:
                    # Hit profit target
                    risk_df['risk_signal'].iloc[i] = 0
                    in_position = False
                    entry_date = None
                    entry_prices = None
                    logger.info(f"Profit target reached at {curr_idx}: max PnL {risk_df['max_pos_pnl'].iloc[i]:.4f}")
                    continue
                
                # Check maximum holding period
                if risk_df['days_in_position'].iloc[i] >= max_holding_days:
                    # Exit due to max holding period
                    risk_df['risk_signal'].iloc[i] = 0
                    in_position = False
                    entry_date = None
                    entry_prices = None
                    logger.info(f"Max holding period reached at {curr_idx}: {max_holding_days} days")
                    continue
        
        # Handle signal transitions
        if not in_position and curr_signal != 0:
            # New entry
            in_position = True
            entry_date = curr_idx
            entry_prices = pair_prices.loc[curr_idx, [stock1, stock2]].to_dict()
            
            # Record entry z-score and spread
            risk_df['entry_z_score'].iloc[i] = risk_df['z_score'].iloc[i]
            risk_df['entry_spread'].iloc[i] = risk_df['spread'].iloc[i]
            
            # Set profit target based on entry z-score
            profit_z = abs(risk_df['entry_z_score'].iloc[i]) * profit_take_multiple
            risk_df['profit_target'].iloc[i] = profit_z * 0.01  # Convert z to approx return
            
            # Reset PnL tracking
            risk_df['max_pos_pnl'].iloc[i] = 0.0
            risk_df['max_neg_pnl'].iloc[i] = 0.0
            
            # Set stop loss (as a negative value)
            risk_df['stop_loss'].iloc[i] = max_drawdown
            
            # Apply the signal
            risk_df['risk_signal'].iloc[i] = curr_signal
            
            logger.info(f"New position at {curr_idx}: signal={curr_signal}, entry_z={risk_df['entry_z_score'].iloc[i]:.2f}")
            
        elif in_position and curr_signal == 0:
            # Exit signal from original logic
            in_position = False
            entry_date = None
            entry_prices = None
            
            # Apply the signal
            risk_df['risk_signal'].iloc[i] = 0
            
            logger.info(f"Position closed at {curr_idx} (original signal)")
            
        elif in_position:
            # Copy forward targets
            risk_df['profit_target'].iloc[i] = risk_df['profit_target'].iloc[i-1]
            risk_df['stop_loss'].iloc[i] = risk_df['stop_loss'].iloc[i-1]
            risk_df['entry_z_score'].iloc[i] = risk_df['entry_z_score'].iloc[i-1]
            risk_df['entry_spread'].iloc[i] = risk_df['entry_spread'].iloc[i-1]
            
            # Keep previous signal
            risk_df['risk_signal'].iloc[i] = prev_signal
    
    # Adjust position sizes based on risk signal
    risk_df['risk_position_size'] = risk_df['risk_signal'] * risk_df['vol_ratio'] if 'vol_ratio' in risk_df.columns else risk_df['risk_signal']
    
    return risk_df


def calculate_portfolio_var(positions, price_data, correlation_matrix=None, 
                          confidence_level=0.99, horizon_days=1):
    """
    Calculate Value-at-Risk (VaR) for a portfolio of positions.
    
    Parameters:
    -----------
    positions : dict or Series
        Dictionary or Series with position quantities
    price_data : DataFrame
        DataFrame with price series
    correlation_matrix : DataFrame, optional
        Correlation matrix for assets
    confidence_level : float
        Confidence level for VaR calculation
    horizon_days : int
        Horizon for VaR calculation in days
        
    Returns:
    --------
    Dictionary with VaR calculations
    """
    import scipy.stats as stats
    
    # Convert positions to Series if dict
    if isinstance(positions, dict):
        positions = pd.Series(positions)
    
    # Filter to relevant assets
    assets = positions.index
    prices = price_data[assets]
    
    # Calculate returns and volatility
    returns = prices.pct_change().dropna()
    
    # Daily volatility
    volatility = returns.std()
    
    # Position values
    position_values = {}
    total_position_value = 0
    
    for asset in assets:
        position_values[asset] = positions[asset] * prices[asset].iloc[-1]
        total_position_value += abs(position_values[asset])
    
    # Calculate VaR
    if correlation_matrix is None:
        # Calculate correlation matrix
        correlation_matrix = returns.corr()
    
    # Create variance-covariance matrix
    varcov_matrix = np.outer(volatility, volatility) * correlation_matrix
    
    # Calculate portfolio volatility
    weights = np.array([position_values[asset] / total_position_value for asset in assets])
    port_variance = np.dot(weights.T, np.dot(varcov_matrix, weights))
    port_volatility = np.sqrt(port_variance)
    
    # VaR calculation
    z_score = stats.norm.ppf(confidence_level)
    var_pct = z_score * port_volatility * np.sqrt(horizon_days)
    var_amt = var_pct * total_position_value
    
    # Expected shortfall (Conditional VaR)
    z_expected_shortfall = stats.norm.pdf(z_score) / (1 - confidence_level)
    cvar_pct = z_expected_shortfall * port_volatility * np.sqrt(horizon_days)
    cvar_amt = cvar_pct * total_position_value
    
    logger.info(f"Portfolio VaR ({confidence_level*100:.1f}%, {horizon_days}-day): ${var_amt:.2f} ({var_pct:.2%} of ${total_position_value:.2f})")
    logger.info(f"Expected Shortfall: ${cvar_amt:.2f} ({cvar_pct:.2%})")
    
    return {
        'VaR': var_amt,
        'VaR_percentage': var_pct,
        'CVaR': cvar_amt,
        'CVaR_percentage': cvar_pct,
        'portfolio_volatility': port_volatility,
        'total_position_value': total_position_value
    }


def set_position_limits(prices, pair, volatility_scaling=True, max_notional=1000000,
                       max_capital_pct=0.2, min_position_size=10000):
    """
    Set position limits based on various constraints.
    
    Parameters:
    -----------
    prices : DataFrame
        DataFrame with price series
    pair : tuple
        Tuple containing the ticker symbols of the pair
    volatility_scaling : bool
        Whether to scale position limits by volatility
    max_notional : float
        Maximum notional position size
    max_capital_pct : float
        Maximum percentage of capital to allocate to a pair
    min_position_size : float
        Minimum position size
        
    Returns:
    --------
    Dictionary with position limits
    """
    stock1, stock2 = pair
    
    # Check if pair exists in data
    if stock1 not in prices.columns or stock2 not in prices.columns:
        logger.error(f"Pair {stock1}-{stock2} not found in price data")
        return None
    
    # Calculate volatilities
    vol1 = prices[stock1].pct_change().std() * np.sqrt(252)
    vol2 = prices[stock2].pct_change().std() * np.sqrt(252)
    
    # Calculate latest prices
    price1 = prices[stock1].iloc[-1]
    price2 = prices[stock2].iloc[-1]
    
    # Base sizing
    base_position = min(max_notional, max_capital_pct * max_notional * 5)
    
    if volatility_scaling:
        # Scale position by inverse of volatility
        vol_scaling1 = 0.2 / vol1 if vol1 > 0 else 1
        vol_scaling2 = 0.2 / vol2 if vol2 > 0 else 1
        
        # Cap scaling between 0.5 and 2.0
        vol_scaling1 = max(0.5, min(2.0, vol_scaling1))
        vol_scaling2 = max(0.5, min(2.0, vol_scaling2))
        
        # Apply scaling to position sizes
        notional1 = base_position * vol_scaling1
        notional2 = base_position * vol_scaling2
    else:
        notional1 = base_position
        notional2 = base_position
    
    # Calculate quantities
    max_qty1 = notional1 / price1
    max_qty2 = notional2 / price2
    
    # Ensure minimum position size
    min_qty1 = min_position_size / price1
    min_qty2 = min_position_size / price2
    
    max_qty1 = max(max_qty1, min_qty1)
    max_qty2 = max(max_qty2, min_qty2)
    
    logger.info(f"Position limits for {pair}:")
    logger.info(f"  {stock1}: max ${notional1:.2f} ({max_qty1:.2f} shares)")
    logger.info(f"  {stock2}: max ${notional2:.2f} ({max_qty2:.2f} shares)")
    
    return {
        'max_notional': {stock1: notional1, stock2: notional2},
        'max_quantity': {stock1: max_qty1, stock2: max_qty2},
        'min_quantity': {stock1: min_qty1, stock2: min_qty2}
    }


def calculate_stress_test_scenarios(portfolio, price_data, scenarios=None):
    """
    Perform stress testing on the portfolio under different scenarios.
    
    Parameters:
    -----------
    portfolio : dict or Series
        Dictionary or Series with position quantities
    price_data : DataFrame
        DataFrame with price series
    scenarios : dict, optional
        Dictionary with stress scenarios
        
    Returns:
    --------
    DataFrame with stress test results
    """
    # Default scenarios if none provided
    if scenarios is None:
        scenarios = {
            'Market Crash': {
                'equity': -0.15,  # Equities down 15%
                'volatility': 1.5,  # VIX up 150%
                'interest_rate': -0.25  # Rates down 25 bps
            },
            'Rising Rates': {
                'equity': -0.05,  # Equities down 5%
                'volatility': 0.3,  # VIX up 30%
                'interest_rate': 0.5  # Rates up 50 bps
            },
            'Flight to Quality': {
                'equity': -0.10,  # Equities down 10%
                'interest_rate': -0.5,  # Rates down 50 bps
                'credit_spread': 0.5  # Credit spreads widen 50 bps
            },
            'Inflation Shock': {
                'equity': -0.08,  # Equities down 8%
                'interest_rate': 0.75,  # Rates up 75 bps
                'commodity': 0.15  # Commodities up 15%
            }
        }
    
    # Convert portfolio to Series if dict
    if isinstance(portfolio, dict):
        portfolio = pd.Series(portfolio)
    
    # Get assets
    assets = portfolio.index
    
    # Create asset classification dictionary
    asset_types = {}
    
    for asset in assets:
        # Simple classification based on ticker characteristics
        if asset.startswith('^'):
            asset_types[asset] = 'index'
        elif asset in ['SPY', 'QQQ', 'IWM', 'DIA'] or asset.startswith('XL'):
            asset_types[asset] = 'equity'
        elif asset in ['TLT', 'IEF', 'SHY']:
            asset_types[asset] = 'bond'
        elif asset in ['GLD', 'SLV', 'USO']:
            asset_types[asset] = 'commodity'
        elif asset in ['VXX', '^VIX']:
            asset_types[asset] = 'volatility'
        else:
            asset_types[asset] = 'equity'  # Default to equity
    
    # Calculate current portfolio value
    current_prices = price_data[assets].iloc[-1]
    current_values = portfolio * current_prices
    total_value = current_values.sum()
    
    # Initialize results
    results = []
    
    # Calculate impact of each scenario
    for scenario_name, shocks in scenarios.items():
        scenario_impact = 0
        asset_impacts = {}
        
        for asset in assets:
            asset_type = asset_types[asset]
            position_value = current_values[asset]
            
            # Determine shock based on asset type
            shock = 0
            
            if asset_type == 'equity':
                shock = shocks.get('equity', 0)
            elif asset_type == 'bond':
                # For bonds, impact depends on duration
                # Approximate duration based on ticker
                if asset == 'TLT':  # Long-term Treasury
                    duration = 17
                elif asset == 'IEF':  # Intermediate-term Treasury
                    duration = 7
                else:  # Short-term
                    duration = 2
                
                # Bond price change = -duration * yield change
                shock = -duration * shocks.get('interest_rate', 0) / 100
            elif asset_type == 'commodity':
                shock = shocks.get('commodity', 0)
            elif asset_type == 'volatility':
                shock = shocks.get('volatility', 0)
            
            # Calculate impact for this asset
            impact = position_value * shock
            scenario_impact += impact
            asset_impacts[asset] = {'shock': shock, 'impact': impact}
        
        # Store scenario results
        results.append({
            'scenario': scenario_name,
            'total_impact': scenario_impact,
            'impact_pct': scenario_impact / total_value if total_value != 0 else 0,
            'asset_impacts': asset_impacts
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame([{
        'scenario': r['scenario'],
        'total_impact': r['total_impact'],
        'impact_pct': r['impact_pct']
    } for r in results])
    
    # Log results
    logger.info("Stress Test Results:")
    for _, row in results_df.iterrows():
        logger.info(f"  {row['scenario']}: ${row['total_impact']:.2f} ({row['impact_pct']:.2%})")
    
    return results_df, results
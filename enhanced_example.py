"""
Example script demonstrating the enhanced regime-switching statistical arbitrage strategy.
"""
import logging
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from strategy.enhanced import EnhancedRegimeSwitchingStatArb
from utils.helpers import expand_universe, filter_universe

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_stat_arb.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EnhancedStatArbExample")

def main():
    """
    Execute the enhanced statistical arbitrage strategy.
    """
    # Initialize the strategy with enhanced parameters
    stat_arb = EnhancedRegimeSwitchingStatArb(
        lookback_period=252,      # 1 year of data
        z_entry=2.0,              # Entry threshold
        z_exit=0.5,               # Exit threshold
        n_regimes=3,              # Number of regimes
        confidence=0.05,          # Confidence level for cointegration
        holding_period=20,        # Maximum holding period in days
        min_half_life=5,          # Minimum half-life for mean reversion
        max_half_life=100,        # Maximum half-life for mean reversion
        regime_adapt=True,        # Adapt parameters to regimes
        volatility_scaling=True,  # Scale positions by volatility
        cost_model='advanced',    # Advanced transaction cost modeling
        correlation_filter=True,  # Filter by correlation
        min_correlation=-0.5,     # Minimum correlation threshold
        beta_neutral=True,        # Maintain beta neutrality
        allocation_method='risk_parity',  # Risk parity allocation
        dynamic_hedge=True,       # Use dynamic hedge ratios
        cointegration_method='johansen',  # Improved cointegration testing
        market_regime_aware=True, # Adapt to market regimes
        trade_sizing='vol_adj'    # Volatility-adjusted sizing
    )
    
    # Define universe
    base_tickers = [
        # Sector ETFs
        'XLF', 'XLK', 'XLV', 'XLI', 'XLP', 'XLY', 'XLE', 'XLU', 'XLB', 'XLRE',
    ]
    
    # Expand universe
    tickers = expand_universe(base_tickers)
    
    # Define time period
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
    
    # Execute strategy
    results = stat_arb.execute_strategy(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        top_n_pairs=10,
        market_neutral=True,
        sector_aware=True
    )
    
    # Display results
    if "error" in results:
        logger.error(f"Strategy execution failed: {results['error']}")
        return
    
    # Print aggregate metrics
    logger.info("\nStrategy Performance Summary:")
    if "portfolio" in results:
        portfolio_metrics = results["portfolio"]["metrics"]
        logger.info(f"Total Return: {portfolio_metrics['total_return']:.2%}")
        logger.info(f"Annual Return: {portfolio_metrics['annual_return']:.2%}")
        logger.info(f"Annual Volatility: {portfolio_metrics['annual_volatility']:.2%}")
        logger.info(f"Sharpe Ratio: {portfolio_metrics['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {portfolio_metrics['max_drawdown']:.2%}")
        logger.info(f"Calmar Ratio: {portfolio_metrics['calmar_ratio']:.2f}")
        logger.info(f"Win Rate: {portfolio_metrics['win_rate']:.2%}")
    elif "aggregate" in results:
        agg_metrics = results["aggregate"]["metrics"]
        logger.info(f"Total Return: {agg_metrics['total_return']:.2%}")
        logger.info(f"Annual Return: {agg_metrics['annual_return']:.2%}")
        logger.info(f"Annual Volatility: {agg_metrics['annual_volatility']:.2%}")
        logger.info(f"Sharpe Ratio: {agg_metrics['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {agg_metrics['max_drawdown']:.2%}")
        logger.info(f"Calmar Ratio: {agg_metrics['calmar_ratio']:.2f}")
        logger.info(f"Win Rate: {agg_metrics['win_rate']:.2%}")
    
    # Plot results
    if "portfolio" in results:
        plot_portfolio_results(results["portfolio"])
    elif "aggregate" in results:
        plot_aggregate_results(results)

def plot_portfolio_results(portfolio_results):
    """
    Plot portfolio results.
    
    Parameters:
    -----------
    portfolio_results : dict
        Dictionary with portfolio results
    """
    plt.figure(figsize=(12, 8))
    
    # Plot portfolio equity curve
    plt.subplot(2, 1, 1)
    plt.plot(portfolio_results['portfolio'])
    plt.title('Portfolio Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.grid(True)
    
    # Plot drawdown
    plt.subplot(2, 1, 2)
    equity = portfolio_results['portfolio']
    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1
    plt.plot(drawdown)
    plt.title('Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('portfolio_results.png')
    plt.show()

def plot_aggregate_results(results):
    """
    Plot aggregate results.
    
    Parameters:
    -----------
    results : dict
        Dictionary with strategy results
    """
    plt.figure(figsize=(12, 8))
    
    # Plot aggregate equity curve
    plt.subplot(2, 1, 1)
    plt.plot(results['aggregate']['portfolio'])
    plt.title('Aggregate Portfolio Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.grid(True)
    
    # Plot drawdown
    plt.subplot(2, 1, 2)
    equity = results['aggregate']['portfolio']
    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1
    plt.plot(drawdown)
    plt.title('Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('aggregate_results.png')
    plt.show()

if __name__ == "__main__":
    main()
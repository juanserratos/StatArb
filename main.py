"""
Main entry point for the enhanced statistical arbitrage strategy.
"""
import logging
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse

from strategy.enhanced import EnhancedRegimeSwitchingStatArb
from utils.helpers import expand_universe, filter_universe

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stat_arb_strategy.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EnhancedStatArb")

def parse_args():
    parser = argparse.ArgumentParser(description='Enhanced Statistical Arbitrage Strategy')
    
    parser.add_argument('--start-date', type=str, default='', 
                        help='Start date (YYYY-MM-DD), defaults to 5 years ago')
    parser.add_argument('--end-date', type=str, default='',
                        help='End date (YYYY-MM-DD), defaults to today')
    parser.add_argument('--pairs', type=int, default=10,
                        help='Number of top pairs to use')
    parser.add_argument('--regimes', type=int, default=3, 
                        help='Number of regimes to detect')
    parser.add_argument('--plot', action='store_true',
                        help='Whether to plot results')
    parser.add_argument('--universe', choices=['small', 'medium', 'large'], default='medium',
                        help='Size of universe to use')
    
    return parser.parse_args()

def get_universe(size='medium'):
    """Get ticker universe based on size."""
    if size == 'small':
        # Small universe (sector ETFs)
        return [
            'XLF', 'XLK', 'XLV', 'XLI', 'XLP', 'XLY', 'XLE', 'XLU'
        ]
    elif size == 'medium':
        # Medium universe (sector ETFs + indexes + some liquid ETFs)
        return [
            'XLF', 'XLK', 'XLV', 'XLI', 'XLP', 'XLY', 'XLE', 'XLU', 'XLB', 'XLRE',
            'SPY', 'QQQ', 'IWM', 'DIA', 'VXX', 'TLT', 'GLD', 'USO', 'UUP', 'FXE'
        ]
    else:
        # Large universe (many ETFs)
        return [
            'XLF', 'XLK', 'XLV', 'XLI', 'XLP', 'XLY', 'XLE', 'XLU', 'XLB', 'XLRE',
            'SPY', 'QQQ', 'IWM', 'DIA', 'VXX', 'TLT', 'GLD', 'USO', 'UUP', 'FXE',
            'KBE', 'KRE', 'XHB', 'XRT', 'XME', 'XOP', 'XBI', 'XPH', 
            'FXY', 'SHY', 'IEF', 'SLV', 'BIL'
        ]

def main():
    """Execute the enhanced statistical arbitrage strategy."""
    args = parse_args()
    
    # Set up date range
    if args.end_date:
        end_date = args.end_date
    else:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    if args.start_date:
        start_date = args.start_date
    else:
        start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
    
    # Initialize the strategy with enhanced parameters
    strategy = EnhancedRegimeSwitchingStatArb(
        lookback_period=252,      # 1 year of data
        z_entry=2.0,              # Entry threshold
        z_exit=0.5,               # Exit threshold
        n_regimes=args.regimes,   # Number of regimes
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
    
    # Get ticker universe
    base_tickers = get_universe(args.universe)
    
    # Execute strategy
    results = strategy.execute_strategy(
        tickers=base_tickers,
        start_date=start_date,
        end_date=end_date,
        top_n_pairs=args.pairs,
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
    
    # Plot results if requested
    if args.plot:
        plt.figure(figsize=(12, 8))
        
        if "portfolio" in results:
            equity = results["portfolio"]["portfolio"]
        elif "aggregate" in results:
            equity = results["aggregate"]["portfolio"]
        else:
            logger.error("No portfolio equity curve available to plot")
            return
        
        # Plot equity curve
        plt.subplot(2, 1, 1)
        plt.plot(equity)
        plt.title('Portfolio Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.grid(True)
        
        # Plot drawdown
        plt.subplot(2, 1, 2)
        running_max = equity.cummax()
        drawdown = (equity / running_max) - 1
        plt.plot(drawdown)
        plt.title('Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('strategy_results.png')
        plt.show()

if __name__ == "__main__":
    main()
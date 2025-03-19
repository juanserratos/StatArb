# Create a file called main_relaxed.py with these changes:

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
        logging.FileHandler("stat_arb_strategy_relaxed.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EnhancedStatArb")

def parse_args():
    parser = argparse.ArgumentParser(description='Enhanced Statistical Arbitrage Strategy (Relaxed Criteria)')
    
    parser.add_argument('--start-date', type=str, default='2010-01-01', 
                        help='Start date (YYYY-MM-DD), defaults to 2010-01-01')
    parser.add_argument('--end-date', type=str, default='2023-12-31',
                        help='End date (YYYY-MM-DD), defaults to 2023-12-31')
    parser.add_argument('--pairs', type=int, default=20,
                        help='Number of top pairs to use')
    parser.add_argument('--regimes', type=int, default=2, 
                        help='Number of regimes to detect')
    parser.add_argument('--plot', action='store_true',
                        help='Whether to plot results')
    parser.add_argument('--universe', choices=['small', 'medium', 'large'], default='large',
                        help='Size of universe to use')
    
    return parser.parse_args()

def get_universe(size='large'):
    """Get ticker universe based on size."""
    if size == 'small':
        return [
            'XLF', 'XLK', 'XLV', 'XLI', 'XLP', 'XLY', 'XLE', 'XLU'
        ]
    elif size == 'medium':
        return [
            'XLF', 'XLK', 'XLV', 'XLI', 'XLP', 'XLY', 'XLE', 'XLU', 'XLB', 'XLRE',
            'SPY', 'QQQ', 'IWM', 'DIA', 'TLT', 'GLD', 'USO', 'UUP', 'FXE'
        ]
    else:
        return [
            'XLF', 'XLK', 'XLV', 'XLI', 'XLP', 'XLY', 'XLE', 'XLU', 'XLB', 'XLRE',
            'SPY', 'QQQ', 'IWM', 'DIA', 'TLT', 'GLD', 'USO', 'UUP', 'FXE',
            'KBE', 'KRE', 'XHB', 'XRT', 'XME', 'XOP', 'XBI', 'XPH', 
            'FXY', 'SHY', 'IEF', 'SLV', 'BIL'
        ]

def main():
    """Execute the enhanced statistical arbitrage strategy with RELAXED criteria."""
    args = parse_args()
    
    # Initialize the strategy with VERY RELAXED parameters
    strategy = EnhancedRegimeSwitchingStatArb(
        lookback_period=252,      
        z_entry=1.8,              # Lower threshold for easier entries
        z_exit=0.8,               # Higher threshold for easier exits
        n_regimes=args.regimes,   
        confidence=0.15,          # MUCH more lenient cointegration testing (0.15 instead of 0.05)
        holding_period=25,        # Longer holding period
        min_half_life=2,          # Much lower minimum half-life
        max_half_life=150,        # Much higher maximum half-life
        regime_adapt=True,       
        volatility_scaling=True,  
        cost_model='simple',      # Simpler cost model for initial testing
        correlation_filter=False, # Turn OFF correlation filtering
        min_correlation=0.0,      # No correlation requirement (unused if filter is off)
        beta_neutral=False,       # Turn OFF beta neutrality requirement
        allocation_method='equal_weight',  # Simpler allocation method
        dynamic_hedge=True,       
        cointegration_method='engle_granger',  # Use simpler method
        market_regime_aware=False,  # Turn off market regime awareness to simplify
        trade_sizing='equal'      # Equal position sizing for simplicity
    )
    
    # Get ticker universe
    base_tickers = get_universe(args.universe)
    
    # Execute strategy
    results = strategy.execute_strategy(
        tickers=base_tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        top_n_pairs=args.pairs,
        market_neutral=False,  # Turn off market neutrality
        sector_aware=False     # Turn off sector awareness
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
        plt.savefig('strategy_results_relaxed.png')
        plt.show()

if __name__ == "__main__":
    main()
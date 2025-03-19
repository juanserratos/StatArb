"""
Basic example script for testing the regime-switching statistical arbitrage strategy.
"""
import logging
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from strategy.regime_switching import RegimeSwitchingStatArb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stat_arb_example.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("StatArbExample")

def main():
    """
    Run a basic example of the strategy.
    """
    # Initialize the strategy
    stat_arb = RegimeSwitchingStatArb(
        lookback_period=252,  # 1 year of data
        z_entry=2.0,          # Entry threshold
        z_exit=0.5,           # Exit threshold
        n_regimes=2,          # Number of regimes
        confidence=0.05,      # Confidence level for cointegration
        holding_period=20     # Maximum holding period in days
    )
    
    # Define universe - sector ETFs
    tickers = [
        'XLF', 'XLK', 'XLV', 'XLI', 'XLP', 'XLY', 'XLE', 'XLU', 'XLB', 'XLRE',
    ]
    
    # Alternative smaller universe for faster testing
    small_tickers = [
        'XLF', 'XLK', 'XLV', 'XLI', 'XLP', 'XLY', 'XLE', 'XLU'
    ]
    
    # Define time period
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
    
    # Run strategy with the small universe for testing
    results = stat_arb.run_strategy(
        tickers=small_tickers,
        start_date=start_date,
        end_date=end_date,
        top_n_pairs=3
    )
    
    # Plot results
    if "aggregate" in results:
        plot_results(results)
        print_metrics(results["aggregate"]["metrics"])

def plot_results(results):
    """
    Plot the results of the strategy.
    """
    plt.figure(figsize=(12, 8))
    
    # Portfolio equity curve
    plt.subplot(2, 1, 1)
    plt.plot(results['aggregate']['portfolio'])
    plt.title('Aggregate Portfolio Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.grid(True)
    
    # Drawdown curve
    plt.subplot(2, 1, 2)
    running_max = results['aggregate']['portfolio'].cummax()
    drawdown = (results['aggregate']['portfolio'] / running_max) - 1
    plt.plot(drawdown)
    plt.title('Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('aggregate_results.png')
    plt.show()

def print_metrics(metrics):
    """
    Print strategy metrics.
    """
    print("\nAggregate Strategy Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()
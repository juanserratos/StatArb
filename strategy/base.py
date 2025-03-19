"""
Base strategy class that defines the interface for all strategy implementations.
"""
import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime

from data.market_data import MarketDataHandler
from performance.metrics import calculate_performance_metrics

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    Defines the interface that all strategies must implement, including
    methods for data fetching, signal generation, and backtesting.
    """
    
    def __init__(self, name="BaseStrategy"):
        """
        Initialize the strategy.
        
        Parameters:
        -----------
        name : str
            Name of the strategy
        """
        self.name = name
        self.market_data_handler = MarketDataHandler()
        logger.info(f"Initialized {self.name}")
    
    @abstractmethod
    def fetch_data(self, tickers, start_date, end_date):
        """
        Fetch price data for the given tickers.
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
            
        Returns:
        --------
        DataFrame with price data
        """
        pass
    
    @abstractmethod
    def generate_signals(self, data):
        """
        Generate trading signals from the data.
        
        Parameters:
        -----------
        data : DataFrame
            DataFrame with price and/or indicator data
            
        Returns:
        --------
        DataFrame or Series with trading signals
        """
        pass
    
    @abstractmethod
    def backtest(self, data, signals, **kwargs):
        """
        Backtest the strategy with historical data.
        
        Parameters:
        -----------
        data : DataFrame
            DataFrame with price data
        signals : DataFrame or Series
            DataFrame or Series with trading signals
        **kwargs : dict
            Additional parameters for the backtest
            
        Returns:
        --------
        Dictionary with backtest results
        """
        pass
    
    def run_strategy(self, tickers, start_date, end_date, **kwargs):
        """
        Run the complete strategy from data fetching to backtesting.
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        **kwargs : dict
            Additional parameters for the strategy
            
        Returns:
        --------
        Dictionary with strategy results
        """
        logger.info(f"Running strategy {self.name} from {start_date} to {end_date}")
        
        # Fetch data
        data = self.fetch_data(tickers, start_date, end_date)
        
        if data is None or data.empty:
            logger.error("Failed to fetch data, aborting strategy execution")
            return {"error": "Failed to fetch data"}
        
        # Generate signals
        signals = self.generate_signals(data, **kwargs)
        
        if signals is None:
            logger.error("Failed to generate signals, aborting strategy execution")
            return {"error": "Failed to generate signals"}
        
        # Run backtest
        results = self.backtest(data, signals, **kwargs)
        
        if results is None:
            logger.error("Failed to run backtest, aborting strategy execution")
            return {"error": "Failed to run backtest"}
        
        # Calculate performance metrics
        if "portfolio" in results and isinstance(results["portfolio"], pd.Series):
            metrics = calculate_performance_metrics(results["portfolio"])
            results["metrics"] = metrics
            
            logger.info(f"Strategy execution completed with metrics: " +
                       f"Return={metrics['total_return']:.4f}, " +
                       f"Sharpe={metrics['sharpe_ratio']:.4f}")
        
        return results
    
    def evaluate_performance(self, results, benchmark_ticker="SPY"):
        """
        Evaluate the performance of the strategy against a benchmark.
        
        Parameters:
        -----------
        results : dict
            Dictionary with strategy results
        benchmark_ticker : str
            Ticker symbol for the benchmark
            
        Returns:
        --------
        Dictionary with performance evaluation
        """
        if "portfolio" not in results or "metrics" not in results:
            logger.error("Invalid results format for performance evaluation")
            return None
        
        portfolio = results["portfolio"]
        metrics = results["metrics"]
        
        # Get benchmark data
        start_date = portfolio.index[0].strftime('%Y-%m-%d')
        end_date = portfolio.index[-1].strftime('%Y-%m-%d')
        
        benchmark_data = self.fetch_data([benchmark_ticker], start_date, end_date)
        
        if benchmark_data is None or benchmark_data.empty:
            logger.error(f"Failed to fetch benchmark data for {benchmark_ticker}")
            return metrics
        
        # Calculate benchmark returns
        benchmark_returns = benchmark_data[benchmark_ticker].pct_change().dropna()
        
        # Align portfolio returns with benchmark
        portfolio_returns = portfolio.pct_change().dropna()
        common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
        
        portfolio_returns = portfolio_returns.loc[common_idx]
        benchmark_returns = benchmark_returns.loc[common_idx]
        
        # Calculate benchmark metrics
        benchmark_metrics = calculate_performance_metrics(
            (1 + benchmark_returns).cumprod()
        )
        
        # Calculate relative metrics
        beta = portfolio_returns.cov(benchmark_returns) / benchmark_returns.var()
        alpha = (metrics["annual_return"] - 
                benchmark_metrics["annual_return"] * beta)
        
        # Calculate tracking error
        tracking_error = (portfolio_returns - benchmark_returns * beta).std() * np.sqrt(252)
        information_ratio = alpha / tracking_error if tracking_error > 0 else 0
        
        # Calculate correlation
        correlation = portfolio_returns.corr(benchmark_returns)
        
        # Add relative metrics to results
        relative_metrics = {
            "beta": beta,
            "alpha": alpha,
            "tracking_error": tracking_error,
            "information_ratio": information_ratio,
            "correlation": correlation,
            "benchmark_return": benchmark_metrics["total_return"],
            "benchmark_annual_return": benchmark_metrics["annual_return"],
            "benchmark_volatility": benchmark_metrics["annual_volatility"],
            "benchmark_sharpe": benchmark_metrics["sharpe_ratio"],
            "benchmark_max_drawdown": benchmark_metrics["max_drawdown"]
        }
        
        return {**metrics, **relative_metrics}
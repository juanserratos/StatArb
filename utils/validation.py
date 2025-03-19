"""
Cross-validation and walk-forward testing utilities.
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Implements walk-forward validation for backtesting strategies.
    
    Walk-forward validation avoids look-ahead bias by training on a historical
    window and testing on a subsequent out-of-sample window, then rolling
    forward and repeating the process.
    """
    
    def __init__(self, train_window=252*2, test_window=126, step_size=63, 
                overlap_ratio=0.5, min_train_samples=252):
        """
        Initialize the walk-forward validator.
        
        Parameters:
        -----------
        train_window : int
            Size of training window in days
        test_window : int
            Size of testing window in days
        step_size : int
            Step size for rolling forward in days
        overlap_ratio : float
            Allowed overlap ratio between training sets
        min_train_samples : int
            Minimum required training samples
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.overlap_ratio = overlap_ratio
        self.min_train_samples = min_train_samples
        
    def generate_windows(self, data_start_date, data_end_date):
        """
        Generate train/test windows for walk-forward testing.
        
        Parameters:
        -----------
        data_start_date : datetime or str
            Start date of available data
        data_end_date : datetime or str
            End date of available data
            
        Returns:
        --------
        List of (train_start, train_end, test_start, test_end) tuples
        """
        # Convert to datetime if needed
        if isinstance(data_start_date, str):
            data_start_date = datetime.strptime(data_start_date, '%Y-%m-%d')
        if isinstance(data_end_date, str):
            data_end_date = datetime.strptime(data_end_date, '%Y-%m-%d')
            
        # Initialize windows list
        windows = []
        
        # Set initial dates
        train_start = data_start_date
        train_end = train_start + timedelta(days=self.train_window)
        test_start = train_end
        test_end = test_start + timedelta(days=self.test_window)
        
        # Generate windows until we reach the end of data
        while test_end <= data_end_date:
            windows.append((
                train_start.strftime('%Y-%m-%d'),
                train_end.strftime('%Y-%m-%d'),
                test_start.strftime('%Y-%m-%d'),
                test_end.strftime('%Y-%m-%d')
            ))
            
            # Roll forward
            train_start = train_start + timedelta(days=self.step_size)
            train_end = train_start + timedelta(days=self.train_window)
            test_start = train_end
            test_end = test_start + timedelta(days=self.test_window)
        
        return windows
    
    def run_walk_forward_test(self, strategy, data, windows=None, **strategy_params):
        """
        Run walk-forward testing on a strategy.
        
        Parameters:
        -----------
        strategy : object
            Strategy object that implements generate_signals and backtest methods
        data : DataFrame
            Full dataset for testing
        windows : list, optional
            List of (train_start, train_end, test_start, test_end) tuples
            If None, will be generated from data index
        **strategy_params : dict
            Additional parameters for the strategy
            
        Returns:
        --------
        Dictionary with combined results and individual window results
        """
        # Generate windows if not provided
        if windows is None:
            data_start = data.index[0]
            data_end = data.index[-1]
            windows = self.generate_windows(data_start, data_end)
            
        logger.info(f"Running walk-forward test with {len(windows)} windows")
        
        # Store results for each window
        all_window_results = {}
        combined_portfolio = pd.Series()
        
        # Run test for each window
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            logger.info(f"Window {i+1}/{len(windows)}: "
                       f"Train {train_start} to {train_end}, "
                       f"Test {test_start} to {test_end}")
            
            # Split data
            train_mask = (data.index >= train_start) & (data.index < train_end)
            test_mask = (data.index >= test_start) & (data.index <= test_end)
            
            train_data = data[train_mask]
            test_data = data[test_mask]
            
            if len(train_data) < self.min_train_samples:
                logger.warning(f"Window {i+1}: Insufficient training data "
                              f"({len(train_data)} < {self.min_train_samples})")
                continue
            
            # Generate signals on training data
            signals = strategy.generate_signals(train_data, **strategy_params)
            
            if not signals:
                logger.warning(f"Window {i+1}: No signals generated")
                continue
            
            # Backtest on testing data
            results = strategy.backtest(test_data, signals, **strategy_params)
            
            if results is None or 'aggregate' not in results:
                logger.warning(f"Window {i+1}: Backtest failed")
                continue
            
            # Store window results
            all_window_results[f"window_{i+1}"] = {
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'portfolio': results['aggregate']['portfolio'],
                'metrics': results['aggregate']['metrics']
            }
            
            # Add to combined portfolio
            window_portfolio = results['aggregate']['portfolio']
            
            if combined_portfolio.empty:
                # Initialize with first window
                combined_portfolio = window_portfolio
            else:
                # Scale the new window to start at the end value of the combined portfolio
                scale_factor = combined_portfolio.iloc[-1] / window_portfolio.iloc[0]
                scaled_window = window_portfolio * scale_factor
                
                # Append to combined portfolio
                combined_portfolio = pd.concat([combined_portfolio, scaled_window])
        
        # Calculate combined metrics
        combined_returns = combined_portfolio.pct_change().fillna(0)
        total_return = (combined_portfolio.iloc[-1] / combined_portfolio.iloc[0]) - 1
        annual_return = ((1 + total_return) ** (252 / len(combined_portfolio))) - 1
        annual_volatility = combined_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Calculate drawdown
        running_max = combined_portfolio.cummax()
        drawdown = (combined_portfolio / running_max) - 1
        max_drawdown = drawdown.min()
        
        combined_metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': -annual_return / max_drawdown if max_drawdown != 0 else float('inf'),
            'win_rate': (combined_returns > 0).mean()
        }
        
        # Create overall results
        results = {
            'combined_portfolio': combined_portfolio,
            'combined_metrics': combined_metrics,
            'window_results': all_window_results
        }
        
        # Log combined metrics
        logger.info(f"Walk-forward test results:")
        logger.info(f"Total Return: {combined_metrics['total_return']:.2%}")
        logger.info(f"Annual Return: {combined_metrics['annual_return']:.2%}")
        logger.info(f"Annual Volatility: {combined_metrics['annual_volatility']:.2%}")
        logger.info(f"Sharpe Ratio: {combined_metrics['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {combined_metrics['max_drawdown']:.2%}")
        logger.info(f"Calmar Ratio: {combined_metrics['calmar_ratio']:.2f}")
        
        return results
    
    def plot_walk_forward_results(self, results, save_path=None):
        """
        Plot the results of walk-forward testing.
        
        Parameters:
        -----------
        results : dict
            Dictionary with walk-forward test results
        save_path : str, optional
            Path to save the plot
        """
        combined_portfolio = results['combined_portfolio']
        window_results = results['window_results']
        combined_metrics = results['combined_metrics']
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Plot combined equity curve
        plt.subplot(211)
        plt.plot(combined_portfolio, label='Combined Portfolio')
        
        # Plot individual windows with different colors
        colors = plt.cm.tab10(np.linspace(0, 1, len(window_results)))
        
        for i, (window_name, window_result) in enumerate(window_results.items()):
            window_portfolio = window_result['portfolio']
            
            # Scale window portfolio to match combined
            common_dates = window_portfolio.index.intersection(combined_portfolio.index)
            if not common_dates.empty:
                first_common_date = common_dates[0]
                scale_factor = combined_portfolio.loc[first_common_date] / window_portfolio.loc[first_common_date]
                scaled_window = window_portfolio * scale_factor
                
                # Plot scaled window
                plt.plot(scaled_window, color=colors[i], alpha=0.5, 
                        label=f"{window_name}: {window_result['test_start']} to {window_result['test_end']}")
        
        plt.title('Walk-Forward Test: Equity Curve', fontsize=14)
        plt.ylabel('Portfolio Value')
        plt.grid(True)
        plt.legend(loc='upper left')
        
        # Plot drawdown
        plt.subplot(212)
        running_max = combined_portfolio.cummax()
        drawdown = (combined_portfolio / running_max) - 1
        
        plt.fill_between(drawdown.index, 0, drawdown, color='red', alpha=0.3)
        plt.plot(drawdown, color='red', linewidth=1)
        
        plt.title('Walk-Forward Test: Drawdown', fontsize=14)
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        
        # Add metrics text
        metrics_text = (
            f"Total Return: {combined_metrics['total_return']:.2%}, "
            f"Annual Return: {combined_metrics['annual_return']:.2%}, "
            f"Annual Vol: {combined_metrics['annual_volatility']:.2%}, "
            f"Sharpe: {combined_metrics['sharpe_ratio']:.2f}, "
            f"Max DD: {combined_metrics['max_drawdown']:.2%}"
        )
        
        plt.figtext(0.5, 0.01, metrics_text, ha='center', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            
        plt.show()


def perform_sensitivity_analysis(strategy, data, param_ranges, fixed_params=None,
                              train_start=None, train_end=None, test_start=None, test_end=None):
    """
    Perform sensitivity analysis by varying parameters and measuring performance.
    
    Parameters:
    -----------
    strategy : object
        Strategy object that implements generate_signals and backtest methods
    data : DataFrame
        Dataset for testing
    param_ranges : dict
        Dictionary mapping parameter names to lists of values to test
    fixed_params : dict, optional
        Dictionary with fixed parameters
    train_start, train_end, test_start, test_end : str, optional
        Date ranges for training and testing
        
    Returns:
    --------
    DataFrame with parameter combinations and resulting metrics
    """
    import itertools
    
    # Set default fixed parameters
    if fixed_params is None:
        fixed_params = {}
        
    # Set default date ranges
    if train_start is None:
        train_start = data.index[0].strftime('%Y-%m-%d')
    if train_end is None:
        mid_point = len(data) // 2
        train_end = data.index[mid_point].strftime('%Y-%m-%d')
    if test_start is None:
        test_start = train_end
    if test_end is None:
        test_end = data.index[-1].strftime('%Y-%m-%d')
    
    # Split data
    train_mask = (data.index >= train_start) & (data.index < train_end)
    test_mask = (data.index >= test_start) & (data.index <= test_end)
    
    train_data = data[train_mask]
    test_data = data[test_mask]
    
    logger.info(f"Sensitivity analysis: train {train_start} to {train_end}, "
               f"test {test_start} to {test_end}")
    
    # Generate all parameter combinations
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    param_combinations = list(itertools.product(*param_values))
    
    logger.info(f"Testing {len(param_combinations)} parameter combinations")
    
    # Store results
    results = []
    
    # Test each combination
    for i, combo in enumerate(param_combinations):
        # Create parameter dictionary
        params = {**fixed_params, **{name: value for name, value in zip(param_names, combo)}}
        
        logger.info(f"Combination {i+1}/{len(param_combinations)}: {params}")
        
        # Generate signals on training data
        signals = strategy.generate_signals(train_data, **params)
        
        if not signals:
            logger.warning(f"Combination {i+1}: No signals generated")
            continue
        
        # Backtest on testing data
        backtest_results = strategy.backtest(test_data, signals, **params)
        
        if backtest_results is None or 'aggregate' not in backtest_results:
            logger.warning(f"Combination {i+1}: Backtest failed")
            continue
        
        # Extract metrics
        metrics = backtest_results['aggregate']['metrics']
        
        # Store results
        result = {**params, **metrics}
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Log best combinations
    if not results_df.empty:
        best_sharpe = results_df.loc[results_df['sharpe_ratio'].idxmax()]
        best_return = results_df.loc[results_df['annual_return'].idxmax()]
        best_drawdown = results_df.loc[results_df['max_drawdown'].idxmin()]
        
        logger.info(f"Best Sharpe ratio: {best_sharpe['sharpe_ratio']:.2f} "
                   f"with parameters: {best_sharpe[param_names].to_dict()}")
        
        logger.info(f"Best annual return: {best_return['annual_return']:.2%} "
                   f"with parameters: {best_return[param_names].to_dict()}")
        
        logger.info(f"Best max drawdown: {best_drawdown['max_drawdown']:.2%} "
                   f"with parameters: {best_drawdown[param_names].to_dict()}")
    
    return results_df
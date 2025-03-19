"""
Production-ready statistical arbitrage strategy with comprehensive evaluation.
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import argparse
import os

from strategy.enhanced import EnhancedRegimeSwitchingStatArb
from execution.realistic_costs import RealisticCostModel, apply_realistic_costs_to_backtest
from risk.enhanced_manager import EnhancedRiskManager, enhance_risk_management
from utils.validation import WalkForwardValidator, perform_sensitivity_analysis
from utils.helpers import expand_universe, filter_universe
from utils.plot_utils import plot_strategy_performance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("production_stat_arb.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ProductionStatArb")

def parse_args():
    parser = argparse.ArgumentParser(description='Production-Ready Statistical Arbitrage Strategy')
    
    parser.add_argument('--start-date', type=str, default='2010-01-01', 
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=datetime.now().strftime('%Y-%m-%d'),
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--pairs', type=int, default=10,
                        help='Number of top pairs to use')
    parser.add_argument('--regimes', type=int, default=2, 
                        help='Number of regimes to detect')
    parser.add_argument('--validation', action='store_true',
                        help='Run walk-forward validation')
    parser.add_argument('--sensitivity', action='store_true',
                        help='Run parameter sensitivity analysis')
    parser.add_argument('--realistic-costs', action='store_true',
                        help='Apply realistic transaction costs')
    parser.add_argument('--risk-management', action='store_true',
                        help='Apply enhanced risk management')
    parser.add_argument('--universe', choices=['small', 'medium', 'large'], default='medium',
                        help='Size of universe to use')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory for output files')
    
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
            'SPY', 'QQQ', 'IWM', 'DIA', 'TLT', 'GLD', 'USO', 'UUP', 'FXE'
        ]
    else:
        # Large universe (many ETFs)
        return [
            'XLF', 'XLK', 'XLV', 'XLI', 'XLP', 'XLY', 'XLE', 'XLU', 'XLB', 'XLRE',
            'SPY', 'QQQ', 'IWM', 'DIA', 'TLT', 'GLD', 'USO', 'UUP', 'FXE',
            'KBE', 'KRE', 'XHB', 'XRT', 'XME', 'XOP', 'XBI', 'XPH', 
            'FXY', 'SHY', 'IEF', 'SLV', 'BIL'
        ]

def run_strategy(args):
    """Execute the strategy with all enhancements."""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the strategy with production parameters
    strategy = EnhancedRegimeSwitchingStatArb(
        lookback_period=252,      # 1 year of data
        z_entry=2.0,              # Balanced entry threshold
        z_exit=0.5,               # Modest exit threshold
        n_regimes=args.regimes,   # Number of regimes
        confidence=0.05,          # Standard confidence level
        holding_period=20,        # Reasonable holding period
        min_half_life=5,          # Minimum half-life
        max_half_life=100,        # Maximum half-life
        regime_adapt=True,        # Adapt parameters to regimes
        volatility_scaling=True,  # Scale positions by volatility
        cost_model='advanced',    # Advanced cost modeling
        correlation_filter=True,  # Filter by correlation
        min_correlation=-0.5,     # Reasonable correlation threshold
        beta_neutral=True,        # Maintain beta neutrality
        allocation_method='risk_parity',  # Risk parity allocation
        dynamic_hedge=True,       # Use dynamic hedge ratios
        cointegration_method='johansen',  # Better cointegration testing
        market_regime_aware=True, # Adapt to market regimes
        trade_sizing='vol_adj'    # Volatility-adjusted sizing
    )
    
    # Get ticker universe
    base_tickers = get_universe(args.universe)
    expanded_tickers = expand_universe(base_tickers)
    
    # Run Walk-Forward Validation if requested
    if args.validation:
        logger.info("Running walk-forward validation...")
        
        # First fetch all price data
        logger.info("Fetching price data for validation...")
        all_price_data = strategy.fetch_data(expanded_tickers, args.start_date, args.end_date)
        filtered_tickers = filter_universe(all_price_data, min_price=5.0)
        all_price_data = all_price_data[filtered_tickers]
        
        # Initialize validator
        validator = WalkForwardValidator(
            train_window=252*2,  # 2 years training
            test_window=126,     # 6 months testing
            step_size=63,        # 3 months step
            overlap_ratio=0.5,
            min_train_samples=252
        )
        
        # Define parameters for validation
        validation_params = {
            'top_n_pairs': args.pairs,
            'market_neutral': True,
            'sector_aware': True
        }
        
        # Run validation
        validation_results = validator.run_walk_forward_test(
            strategy, all_price_data, **validation_params
        )
        
        # Plot results
        validator.plot_walk_forward_results(
            validation_results,
            save_path=os.path.join(args.output_dir, "walk_forward_results.png")
        )
        
        # Save validation metrics
        val_metrics = validation_results['combined_metrics']
        
        logger.info("Walk-forward validation results:")
        logger.info(f"Total Return: {val_metrics['total_return']:.2%}")
        logger.info(f"Annual Return: {val_metrics['annual_return']:.2%}")
        logger.info(f"Annual Volatility: {val_metrics['annual_volatility']:.2%}")
        logger.info(f"Sharpe Ratio: {val_metrics['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {val_metrics['max_drawdown']:.2%}")
        logger.info(f"Calmar Ratio: {val_metrics['calmar_ratio']:.2f}")
        
        # Save validation results
        pd.DataFrame([val_metrics]).to_csv(
            os.path.join(args.output_dir, "validation_metrics.csv")
        )
    
    # Run Parameter Sensitivity Analysis if requested
    if args.sensitivity:
        logger.info("Running parameter sensitivity analysis...")
        
        # First fetch all price data
        logger.info("Fetching price data for sensitivity analysis...")
        all_price_data = strategy.fetch_data(expanded_tickers, args.start_date, args.end_date)
        filtered_tickers = filter_universe(all_price_data, min_price=5.0)
        all_price_data = all_price_data[filtered_tickers]
        
        # Define parameter ranges to test
        param_ranges = {
            'z_entry': [1.5, 1.8, 2.0, 2.2, 2.5],
            'z_exit': [0.3, 0.5, 0.7, 0.9],
            'min_half_life': [3, 5, 7, 10],
            'max_half_life': [60, 80, 100, 120]
        }
        
        # Fixed parameters
        fixed_params = {
            'top_n_pairs': args.pairs,
            'n_regimes': args.regimes,
            'market_neutral': True,
            'sector_aware': True
        }
        
        # Calculate mid-point for train/test split
        mid_point = len(all_price_data) // 2
        train_end = all_price_data.index[mid_point].strftime('%Y-%m-%d')
        
        # Run sensitivity analysis
        sensitivity_results = perform_sensitivity_analysis(
            strategy,
            all_price_data,
            param_ranges,
            fixed_params,
            args.start_date,
            train_end,
            train_end,
            args.end_date
        )
        
        # Save sensitivity results
        sensitivity_results.to_csv(
            os.path.join(args.output_dir, "sensitivity_results.csv")
        )
        
        # Plot heatmap of key parameters vs Sharpe ratio
        plt.figure(figsize=(15, 10))
        
        # Z-entry vs Z-exit heatmap
        plt.subplot(221)
        pivot1 = sensitivity_results.pivot_table(
            values='sharpe_ratio', index='z_entry', columns='z_exit'
        )
        sns.heatmap(pivot1, annot=True, fmt='.2f', cmap='RdYlGn', center=0)
        plt.title('Sharpe Ratio: Z-Entry vs Z-Exit')
        
        # Half-life ranges heatmap
        plt.subplot(222)
        pivot2 = sensitivity_results.pivot_table(
            values='sharpe_ratio', index='min_half_life', columns='max_half_life'
        )
        sns.heatmap(pivot2, annot=True, fmt='.2f', cmap='RdYlGn', center=0)
        plt.title('Sharpe Ratio: Min Half-Life vs Max Half-Life')
        
        # Z-entry vs Min half-life
        plt.subplot(223)
        pivot3 = sensitivity_results.pivot_table(
            values='sharpe_ratio', index='z_entry', columns='min_half_life'
        )
        sns.heatmap(pivot3, annot=True, fmt='.2f', cmap='RdYlGn', center=0)
        plt.title('Sharpe Ratio: Z-Entry vs Min Half-Life')
        
        # Max drawdown heatmap
        plt.subplot(224)
        pivot4 = sensitivity_results.pivot_table(
            values='max_drawdown', index='z_entry', columns='z_exit'
        )
        sns.heatmap(pivot4, annot=True, fmt='.2%', cmap='RdYlGn_r')
        plt.title('Max Drawdown: Z-Entry vs Z-Exit')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "sensitivity_heatmaps.png"))
        plt.close()
        
        # Find best parameter sets
        best_sharpe = sensitivity_results.loc[sensitivity_results['sharpe_ratio'].idxmax()]
        best_return = sensitivity_results.loc[sensitivity_results['annual_return'].idxmax()]
        best_drawdown = sensitivity_results.loc[sensitivity_results['max_drawdown'].idxmin()]
        
        logger.info(f"Best Sharpe parameters: {best_sharpe[param_ranges.keys()].to_dict()}")
        logger.info(f"Best Return parameters: {best_return[param_ranges.keys()].to_dict()}")
        logger.info(f"Best Drawdown parameters: {best_drawdown[param_ranges.keys()].to_dict()}")
    
    # Run main backtest
    logger.info("Running main strategy backtest...")
    
    # Execute strategy
    results = strategy.execute_strategy(
        tickers=expanded_tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        top_n_pairs=args.pairs,
        market_neutral=True,
        sector_aware=True
    )
    
    if "error" in results:
        logger.error(f"Strategy execution failed: {results['error']}")
        return
    
    # Apply realistic costs if requested
    if args.realistic_costs:
        logger.info("Applying realistic transaction costs...")
        cost_model = RealisticCostModel(
            base_spread_bps=2.0,
            base_commission_bps=1.0,
            impact_factor=0.1,
            opportunity_cost_factor=0.05,
            borrow_cost_bps=15.0
        )
        
        realistic_results = apply_realistic_costs_to_backtest(results, cost_model)
        
        # Replace results with realistic version
        results = realistic_results
    
    # Apply enhanced risk management if requested
    if args.risk_management:
        logger.info("Applying enhanced risk management...")
        risk_results = enhance_risk_management(
            results,
            max_drawdown=-0.15,
            max_concentration=0.2,
            max_sector_exposure=0.3,
            max_leverage=1.5
        )
        
        # Replace results with risk-managed version
        results = risk_results
    
    # Plot final results
    if "portfolio" in results:
        portfolio_data = results["portfolio"]
        metrics = results["portfolio"]["metrics"]
    elif "aggregate" in results:
        portfolio_data = results["aggregate"]["portfolio"]
        metrics = results["aggregate"]["metrics"]
    else:
        logger.error("No portfolio data found in results")
        return
    
    # Plot performance
    plot_strategy_performance(
        results,
        save_path=os.path.join(args.output_dir, "strategy_performance.png")
    )
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(args.output_dir, "strategy_metrics.csv"))
    
    # Save equity curve
    portfolio_data.to_csv(os.path.join(args.output_dir, "equity_curve.csv"))
    
    # Print summary
    logger.info("\nStrategy Performance Summary:")
    logger.info(f"Total Return: {metrics['total_return']:.2%}")
    logger.info(f"Annual Return: {metrics['annual_return']:.2%}")
    logger.info(f"Annual Volatility: {metrics['annual_volatility']:.2%}")
    logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    logger.info(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
    logger.info(f"Win Rate: {metrics['win_rate']:.2%}")
    
    return results

if __name__ == "__main__":
    import seaborn as sns
    args = parse_args()
    run_strategy(args)
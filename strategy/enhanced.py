"""
Enhanced implementation of the regime-switching statistical arbitrage strategy.
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime

from strategy.regime_switching import RegimeSwitchingStatArb
from pairs.selection import find_enhanced_pairs, get_sector_mapping
from regime.detector import detect_complex_regimes
from regime.macro_indicators import fetch_enhanced_macro_data, detect_market_regime
from signals.generator import generate_enhanced_signals
from execution.transaction_costs import model_transaction_costs
from execution.simulator import backtest_with_realistic_execution
from portfolio.allocator import calculate_optimal_allocations
from portfolio.optimizer import backtest_portfolio
from risk.manager import add_risk_management
from utils.helpers import expand_universe, filter_universe

logger = logging.getLogger(__name__)


class EnhancedRegimeSwitchingStatArb(RegimeSwitchingStatArb):
    """
    Enhanced version of the RegimeSwitchingStatArb strategy with additional features
    and improvements to address the underwhelming performance of the original strategy.
    """
    
    def __init__(self, lookback_period=252, z_entry=2.0, z_exit=0.5, 
                 n_regimes=3, confidence=0.05, holding_period=20,
                 min_half_life=5, max_half_life=100, regime_adapt=True,
                 volatility_scaling=True, cost_model='advanced',
                 correlation_filter=True, min_correlation=-0.5,
                 beta_neutral=True, allocation_method='risk_parity',
                 dynamic_hedge=True, cointegration_method='johansen',
                 market_regime_aware=True, trade_sizing='vol_adj',
                 name="EnhancedRegimeSwitchingStatArb"):
        """
        Initialize the enhanced strategy with additional parameters.
        
        Parameters:
        -----------
        lookback_period : int
            Period used for estimating cointegration and regime parameters
        z_entry : float
            Z-score threshold for entering positions
        z_exit : float
            Z-score threshold for exiting positions
        n_regimes : int
            Number of regimes in the Hidden Markov Model
        confidence : float
            Confidence level for cointegration testing
        holding_period : int
            Maximum holding period in days
        min_half_life : int
            Minimum acceptable half-life for mean reversion in days
        max_half_life : int
            Maximum acceptable half-life for mean reversion in days
        regime_adapt : bool
            Whether to adapt parameters based on regimes
        volatility_scaling : bool
            Whether to scale positions by volatility
        cost_model : str
            Transaction cost model to use: 'simple', 'intermediate', or 'advanced'
        correlation_filter : bool
            Whether to filter pairs based on correlation
        min_correlation : float
            Minimum correlation threshold for pair selection
        beta_neutral : bool
            Whether to maintain beta neutrality with market index
        allocation_method : str
            Allocation method: 'equal_weight', 'risk_parity', 'min_variance', 'max_sharpe'
        dynamic_hedge : bool
            Whether to use dynamic hedge ratios
        cointegration_method : str
            Method for cointegration testing: 'engle_granger' or 'johansen'
        market_regime_aware : bool
            Whether to adapt strategy to market regimes
        trade_sizing : str
            Position sizing method: 'equal', 'vol_adj', 'kelly'
        name : str
            Name of the strategy
        """
        # Initialize the base class
        super().__init__(
            lookback_period=lookback_period,
            z_entry=z_entry,
            z_exit=z_exit,
            n_regimes=n_regimes,
            confidence=confidence,
            holding_period=holding_period,
            name=name
        )
        
        # Enhanced parameters
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.regime_adapt = regime_adapt
        self.volatility_scaling = volatility_scaling
        self.cost_model = cost_model
        self.correlation_filter = correlation_filter
        self.min_correlation = min_correlation
        self.beta_neutral = beta_neutral
        self.allocation_method = allocation_method
        self.dynamic_hedge = dynamic_hedge
        self.cointegration_method = cointegration_method
        self.market_regime_aware = market_regime_aware
        self.trade_sizing = trade_sizing
        
        # Track market regime
        self.market_regime = None
        self.regime_map = {
            'bull_low_vol': {'z_entry': 2.5, 'z_exit': 0.7, 'holding_period': 15},
            'bull_high_vol': {'z_entry': 2.0, 'z_exit': 0.5, 'holding_period': 10},
            'bear_low_vol': {'z_entry': 1.8, 'z_exit': 0.6, 'holding_period': 12},
            'bear_high_vol': {'z_entry': 1.5, 'z_exit': 0.4, 'holding_period': 8}
        }
        
        logger.info(f"Enhanced strategy initialized with parameters:" +
                   f" lookback={lookback_period}, z_entry={z_entry}, z_exit={z_exit}, " +
                   f"n_regimes={n_regimes}, min_half_life={min_half_life}, " +
                   f"volatility_scaling={volatility_scaling}, cost_model={cost_model}")
    
    def fetch_macro_data(self, start_date, end_date, retry_count=3, retry_delay=10):
        """
        Override to use enhanced macro data fetching.
        """
        return fetch_enhanced_macro_data(
            start_date, end_date, self.lookback_period, retry_count, retry_delay,
            self.market_data_handler
        )
    
    def adapt_parameters_to_regime(self):
        """
        Adapt strategy parameters based on the current market regime.
        
        Returns:
        --------
        Dictionary with adapted parameters
        """
        if not self.market_regime_aware or self.market_regime is None:
            return {
                'z_entry': self.z_entry,
                'z_exit': self.z_exit,
                'holding_period': self.holding_period
            }
        
        # Get regime-specific parameters
        regime_params = self.regime_map.get(self.market_regime, {})
        
        adapted_params = {
            'z_entry': regime_params.get('z_entry', self.z_entry),
            'z_exit': regime_params.get('z_exit', self.z_exit),
            'holding_period': regime_params.get('holding_period', self.holding_period)
        }
        
        logger.info(f"Adapted parameters for {self.market_regime} regime: {adapted_params}")
        
        return adapted_params
    
    def generate_signals(self, data, pairs=None, macro_data=None, top_n_pairs=5):
        """
        Override to use enhanced signal generation.
        
        Parameters:
        -----------
        data : DataFrame
            DataFrame with price data
        pairs : list, optional
            List of predefined pairs to use, if None will find pairs using enhanced methods
        macro_data : DataFrame, optional
            DataFrame with macroeconomic indicators
        top_n_pairs : int
            Number of top pairs to use if pairs is None
            
        Returns:
        --------
        Dictionary with signals for each pair
        """
        # Find pairs if not provided
        if pairs is None:
            # Use sector mapping for enhanced pair selection if available
            sector_map = None
            if hasattr(self, 'get_sector_mapping'):
                logger.info("Getting sector information...")
                sector_map = get_sector_mapping(data.columns)
            
            # Find pairs with enhanced methods
            pairs_with_data = find_enhanced_pairs(
                data, self.lookback_period, self.min_half_life, self.max_half_life,
                self.confidence, self.correlation_filter, self.min_correlation,
                self.cointegration_method, sector_map, macro_data, self.beta_neutral
            )
            
            pairs = [p['pair'] for p in pairs_with_data[:min(top_n_pairs, len(pairs_with_data))]]
        
        logger.info(f"Generating enhanced signals for {len(pairs)} pairs")
        
        # Generate signals for each pair
        pair_signals = {}
        
        for pair in pairs:
            stock1, stock2 = pair
            
            # Check if pair exists in data
            if stock1 not in data.columns or stock2 not in data.columns:
                logger.warning(f"Pair {pair} not found in data, skipping")
                continue
            
            # Calculate dynamic hedge ratio if enabled
            hedge_ratios = self._calculate_hedge_ratio(data, pair) if self.dynamic_hedge else None
            
            # Calculate spread
            spread = self._calculate_spread(data, pair, hedge_ratios)
            
            if spread is None:
                logger.warning(f"Failed to calculate spread for {pair}, skipping")
                continue
            
            # Detect regimes with enhanced methods
            hmm_results = detect_complex_regimes(spread, macro_data, n_regimes=self.n_regimes)
            
            if hmm_results[0] is None:
                logger.warning(f"Failed to detect complex regimes for {pair}, skipping")
                continue
                
            model, hidden_states, regime_labels, feature_df = hmm_results
            
            # Generate trading signals with enhanced methods
            signal_df = generate_enhanced_signals(
                spread, hmm_results, macro_data, volatility_scaling=self.volatility_scaling
            )
            
            if signal_df is None:
                logger.warning(f"Failed to generate enhanced signals for {pair}, skipping")
                continue
            
            # Add risk management if enabled
            if self.regime_adapt:
                risk_df = add_risk_management(signal_df, data, pair, 
                                             max_drawdown=-0.1, 
                                             max_holding_days=self.holding_period,
                                             profit_take_multiple=2.0)
                
                if risk_df is not None:
                    signals = risk_df['risk_signal']
                else:
                    signals = signal_df['signal']
            else:
                signals = signal_df['signal']
            
            pair_signals[str(pair)] = {
                'spread': spread,
                'hedge_ratios': hedge_ratios,
                'regimes': hidden_states,
                'regime_labels': regime_labels,
                'signals': signals,
                'signal_df': signal_df
            }
        
        return pair_signals
    
    def backtest(self, data, pair_signals, initial_capital=1000000, transaction_cost=0.0005):
        """
        Override to use enhanced backtesting.
        
        Parameters:
        -----------
        data : DataFrame
            DataFrame with price data
        pair_signals : dict
            Dictionary with signals for each pair
        initial_capital : float
            Initial capital for the backtest
        transaction_cost : float
            Transaction cost as a fraction of trade value
            
        Returns:
        --------
        Dictionary with backtest results
        """
        if not pair_signals:
            logger.error("No signals to backtest")
            return None
        
        results = {}
        
        # Backtest each pair with realistic execution and costs
        for pair_str, signals_data in pair_signals.items():
            pair = eval(pair_str)  # Convert string back to tuple
            spread = signals_data['spread']
            hedge_ratios = signals_data['hedge_ratios']
            signals = signals_data['signals']
            
            # Run backtest for this pair with realistic execution
            pair_results = backtest_with_realistic_execution(
                pair, signals, data, hedge_ratios,
                initial_capital=initial_capital,
                transaction_cost_model=self.cost_model,
                execution_model='twap'
            )
            
            if pair_results is not None:
                results[pair_str] = pair_results
        
        # Calculate portfolio allocation if we have multiple pairs
        if len(results) > 1:
            logger.info("Optimizing portfolio allocation...")
            allocations = calculate_optimal_allocations(
                results, method=self.allocation_method, lookback_window=60
            )
            
            # Backtest portfolio with optimal allocations
            if allocations is not None:
                portfolio_results = backtest_portfolio(
                    results, allocations, initial_capital, transaction_cost
                )
                
                if portfolio_results is not None:
                    results['portfolio'] = portfolio_results
        
        # Calculate aggregate results if no portfolio optimization
        if 'portfolio' not in results:
            self._calculate_aggregate_results(results, initial_capital)
        
        return results
    
    def execute_strategy(self, tickers, start_date, end_date, top_n_pairs=10, 
                        market_neutral=True, sector_aware=True):
        """
        Execute the enhanced strategy with all improvements.
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        top_n_pairs : int
            Number of top pairs to include
        market_neutral : bool
            Whether to maintain market neutrality
        sector_aware : bool
            Whether to use sector information
            
        Returns:
        --------
        Dictionary with strategy results
        """
        # Step 1: Fetch data with expanded universe
        logger.info("Fetching price data...")
        expanded_tickers = expand_universe(tickers)
        price_data = self.fetch_data(expanded_tickers, start_date, end_date)
        
        if price_data.empty:
            logger.error("Failed to fetch price data, aborting strategy execution")
            return {"error": "Failed to fetch price data"}
        
        # Step 2: Filter universe based on liquidity, etc.
        logger.info("Filtering universe...")
        filtered_tickers = filter_universe(price_data, min_price=5.0)
        price_data = price_data[filtered_tickers]
        
        # Step 3: Fetch macro/market data for regime detection
        logger.info("Fetching macroeconomic data...")
        macro_data = self.fetch_macro_data(start_date, end_date)
        
        # Step 4: Detect market regime if enabled
        if self.market_regime_aware and macro_data is not None:
            logger.info("Detecting market regime...")
            self.market_regime = detect_market_regime(macro_data)
            
            # Adapt parameters to current regime
            logger.info("Adapting parameters to market regime...")
            adapted_params = self.adapt_parameters_to_regime()
            self.z_entry = adapted_params['z_entry']
            self.z_exit = adapted_params['z_exit']
            self.holding_period = adapted_params['holding_period']
        
        # Step 5: Get sector mapping if sector-aware
        sector_map = None
        if sector_aware:
            logger.info("Getting sector information...")
            sector_map = get_sector_mapping(price_data.columns)
        
        # Step 6: Find pairs with enhanced criteria
        logger.info("Finding cointegrated pairs with enhanced criteria...")
        pairs_with_data = find_enhanced_pairs(
            price_data, self.lookback_period, self.min_half_life, self.max_half_life,
            self.confidence, self.correlation_filter, self.min_correlation,
            self.cointegration_method, sector_map, macro_data, self.beta_neutral
        )
        
        if not pairs_with_data:
            logger.error("No suitable pairs found")
            return {"error": "No suitable pairs found", "price_data": price_data}
        
        # Select top pairs
        selected_pairs = [p['pair'] for p in pairs_with_data[:min(top_n_pairs, len(pairs_with_data))]]
        
        # Step 7: Generate signals
        logger.info("Generating trading signals...")
        pair_signals = self.generate_signals(price_data, selected_pairs, macro_data)
        
        # Step 8: Backtest strategy
        logger.info("Backtesting strategy...")
        results = self.backtest(price_data, pair_signals)
        
        return results
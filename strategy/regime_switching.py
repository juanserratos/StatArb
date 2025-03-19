"""
Implementation of the original regime-switching statistical arbitrage strategy.
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
from tqdm import tqdm

from strategy.base import BaseStrategy
from pairs.cointegration import find_cointegrated_pairs
from regime.detector import detect_regimes
from signals.generator import generate_signals
from execution.simulator import backtest_pairs

logger = logging.getLogger(__name__)


class RegimeSwitchingStatArb(BaseStrategy):
    """
    A statistical arbitrage strategy that implements regime-switching cointegration models
    to enhance traditional pairs trading.
    """
    
    def __init__(self, lookback_period=252, z_entry=2.0, z_exit=0.5, 
                 n_regimes=2, confidence=0.05, holding_period=20,
                 name="RegimeSwitchingStatArb"):
        """
        Initialize the strategy parameters.
        
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
        name : str
            Name of the strategy
        """
        super().__init__(name=name)
        
        self.lookback_period = lookback_period
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.n_regimes = n_regimes
        self.confidence = confidence
        self.holding_period = holding_period
        
        logger.info(f"Strategy initialized with parameters: lookback={lookback_period}, "
                   f"z_entry={z_entry}, z_exit={z_exit}, n_regimes={n_regimes}, "
                   f"confidence={confidence}, holding_period={holding_period}")
    
    def fetch_data(self, tickers, start_date, end_date, retry_count=3, retry_delay=10):
        """
        Fetch price data for the given tickers with robust error handling.
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        retry_count : int
            Number of retries for failed downloads
        retry_delay : int
            Delay in seconds between retries
            
        Returns:
        --------
        DataFrame with adjusted close prices
        """
        # Add buffer for lookback
        adj_start_date = (datetime.strptime(start_date, '%Y-%m-%d') - 
                         timedelta(days=int(self.lookback_period * 1.5))).strftime('%Y-%m-%d')
        
        logger.info(f"Fetching price data for {len(tickers)} tickers "
                   f"from {adj_start_date} to {end_date}")
        
        # Use the market data handler to fetch the data
        price_data = self.market_data_handler.fetch_price_data(
            tickers, adj_start_date, end_date, retry_count, retry_delay
        )
        
        # Filter to the date range of interest
        if not price_data.empty:
            mask = (price_data.index >= start_date) & (price_data.index <= end_date)
            price_data = price_data.loc[mask]
        
        # Check if we have any data
        if price_data.empty:
            logger.error("No data was successfully downloaded")
            return pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq='B'))
        
        logger.info(f"Successfully fetched data for {len(price_data.columns)}/{len(tickers)} tickers")
        logger.info(f"Date range: {price_data.index[0]} to {price_data.index[-1]}")
        logger.info(f"Data shape: {price_data.shape}")
        
        return price_data
    
    def fetch_macro_data(self, start_date, end_date, retry_count=3, retry_delay=10):
        """
        Fetch macroeconomic indicators that may influence market regimes.
        
        Parameters:
        -----------
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        retry_count : int
            Number of retries for failed downloads
        retry_delay : int
            Delay in seconds between retries
            
        Returns:
        --------
        DataFrame with macroeconomic indicators
        """
        # Add buffer for lookback
        adj_start_date = (datetime.strptime(start_date, '%Y-%m-%d') - 
                         timedelta(days=int(self.lookback_period * 1.5))).strftime('%Y-%m-%d')
        
        # List of indicators to fetch
        indicators = {
            '^VIX': 'VIX',          # Market volatility
            '^TNX': 'TNX',          # 10-year Treasury yield
            'SPY': 'SPY',           # S&P 500
            'TLT': 'TLT',           # Long-term Treasury bonds
            'DX-Y.NYB': 'USD_Index' # US Dollar index
        }
        
        logger.info(f"Fetching macroeconomic data from {adj_start_date} to {end_date}")
        
        # Use the market data handler to fetch the data
        macro_data = self.market_data_handler.fetch_price_data(
            list(indicators.keys()), adj_start_date, end_date, retry_count, retry_delay
        )
        
        # Rename columns according to the indicators mapping
        macro_data = macro_data.rename(columns=indicators)
        
        # Filter to requested date range
        if not macro_data.empty:
            mask = (macro_data.index >= start_date) & (macro_data.index <= end_date)
            macro_data = macro_data.loc[mask]
        
        # Calculate additional derived features
        try:
            if 'SPY' in macro_data.columns:
                # Rolling volatility (20-day)
                macro_data['SPY_Vol_20d'] = macro_data['SPY'].pct_change().rolling(20).std() * np.sqrt(252)
                
                # Rolling returns
                macro_data['SPY_Ret_20d'] = macro_data['SPY'].pct_change(20)
            
            if 'VIX' in macro_data.columns and 'SPY_Vol_20d' in macro_data.columns:
                # VIX vs. realized vol gap
                macro_data['VIX_Vol_Gap'] = macro_data['VIX'] / 100 - macro_data['SPY_Vol_20d']
                
            if 'TNX' in macro_data.columns:
                # Change in yield
                macro_data['TNX_Change_20d'] = macro_data['TNX'].diff(20)
                
            logger.info(f"Successfully fetched and processed macro data with shape {macro_data.shape}")
        except Exception as e:
            logger.warning(f"Error calculating derived features: {str(e)}")
        
        return macro_data.dropna(how='all')
    
    def generate_signals(self, data, pairs=None, macro_data=None, top_n_pairs=5):
        """
        Generate trading signals for statistical arbitrage pairs.
        
        Parameters:
        -----------
        data : DataFrame
            DataFrame with price data
        pairs : list, optional
            List of predefined pairs to use, if None will find pairs
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
            coint_pairs = find_cointegrated_pairs(data, lookback=self.lookback_period, 
                                                 pvalue_threshold=self.confidence)
            pairs = [pair for pair, _ in coint_pairs[:min(top_n_pairs, len(coint_pairs))]]
        
        logger.info(f"Generating signals for {len(pairs)} pairs")
        
        # Generate signals for each pair
        pair_signals = {}
        
        for pair in pairs:
            stock1, stock2 = pair
            
            # Check if pair exists in data
            if stock1 not in data.columns or stock2 not in data.columns:
                logger.warning(f"Pair {pair} not found in data, skipping")
                continue
            
            # Calculate rolling hedge ratio
            hedge_ratios = self._calculate_hedge_ratio(data, pair)
            
            # Calculate spread
            spread = self._calculate_spread(data, pair, hedge_ratios)
            
            if spread is None:
                logger.warning(f"Failed to calculate spread for {pair}, skipping")
                continue
            
            # Detect regimes
            model, hidden_states, scaler, _ = detect_regimes(spread, macro_data, 
                                                          n_regimes=self.n_regimes)
            
            if model is None:
                logger.warning(f"Failed to detect regimes for {pair}, skipping")
                continue
            
            # Generate trading signals
            signals = generate_signals(spread, hidden_states, self.z_entry, self.z_exit,
                                     holding_period=self.holding_period)
            
            if signals is None:
                logger.warning(f"Failed to generate signals for {pair}, skipping")
                continue
            
            pair_signals[str(pair)] = {
                'spread': spread,
                'hedge_ratios': hedge_ratios,
                'regimes': hidden_states,
                'signals': signals
            }
        
        return pair_signals
    
    def backtest(self, data, pair_signals, initial_capital=1000000, transaction_cost=0.0005):
        """
        Backtest the strategy with historical data.
        
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
        
        # Backtest each pair
        for pair_str, signals_data in pair_signals.items():
            pair = eval(pair_str)  # Convert string back to tuple
            spread = signals_data['spread']
            hedge_ratios = signals_data['hedge_ratios']
            signals = signals_data['signals']
            
            # Run backtest for this pair
            pair_results = backtest_pairs(data, pair, signals, hedge_ratios, 
                                         transaction_cost, initial_capital)
            
            if pair_results is not None:
                results[pair_str] = pair_results
        
        # Calculate aggregate results
        self._calculate_aggregate_results(results, initial_capital)
        
        return results
    
    def _calculate_hedge_ratio(self, price_data, pair, window=60):
        """
        Calculate the hedge ratio between two assets using rolling OLS.
        
        Parameters:
        -----------
        price_data : DataFrame
            DataFrame with price series
        pair : tuple
            Tuple containing the ticker symbols of the pair
        window : int
            Window size for the rolling regression
            
        Returns:
        --------
        Series with the hedge ratios
        """
        stock1, stock2 = pair
        
        # Check if pair exists in data
        if stock1 not in price_data.columns or stock2 not in price_data.columns:
            logger.error(f"Pair {stock1}-{stock2} not found in price data")
            return None
        
        # Calculate rolling hedge ratio using OLS
        hedge_ratios = pd.Series(index=price_data.index)
        
        # Ensure we have sufficient data
        if len(price_data) < window:
            logger.warning(f"Insufficient data for hedge ratio calculation: {len(price_data)} < {window}")
            return None
        
        for i in range(window, len(price_data)):
            train = price_data.iloc[i-window:i]
            
            # Skip if we have NaN values
            if train[stock1].isnull().any() or train[stock2].isnull().any():
                continue
                
            X = train[stock1].values.reshape(-1, 1)
            y = train[stock2].values
            
            try:
                # OLS regression
                beta = np.linalg.lstsq(X, y, rcond=None)[0][0]
                hedge_ratios.iloc[i] = beta
            except Exception as e:
                logger.warning(f"Error calculating hedge ratio at index {i}: {str(e)}")
                continue
        
        # Check if we have sufficient valid hedge ratios
        valid_ratios = hedge_ratios.dropna()
        if len(valid_ratios) < window / 2:
            logger.warning(f"Too few valid hedge ratios: {len(valid_ratios)}")
            return None
            
        logger.info(f"Calculated hedge ratios for {pair}: mean={valid_ratios.mean():.4f}, std={valid_ratios.std():.4f}")
        return hedge_ratios.dropna()
    
    def _calculate_spread(self, price_data, pair, hedge_ratios=None):
        """
        Calculate the spread between two assets, optionally using dynamic hedge ratios.
        
        Parameters:
        -----------
        price_data : DataFrame
            DataFrame with price series
        pair : tuple
            Tuple containing the ticker symbols of the pair
        hedge_ratios : Series, optional
            Series with hedge ratios for each date
            
        Returns:
        --------
        Series with the spread
        """
        stock1, stock2 = pair
        
        # Check if pair exists in data
        if stock1 not in price_data.columns or stock2 not in price_data.columns:
            logger.error(f"Pair {stock1}-{stock2} not found in price data")
            return None
        
        try:
            if hedge_ratios is None:
                # If no hedge ratios provided, use full-sample OLS
                valid_mask = ~price_data[stock1].isnull() & ~price_data[stock2].isnull()
                X = price_data.loc[valid_mask, stock1].values.reshape(-1, 1)
                y = price_data.loc[valid_mask, stock2].values
                
                if len(X) < 30:  # Minimum required observations
                    logger.warning(f"Insufficient valid observations for hedge ratio calculation: {len(X)}")
                    return None
                
                beta = np.linalg.lstsq(X, y, rcond=None)[0][0]
                spread = price_data[stock2] - beta * price_data[stock1]
                logger.info(f"Calculated spread using static hedge ratio: {beta:.4f}")
            else:
                # Use dynamic hedge ratios - align indices
                common_idx = price_data.index.intersection(hedge_ratios.index)
                if len(common_idx) < 30:
                    logger.warning(f"Insufficient overlapping observations: {len(common_idx)}")
                    return None
                    
                price_subset = price_data.loc[common_idx]
                hedge_subset = hedge_ratios.loc[common_idx]
                
                # Calculate spread
                spread = price_subset[stock2] - hedge_subset * price_subset[stock1]
                logger.info(f"Calculated spread using dynamic hedge ratios")
            
            return spread
            
        except Exception as e:
            logger.error(f"Error calculating spread for {stock1}-{stock2}: {str(e)}")
            return None
    
    def _calculate_aggregate_results(self, results, initial_capital):
        """
        Calculate aggregate results from individual pair results.
        
        Parameters:
        -----------
        results : dict
            Dictionary with results for each pair
        initial_capital : float
            Initial capital
            
        Returns:
        --------
        None, updates results dict in place
        """
        portfolio_equity = None  # Aggregate portfolio
        
        # Process each pair
        for pair_str, backtest_results in results.items():
            # Add to aggregate portfolio (equal weight)
            if portfolio_equity is None:
                portfolio_equity = backtest_results['portfolio'] / len(results)
            else:
                # Align dates
                pair_equity = backtest_results['portfolio'] / len(results)
                common_dates = portfolio_equity.index.intersection(pair_equity.index)
                portfolio_equity = portfolio_equity[portfolio_equity.index.isin(common_dates)]
                pair_equity = pair_equity[pair_equity.index.isin(common_dates)]
                portfolio_equity = portfolio_equity.add(pair_equity, fill_value=0)
        
        # Calculate aggregate metrics
        if portfolio_equity is not None and len(portfolio_equity) > 0:
            aggregate_returns = portfolio_equity.pct_change().fillna(0)
            total_return = (portfolio_equity.iloc[-1] / initial_capital) - 1
            annual_return = ((1 + total_return) ** (252 / len(portfolio_equity))) - 1
            daily_returns = aggregate_returns[1:]  # Skip first day
            annual_volatility = daily_returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
            
            # Maximum drawdown
            running_max = portfolio_equity.cummax()
            drawdown = (portfolio_equity / running_max) - 1
            max_drawdown = drawdown.min()
            
            # Store aggregate results
            results['aggregate'] = {
                'portfolio': portfolio_equity,
                'returns': aggregate_returns,
                'metrics': {
                    'total_return': total_return,
                    'annual_return': annual_return,
                    'annual_volatility': annual_volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'calmar_ratio': -annual_return / max_drawdown if max_drawdown != 0 else 0,
                    'profit_factor': sum(r for r in daily_returns if r > 0) / abs(sum(r for r in daily_returns if r < 0)) if sum(r for r in daily_returns if r < 0) != 0 else float('inf'),
                    'win_rate': sum(r > 0 for r in daily_returns) / len(daily_returns)
                }
            }
            
            logger.info(f"Aggregate metrics: " +
                       f"Return={total_return:.4f}, " +
                       f"Annual={annual_return:.4f}, " +
                       f"Sharpe={sharpe_ratio:.4f}, " +
                       f"Drawdown={max_drawdown:.4f}")
    
    def run_strategy(self, tickers, start_date, end_date, top_n_pairs=5):
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
        top_n_pairs : int
            Number of top pairs to backtest
            
        Returns:
        --------
        Dictionary with strategy results
        """
        # 1. Fetch data
        logger.info("Fetching price data...")
        price_data = self.fetch_data(tickers, start_date, end_date)
        
        if price_data.empty:
            logger.error("Failed to fetch price data, aborting strategy execution")
            return {"error": "Failed to fetch price data"}
        
        logger.info("Fetching macroeconomic data...")
        macro_data = self.fetch_macro_data(start_date, end_date)
        
        # 2. Find cointegrated pairs
        logger.info("Finding cointegrated pairs...")
        coint_pairs = find_cointegrated_pairs(price_data, lookback=self.lookback_period, 
                                             pvalue_threshold=self.confidence)
        
        logger.info(f"Found {len(coint_pairs)} cointegrated pairs.")
        if len(coint_pairs) == 0:
            return {"error": "No cointegrated pairs found", "price_data": price_data}
        
        # Select top N pairs
        selected_pairs = [pair for pair, _ in coint_pairs[:min(top_n_pairs, len(coint_pairs))]]
        
        # 3. Generate signals
        logger.info("Generating trading signals...")
        pair_signals = self.generate_signals(price_data, selected_pairs, macro_data)
        
        # 4. Backtest
        logger.info("Backtesting strategy...")
        results = self.backtest(price_data, pair_signals)
        
        return results
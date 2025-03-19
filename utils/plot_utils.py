"""
Utility functions for plotting and visualizing strategy results.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def plot_strategy_performance(results, save_path=None):
    """
    Create a comprehensive visualization of strategy performance.
    
    Parameters:
    -----------
    results : dict
        Dictionary with strategy results
    save_path : str, optional
        Path to save the plot
    """
    if "portfolio" in results:
        portfolio_equity = results["portfolio"]["portfolio"]
        metrics = results["portfolio"]["metrics"]
    elif "aggregate" in results:
        portfolio_equity = results["aggregate"]["portfolio"]
        metrics = results["aggregate"]["metrics"]
    else:
        print("No portfolio equity data found in results")
        return
    
    # Calculate returns
    returns = portfolio_equity.pct_change().fillna(0)
    
    # Set up figure
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Equity Curve
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax1.plot(portfolio_equity, linewidth=2)
    ax1.set_title('Portfolio Equity Curve', fontsize=14)
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_xlabel('')
    ax1.grid(True)
    
    # Annotate key metrics on the plot
    metrics_text = (
        f"Total Return: {metrics['total_return']:.2%}  |  "
        f"Ann. Return: {metrics['annual_return']:.2%}  |  "
        f"Ann. Vol: {metrics['annual_volatility']:.2%}  |  "
        f"Sharpe: {metrics['sharpe_ratio']:.2f}  |  "
        f"Max DD: {metrics['max_drawdown']:.2%}"
    )
    ax1.annotate(metrics_text, xy=(0.5, 0.95), xycoords='axes fraction',
                ha='center', va='top', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # 2. Drawdown Plot
    ax2 = plt.subplot2grid((3, 2), (1, 0), colspan=2)
    running_max = portfolio_equity.cummax()
    drawdown = (portfolio_equity / running_max) - 1
    ax2.fill_between(drawdown.index, 0, drawdown, color='red', alpha=0.3)
    ax2.plot(drawdown, color='red', linewidth=1.5)
    ax2.set_title('Drawdown', fontsize=14)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_ylim([min(drawdown)*1.1, 0.01])
    ax2.grid(True)
    
    # 3. Monthly Returns Heatmap
    ax3 = plt.subplot2grid((3, 2), (2, 0))
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    monthly_returns_df = pd.DataFrame({
        'Year': monthly_returns.index.year,
        'Month': monthly_returns.index.month,
        'Return': monthly_returns.values
    })
    
    if not monthly_returns_df.empty:
        pivot_table = monthly_returns_df.pivot('Year', 'Month', 'Return')
        sns.heatmap(pivot_table, annot=True, fmt='.1%', cmap='RdYlGn', center=0, ax=ax3)
        ax3.set_title('Monthly Returns', fontsize=14)
        ax3.set_ylabel('Year')
        ax3.set_xlabel('Month')
    
    # 4. Distribution of Returns
    ax4 = plt.subplot2grid((3, 2), (2, 1))
    sns.histplot(returns * 100, bins=50, kde=True, ax=ax4)
    ax4.axvline(0, color='r', linestyle='--')
    ax4.set_title('Daily Returns Distribution', fontsize=14)
    ax4.set_xlabel('Daily Return (%)')
    ax4.set_ylabel('Frequency')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    plt.show()

def plot_pair_analysis(pair, price_data, spread, signals, hedge_ratios=None, regimes=None):
    """
    Create detailed visualizations for a specific trading pair.
    
    Parameters:
    -----------
    pair : tuple
        Tuple containing ticker symbols
    price_data : DataFrame
        DataFrame with price data
    spread : Series
        Series with spread values
    signals : Series
        Series with trading signals
    hedge_ratios : Series, optional
        Series with hedge ratios
    regimes : array, optional
        Array with regime classifications
    """
    stock1, stock2 = pair
    
    # Create figure
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Price Series
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    
    # Normalize to start at 100
    price1 = price_data[stock1] / price_data[stock1].iloc[0] * 100
    price2 = price_data[stock2] / price_data[stock2].iloc[0] * 100
    
    ax1.plot(price1, label=stock1)
    ax1.plot(price2, label=stock2)
    ax1.set_title(f'Normalized Price Series: {stock1} vs {stock2}', fontsize=14)
    ax1.legend()
    ax1.grid(True)
    
    # 2. Spread and Z-Score
    ax2 = plt.subplot2grid((3, 2), (1, 0), colspan=2)
    
    # Calculate z-score (could be regime-specific, simplified here)
    z_score = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
    
    ax2.plot(spread, label='Spread', color='blue', alpha=0.6)
    ax2_right = ax2.twinx()
    ax2_right.plot(z_score, label='Z-Score', color='red')
    
    ax2.set_title('Spread and Z-Score', fontsize=14)
    ax2.set_ylabel('Spread')
    ax2_right.set_ylabel('Z-Score')
    
    # Add horizontal lines for z-score thresholds
    ax2_right.axhline(2.0, color='green', linestyle='--', alpha=0.5)
    ax2_right.axhline(-2.0, color='green', linestyle='--', alpha=0.5)
    ax2_right.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax2_right.axhline(-0.5, color='gray', linestyle='--', alpha=0.5)
    
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_right.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc='upper left')
    ax2.grid(True)
    
    # 3. Signals and Trades
    ax3 = plt.subplot2grid((3, 2), (2, 0))
    
    # Create a scatter plot for trade signals
    buy_signals = signals[signals == 1].index
    sell_signals = signals[signals == -1].index
    exit_signals = signals[(signals.shift(1) != 0) & (signals == 0)].index
    
    ax3.plot(spread, color='gray', alpha=0.5)
    ax3.scatter(buy_signals, spread[buy_signals], color='green', s=50, label='Long Signal')
    ax3.scatter(sell_signals, spread[sell_signals], color='red', s=50, label='Short Signal')
    ax3.scatter(exit_signals, spread[exit_signals], color='blue', s=30, label='Exit Signal')
    
    ax3.set_title('Trading Signals', fontsize=14)
    ax3.legend()
    ax3.grid(True)
    
    # 4. Regime Analysis (if regimes provided)
    ax4 = plt.subplot2grid((3, 2), (2, 1))
    
    if regimes is not None and len(regimes) > 0:
        # Create a colormap for regimes
        unique_regimes = np.unique(regimes)
        regime_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_regimes)))
        
        # Plot spread
        ax4.plot(spread, color='gray', alpha=0.3)
        
        # Plot regime background colors
        regime_dates = spread.index
        for i, regime in enumerate(unique_regimes):
            mask = (regimes == regime)
            if sum(mask) > 0:
                # Get contiguous segments
                regime_segments = []
                in_segment = False
                seg_start = None
                
                for j, val in enumerate(mask):
                    if val and not in_segment:
                        in_segment = True
                        seg_start = j
                    elif not val and in_segment:
                        in_segment = False
                        regime_segments.append((seg_start, j-1))
                
                # Add last segment if it's open
                if in_segment:
                    regime_segments.append((seg_start, len(mask)-1))
                
                # Plot segments
                for start, end in regime_segments:
                    ax4.axvspan(regime_dates[start], regime_dates[end], 
                               alpha=0.3, color=regime_colors[i])
        
        # Add regime labels
        regime_patches = [plt.Rectangle((0,0), 1, 1, fc=regime_colors[i], alpha=0.3) 
                         for i in range(len(unique_regimes))]
        ax4.legend(regime_patches, [f'Regime {i+1}' for i in range(len(unique_regimes))], 
                  loc='best')
    
    else:
        # If no regimes, show hedge ratio
        if hedge_ratios is not None:
            ax4.plot(hedge_ratios, label='Hedge Ratio')
            ax4.set_title('Hedge Ratio Evolution', fontsize=14)
            ax4.legend()
            ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_regime_analysis(spread, regimes, regime_labels=None):
    """
    Visualize detected regimes and their characteristics.
    
    Parameters:
    -----------
    spread : Series
        Series with spread values
    regimes : array
        Array with regime classifications
    regime_labels : dict, optional
        Dictionary mapping regime IDs to labels
    """
    if regimes is None or len(regimes) == 0:
        print("No regime data provided")
        return
    
    # Create regime DataFrame
    regime_df = pd.DataFrame({
        'spread': spread.values,
        'regime': regimes
    }, index=spread.index)
    
    # Get unique regimes
    unique_regimes = np.unique(regimes)
    
    # Set up figure
    plt.figure(figsize=(15, 10))
    
    # 1. Spread with regime coloring
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    
    # Plot base spread
    ax1.plot(spread, color='black', alpha=0.4)
    
    # Create colormap
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_regimes)))
    
    # Highlight regimes with different colors
    for i, regime in enumerate(unique_regimes):
        if regime_labels and regime in regime_labels:
            label = regime_labels[regime]
        else:
            label = f"Regime {regime}"
            
        regime_data = regime_df[regime_df['regime'] == regime]
        ax1.scatter(regime_data.index, regime_data['spread'], 
                   label=label, color=colors[i], s=15, alpha=0.7)
    
    ax1.set_title('Spread with Regime Classification', fontsize=14)
    ax1.legend()
    ax1.grid(True)
    
    # 2. Regime distributions
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    
    # Create box plots for spread by regime
    data = [regime_df[regime_df['regime'] == r]['spread'] for r in unique_regimes]
    
    if regime_labels:
        labels = [regime_labels.get(r, f"Regime {r}") for r in unique_regimes]
    else:
        labels = [f"Regime {r}" for r in unique_regimes]
    
    ax2.boxplot(data, labels=labels)
    ax2.set_title('Spread Distribution by Regime', fontsize=14)
    ax2.set_ylabel('Spread Value')
    ax2.grid(True, axis='y')
    
    # 3. Regime transition matrix
    ax3 = plt.subplot2grid((2, 2), (1, 1))
    
    # Calculate transition matrix
    transitions = np.zeros((len(unique_regimes), len(unique_regimes)))
    
    for i in range(1, len(regimes)):
        from_regime = regimes[i-1]
        to_regime = regimes[i]
        
        if from_regime != to_regime:  # Only count actual transitions
            from_idx = np.where(unique_regimes == from_regime)[0][0]
            to_idx = np.where(unique_regimes == to_regime)[0][0]
            transitions[from_idx, to_idx] += 1
    
    # Normalize rows to get probabilities
    row_sums = transitions.sum(axis=1, keepdims=True)
    transition_probs = np.zeros_like(transitions)
    np.divide(transitions, row_sums, out=transition_probs, where=row_sums!=0)
    
    # Plot heatmap
    sns.heatmap(transition_probs, annot=True, fmt='.2f', cmap='Blues',
               xticklabels=labels, yticklabels=labels, ax=ax3)
    ax3.set_title('Regime Transition Probabilities', fontsize=14)
    ax3.set_xlabel('To Regime')
    ax3.set_ylabel('From Regime')
    
    plt.tight_layout()
    plt.show()
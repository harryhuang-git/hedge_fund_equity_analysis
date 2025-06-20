"""
Interactive dashboard for visualizing trading results and strategy performance.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List
import json
import os
from datetime import datetime

def load_backtest_results(strategy_name: str) -> Dict:
    """Load backtest results from JSON files."""
    metrics_file = f"outputs/metrics/{strategy_name}_latest.json"
    trades_file = f"outputs/trades/{strategy_name}_latest.json"
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    with open(trades_file, 'r') as f:
        trades = json.load(f)
    
    return metrics, trades

def plot_equity_curve(equity_curve: pd.DataFrame):
    """Create interactive equity curve plot."""
    fig = go.Figure()
    
    # Add equity curve
    fig.add_trace(go.Scatter(
        x=equity_curve.index,
        y=equity_curve['value'],
        name='Portfolio Value',
        line=dict(color='blue')
    ))
    
    # Add drawdown
    fig.add_trace(go.Scatter(
        x=equity_curve.index,
        y=equity_curve['drawdown'] * 100,
        name='Drawdown (%)',
        line=dict(color='red'),
        yaxis='y2'
    ))
    
    # Update layout
    fig.update_layout(
        title='Portfolio Performance',
        xaxis_title='Date',
        yaxis_title='Portfolio Value',
        yaxis2=dict(
            title='Drawdown (%)',
            overlaying='y',
            side='right',
            range=[-100, 0]
        ),
        hovermode='x unified'
    )
    
    return fig

def plot_trade_heatmap(trades: pd.DataFrame):
    """Create trade heatmap by stock and date."""
    # Create pivot table for heatmap
    trade_pivot = trades.pivot_table(
        values='pnl',
        index=trades['timestamp'].dt.date,
        columns='ticker',
        aggfunc='sum'
    )
    
    # Create heatmap
    fig = px.imshow(
        trade_pivot,
        title='Trade PnL Heatmap',
        labels=dict(x='Stock', y='Date', color='PnL'),
        color_continuous_scale='RdYlGn'
    )
    
    return fig

def plot_factor_scores(factor_scores: pd.DataFrame):
    """Plot factor scores over time."""
    fig = go.Figure()
    
    for factor in factor_scores.columns:
        fig.add_trace(go.Scatter(
            x=factor_scores.index,
            y=factor_scores[factor],
            name=factor,
            mode='lines'
        ))
    
    fig.update_layout(
        title='Factor Scores Over Time',
        xaxis_title='Date',
        yaxis_title='Score',
        hovermode='x unified'
    )
    
    return fig

def plot_monthly_returns(equity_curve: pd.DataFrame):
    """Create monthly returns heatmap."""
    # Calculate monthly returns
    monthly_returns = equity_curve['returns'].resample('M').apply(
        lambda x: (1 + x).prod() - 1
    )
    
    # Create monthly returns matrix
    returns_matrix = monthly_returns.to_frame()
    returns_matrix.index = pd.to_datetime(returns_matrix.index)
    returns_matrix['year'] = returns_matrix.index.year
    returns_matrix['month'] = returns_matrix.index.month
    returns_matrix = returns_matrix.pivot(
        index='year',
        columns='month',
        values='returns'
    )
    
    # Create heatmap
    fig = px.imshow(
        returns_matrix * 100,
        title='Monthly Returns (%)',
        labels=dict(x='Month', y='Year', color='Return (%)'),
        color_continuous_scale='RdYlGn'
    )
    
    return fig

def main():
    st.set_page_config(page_title="HFT Strategy Dashboard", layout="wide")
    
    st.title("HFT Strategy Performance Dashboard")
    
    # Sidebar for strategy selection
    strategy_files = [f for f in os.listdir("outputs/metrics") if f.endswith("_latest.json")]
    strategies = [f.split("_latest.json")[0] for f in strategy_files]
    
    selected_strategy = st.sidebar.selectbox(
        "Select Strategy",
        strategies
    )
    
    if selected_strategy:
        # Load backtest results
        metrics, trades = load_backtest_results(selected_strategy)
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Return", f"{metrics['total_return']:.2%}")
        with col2:
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        with col3:
            st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
        with col4:
            st.metric("Win Rate", f"{metrics['win_rate']:.2%}")
        
        # Convert data to DataFrames
        equity_curve = pd.DataFrame(metrics['equity_curve'])
        equity_curve['timestamp'] = pd.to_datetime(equity_curve['timestamp'])
        equity_curve.set_index('timestamp', inplace=True)
        
        trades_df = pd.DataFrame(trades)
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "Equity Curve",
            "Trade Analysis",
            "Factor Analysis",
            "Monthly Returns"
        ])
        
        with tab1:
            st.plotly_chart(plot_equity_curve(equity_curve), use_container_width=True)
        
        with tab2:
            st.plotly_chart(plot_trade_heatmap(trades_df), use_container_width=True)
            
            # Trade statistics
            st.subheader("Trade Statistics")
            trade_stats = trades_df.groupby('ticker').agg({
                'pnl': ['count', 'sum', 'mean'],
                'shares': 'sum'
            }).round(2)
            st.dataframe(trade_stats)
        
        with tab3:
            if 'factor_scores' in metrics:
                factor_scores = pd.DataFrame(metrics['factor_scores'])
                st.plotly_chart(plot_factor_scores(factor_scores), use_container_width=True)
            else:
                st.info("Factor scores not available for this strategy")
        
        with tab4:
            st.plotly_chart(plot_monthly_returns(equity_curve), use_container_width=True)

if __name__ == "__main__":
    main() 
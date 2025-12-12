"""
Visualization module for CF Recommender Lab.

Contains functions to create various plots and charts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from typing import Optional, Dict, List

def plot_similarity_heatmap(similarity_df: pd.DataFrame, title: str, max_users: int = 20):
    """
    Plot heatmap of user-user or item-item similarity matrix.

    Args:
        similarity_df: Similarity matrix
        title: Plot title
        max_users: Maximum number of users/items to display
    """
    # Subset for visualization
    subset = similarity_df.iloc[:max_users, :max_users]

    fig = px.imshow(subset,
                   text_auto=True,
                   aspect="auto",
                   title=title,
                   color_continuous_scale='RdBu_r')

    st.plotly_chart(fig, use_container_width=True)

def plot_metrics_comparison(ubcf_metrics: Dict, ibcf_metrics: Dict):
    """
    Plot bar chart comparing UBCF and IBCF metrics.

    Args:
        ubcf_metrics: UBCF evaluation metrics
        ibcf_metrics: IBCF evaluation metrics
    """
    metrics = ['rmse', 'mae', 'precision@5', 'recall@5', 'f1@5', 'ndcg@5', 'coverage']
    ubcf_values = [ubcf_metrics.get(m, 0) for m in metrics]
    ibcf_values = [ibcf_metrics.get(m, 0) for m in metrics]

    fig = go.Figure(data=[
        go.Bar(name='UBCF', x=metrics, y=ubcf_values, marker_color='blue'),
        go.Bar(name='IBCF', x=metrics, y=ibcf_values, marker_color='red')
    ])

    fig.update_layout(
        title='UBCF vs IBCF Performance Comparison',
        xaxis_title='Metrics',
        yaxis_title='Values',
        barmode='group'
    )

    st.plotly_chart(fig, use_container_width=True)

def plot_precision_recall_vs_k(precisions_ubcf: List[float], recalls_ubcf: List[float],
                             precisions_ibcf: List[float], recalls_ibcf: List[float],
                             k_values: List[int]):
    """
    Plot precision and recall vs K for both algorithms.

    Args:
        precisions_ubcf: UBCF precision values
        recalls_ubcf: UBCF recall values
        precisions_ibcf: IBCF precision values
        recalls_ibcf: IBCF recall values
        k_values: K values
    """
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Precision@K', 'Recall@K'))

    fig.add_trace(
        go.Scatter(x=k_values, y=precisions_ubcf, mode='lines+markers', name='UBCF Precision', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=k_values, y=precisions_ibcf, mode='lines+markers', name='IBCF Precision', line=dict(color='red')),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=k_values, y=recalls_ubcf, mode='lines+markers', name='UBCF Recall', line=dict(color='blue')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=k_values, y=recalls_ibcf, mode='lines+markers', name='IBCF Recall', line=dict(color='red')),
        row=1, col=2
    )

    fig.update_layout(title='Precision and Recall vs K')
    fig.update_xaxes(title_text='K', row=1, col=1)
    fig.update_xaxes(title_text='K', row=1, col=2)
    fig.update_yaxes(title_text='Precision', row=1, col=1)
    fig.update_yaxes(title_text='Recall', row=1, col=2)

    st.plotly_chart(fig, use_container_width=True)

def plot_predicted_vs_actual(predictions: np.ndarray, actuals: np.ndarray):
    """
    Plot scatter plot of predicted vs actual ratings.

    Args:
        predictions: Predicted ratings
        actuals: Actual ratings
    """
    fig = px.scatter(x=actuals, y=predictions,
                    labels={'x': 'Actual Ratings', 'y': 'Predicted Ratings'},
                    title='Predicted vs Actual Ratings')

    # Add diagonal line
    min_val = min(np.min(actuals), np.min(predictions))
    max_val = max(np.max(actuals), np.max(predictions))
    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                           mode='lines', name='Perfect Prediction', line=dict(dash='dash', color='red')))

    st.plotly_chart(fig, use_container_width=True)

def plot_rating_distributions(df: pd.DataFrame, predictions: Optional[np.ndarray] = None):
    """
    Plot distribution of ratings and optionally predictions.

    Args:
        df: DataFrame with ratings
        predictions: Predicted ratings array
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Actual ratings distribution
    sns.histplot(df['rating'], bins=np.arange(0, 5.5, 0.5), ax=ax1, kde=True)
    ax1.set_title('Actual Ratings Distribution')
    ax1.set_xlabel('Rating')
    ax1.set_ylabel('Frequency')

    # Predicted ratings distribution
    if predictions is not None and len(predictions) > 0:
        sns.histplot(predictions, bins=np.arange(0, 5.5, 0.5), ax=ax2, kde=True, color='orange')
        ax2.set_title('Predicted Ratings Distribution')
        ax2.set_xlabel('Rating')
        ax2.set_ylabel('Frequency')
    else:
        ax2.text(0.5, 0.5, 'No predictions available', transform=ax2.transAxes, ha='center')
        ax2.set_title('Predicted Ratings Distribution')

    plt.tight_layout()
    st.pyplot(fig)

def plot_user_item_heatmap(matrix: pd.DataFrame, max_users: int = 20, max_items: int = 20):
    """
    Plot heatmap of user-item rating matrix subset.

    Args:
        matrix: User-item rating matrix
        max_users: Max users to display
        max_items: Max items to display
    """
    subset = matrix.iloc[:max_users, :max_items]

    fig = px.imshow(subset,
                   text_auto=True,
                   aspect="auto",
                   title=f'User-Item Rating Matrix (subset: {max_users} users x {max_items} items)',
                   color_continuous_scale='Blues')

    st.plotly_chart(fig, use_container_width=True)

def plot_coverage_analysis(ubcf_coverage: float, ibcf_coverage: float, total_items: int):
    """
    Plot coverage analysis.

    Args:
        ubcf_coverage: UBCF coverage percentage
        ibcf_coverage: IBCF coverage percentage
        total_items: Total number of items
    """
    fig = go.Figure(data=[
        go.Bar(name='Coverage %', x=['UBCF', 'IBCF'], y=[ubcf_coverage, ibcf_coverage],
               marker_color=['blue', 'red'], text=[f'{ubcf_coverage:.1f}%', f'{ibcf_coverage:.1f}%'],
               textposition='auto')
    ])

    fig.update_layout(
        title=f'Item Coverage (Total Items: {total_items})',
        xaxis_title='Algorithm',
        yaxis_title='Coverage (%)'
    )

    st.plotly_chart(fig, use_container_width=True)
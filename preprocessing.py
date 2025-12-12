"""
Preprocessing module for CF Recommender Lab.

Handles data loading, validation, matrix building, and preprocessing steps.
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional, Tuple
from utils import validate_rating_columns, clean_dataset, calculate_sparsity, generate_dummy_data

def load_and_validate_data(uploaded_file: Optional[st.runtime.uploaded_file_manager.UploadedFile],
                          use_dummy: bool = False) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Load and validate dataset from upload or generate dummy data.

    Args:
        uploaded_file: Uploaded CSV file
        use_dummy: Whether to use dummy data

    Returns:
        Tuple of (DataFrame or None, error_message)
    """
    if use_dummy:
        df = generate_dummy_data()
        st.info("Using dummy dataset (100 users, 200 items, ~1000 ratings)")
    elif uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            return None, f"Error reading CSV file: {str(e)}"
    else:
        return None, "Please upload a CSV file or select dummy data"

    # Validate columns
    is_valid, error_msg = validate_rating_columns(df)
    if not is_valid:
        return None, error_msg

    # Clean dataset
    original_rows = len(df)
    df = clean_dataset(df)
    cleaned_rows = len(df)

    if cleaned_rows < original_rows:
        st.warning(f"Removed {original_rows - cleaned_rows} invalid rows")

    if cleaned_rows == 0:
        return None, "No valid data after cleaning"

    return df, ""

def build_rating_matrix(df: pd.DataFrame, normalize: bool = False,
                       min_user_ratings: int = 1, min_item_ratings: int = 1) -> pd.DataFrame:
    """
    Build user-item rating matrix with optional filtering and normalization.

    Args:
        df: Input DataFrame with userId, itemId, rating
        normalize: Whether to mean-center the ratings
        min_user_ratings: Minimum ratings per user
        min_item_ratings: Minimum ratings per item

    Returns:
        User-item rating matrix
    """
    # Filter users and items
    user_counts = df.groupby('userId').size()
    item_counts = df.groupby('itemId').size()

    valid_users = user_counts[user_counts >= min_user_ratings].index
    valid_items = item_counts[item_counts >= min_item_ratings].index

    filtered_df = df[df['userId'].isin(valid_users) & df['itemId'].isin(valid_items)]

    if len(filtered_df) == 0:
        st.error("No data left after filtering. Try lowering the minimum ratings thresholds.")
        return pd.DataFrame()

    # Remove duplicates before pivoting (additional safety measure)
    filtered_df = filtered_df.drop_duplicates(subset=['userId', 'itemId'], keep='first')

    # Build matrix
    matrix = filtered_df.pivot(index='userId', columns='itemId', values='rating')

    if normalize:
        # Mean-centering
        user_means = matrix.mean(axis=1, skipna=True)
        matrix = matrix.sub(user_means, axis=0)

    return matrix

def get_dataset_stats(df: pd.DataFrame) -> dict:
    """
    Calculate basic dataset statistics.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'unique_users': df['userId'].nunique(),
        'unique_items': df['itemId'].nunique(),
        'avg_rating': df['rating'].mean(),
        'min_rating': df['rating'].min(),
        'max_rating': df['rating'].max(),
        'rating_std': df['rating'].std(),
        'sparsity': None  # Will be calculated after matrix build
    }
    return stats

def display_dataset_overview(df: pd.DataFrame, matrix: pd.DataFrame):
    """
    Display dataset overview in Streamlit.

    Args:
        df: Original DataFrame
        matrix: Rating matrix
    """
    st.subheader("Dataset Overview")

    # Basic stats
    stats = get_dataset_stats(df)
    stats['sparsity'] = calculate_sparsity(matrix)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Ratings", stats['total_rows'])
        st.metric("Unique Users", stats['unique_users'])
    with col2:
        st.metric("Unique Items", stats['unique_items'])
        st.metric("Average Rating", ".2f")
    with col3:
        st.metric("Matrix Sparsity", ".1f")
        st.metric("Rating Range", f"{stats['min_rating']:.1f} - {stats['max_rating']:.1f}")

    # Preview
    st.subheader("Data Preview (First 10 rows)")
    st.dataframe(df.head(10))

    # Rating distribution
    st.subheader("Rating Distribution")
    fig = pd.cut(df['rating'], bins=np.arange(0, 5.5, 0.5)).value_counts().sort_index().plot.bar()
    st.pyplot(fig.figure)
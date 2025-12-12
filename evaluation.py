"""
Evaluation module for CF Recommender Lab.

Implements evaluation metrics, train/test splitting, and cross-validation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from typing import Dict, List, Tuple, Callable
import warnings
from utils import safe_divide
from user_based import compute_user_similarity, predict_user_rating, find_user_neighbors
from item_based import compute_item_similarity, predict_item_rating

def train_test_split_ratings(df: pd.DataFrame, test_size: float = 0.2,
                           random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split ratings into train and test sets.

    Args:
        df: Ratings DataFrame
        test_size: Proportion for test set
        random_state: Random seed

    Returns:
        Tuple of (train_df, test_df)
    """
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state,
                                        stratify=df['userId'] if len(df['userId'].unique()) > 1 else None)
    return train_df, test_df

def cross_validate_ratings(df: pd.DataFrame, k_folds: int = 5,
                          random_state: int = 42) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Perform k-fold cross-validation split.

    Args:
        df: Ratings DataFrame
        k_folds: Number of folds
        random_state: Random seed

    Returns:
        List of (train_df, test_df) tuples for each fold
    """
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    folds = []

    for train_idx, test_idx in kf.split(df):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        folds.append((train_df, test_df))

    return folds

def calculate_rmse_mae(predictions: np.ndarray, actuals: np.ndarray) -> Tuple[float, float]:
    """
    Calculate RMSE and MAE.

    Args:
        predictions: Predicted ratings
        actuals: Actual ratings

    Returns:
        Tuple of (rmse, mae)
    """
    if len(predictions) == 0:
        return np.nan, np.nan

    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))

    return rmse, mae

def calculate_precision_recall_at_k(recommendations: List[List[int]], relevant_items: List[set],
                                   k: int) -> Tuple[float, float]:
    """
    Calculate Precision@K and Recall@K.

    Args:
        recommendations: List of recommended item lists for each user
        relevant_items: List of relevant item sets for each user
        k: K value

    Returns:
        Tuple of (precision@k, recall@k)
    """
    if len(recommendations) == 0:
        return 0.0, 0.0

    precisions = []
    recalls = []

    for recs, relevant in zip(recommendations, relevant_items):
        recs_at_k = set(recs[:k])
        relevant_in_recs = recs_at_k.intersection(relevant)

        precision = len(relevant_in_recs) / k if k > 0 else 0
        recall = len(relevant_in_recs) / len(relevant) if len(relevant) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    return np.mean(precisions), np.mean(recalls)

def calculate_f1_at_k(precision: float, recall: float) -> float:
    """
    Calculate F1@K.

    Args:
        precision: Precision@K
        recall: Recall@K

    Returns:
        F1@K score
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def calculate_ndcg_at_k(recommendations: List[List[int]], relevant_items: List[set],
                       k: int) -> float:
    """
    Calculate NDCG@K.

    Args:
        recommendations: List of recommended item lists for each user
        relevant_items: List of relevant item sets for each user
        k: K value

    Returns:
        NDCG@K score
    """
    if len(recommendations) == 0:
        return 0.0

    ndcgs = []

    for recs, relevant in zip(recommendations, relevant_items):
        recs_at_k = recs[:k]

        dcg = 0.0
        idcg = 0.0

        for i, item in enumerate(recs_at_k):
            if item in relevant:
                dcg += 1.0 / np.log2(i + 2)

        # IDCG: ideal DCG (all relevant items at top)
        relevant_count = min(len(relevant), k)
        for i in range(relevant_count):
            idcg += 1.0 / np.log2(i + 2)

        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)

    return np.mean(ndcgs)

def calculate_coverage(recommendations: List[List[int]], all_items: set) -> float:
    """
    Calculate coverage (percentage of items recommended).

    Args:
        recommendations: List of recommended item lists
        all_items: Set of all possible items

    Returns:
        Coverage percentage
    """
    if len(all_items) == 0:
        return 0.0

    recommended_items = set()
    for recs in recommendations:
        recommended_items.update(recs)

    return len(recommended_items) / len(all_items) * 100

def evaluate_ubcf(train_df: pd.DataFrame, test_df: pd.DataFrame, metric: str = 'cosine',
                 k_neighbors: int = 10, min_user_ratings: int = 1, min_item_ratings: int = 1) -> Dict:
    """
    Evaluate UBCF on test set.

    Args:
        train_df: Training ratings
        test_df: Test ratings
        metric: Similarity metric
        k_neighbors: Number of neighbors
        min_user_ratings: Min ratings per user
        min_item_ratings: Min ratings per item

    Returns:
        Dictionary with evaluation metrics
    """
    from preprocessing import build_rating_matrix

    # Build training matrix
    train_matrix = build_rating_matrix(train_df, normalize=False,
                                     min_user_ratings=min_user_ratings,
                                     min_item_ratings=min_item_ratings)

    if train_matrix.empty:
        return {'rmse': np.nan, 'mae': np.nan, 'precision@5': np.nan, 'recall@5': np.nan,
                'f1@5': np.nan, 'ndcg@5': np.nan, 'coverage': np.nan}

    # Compute similarity
    similarity_df = compute_user_similarity(train_matrix, metric)

    # Get predictions for test set
    predictions = []
    actuals = []
    recommendations = []
    relevant_items = []

    test_users = test_df['userId'].unique()

    for user_id in test_users:
        if user_id not in train_matrix.index:
            continue

        neighbors = find_user_neighbors(similarity_df, user_id, k_neighbors)

        user_test_ratings = test_df[test_df['userId'] == user_id]
        user_relevant = set(user_test_ratings['itemId'].tolist())

        user_recs = []
        for _, row in user_test_ratings.iterrows():
            item_id = row['itemId']
            actual = row['rating']

            pred, _ = predict_user_rating(train_matrix, similarity_df, user_id, item_id, neighbors)
            if not np.isnan(pred):
                predictions.append(pred)
                actuals.append(actual)

            # For ranking metrics, consider items with rating >= 4.0 as relevant
            if actual >= 4.0:
                user_recs.append(item_id)

        recommendations.append(user_recs)
        relevant_items.append(user_relevant)

    # Calculate metrics
    rmse, mae = calculate_rmse_mae(np.array(predictions), np.array(actuals))
    precision, recall = calculate_precision_recall_at_k(recommendations, relevant_items, 5)
    f1 = calculate_f1_at_k(precision, recall)
    ndcg = calculate_ndcg_at_k(recommendations, relevant_items, 5)
    coverage = calculate_coverage(recommendations, set(train_matrix.columns))

    return {
        'rmse': rmse,
        'mae': mae,
        'precision@5': precision,
        'recall@5': recall,
        'f1@5': f1,
        'ndcg@5': ndcg,
        'coverage': coverage
    }

def evaluate_ibcf(train_df: pd.DataFrame, test_df: pd.DataFrame, metric: str = 'cosine',
                 min_user_ratings: int = 1, min_item_ratings: int = 1) -> Dict:
    """
    Evaluate IBCF on test set.

    Args:
        Similar to evaluate_ubcf

    Returns:
        Dictionary with evaluation metrics
    """
    from preprocessing import build_rating_matrix

    # Build training matrix
    train_matrix = build_rating_matrix(train_df, normalize=False,
                                     min_user_ratings=min_user_ratings,
                                     min_item_ratings=min_item_ratings)

    if train_matrix.empty:
        return {'rmse': np.nan, 'mae': np.nan, 'precision@5': np.nan, 'recall@5': np.nan,
                'f1@5': np.nan, 'ndcg@5': np.nan, 'coverage': np.nan}

    # Compute similarity
    similarity_df = compute_item_similarity(train_matrix, metric)

    # Similar evaluation logic as UBCF
    predictions = []
    actuals = []
    recommendations = []
    relevant_items = []

    test_users = test_df['userId'].unique()

    for user_id in test_users:
        if user_id not in train_matrix.index:
            continue

        user_test_ratings = test_df[test_df['userId'] == user_id]
        user_relevant = set(user_test_ratings['itemId'].tolist())

        user_recs = []
        for _, row in user_test_ratings.iterrows():
            item_id = row['itemId']
            actual = row['rating']

            pred, _ = predict_item_rating(train_matrix, similarity_df, user_id, item_id)
            if not np.isnan(pred):
                predictions.append(pred)
                actuals.append(actual)

            if actual >= 4.0:
                user_recs.append(item_id)

        recommendations.append(user_recs)
        relevant_items.append(user_relevant)

    # Calculate metrics
    rmse, mae = calculate_rmse_mae(np.array(predictions), np.array(actuals))
    precision, recall = calculate_precision_recall_at_k(recommendations, relevant_items, 5)
    f1 = calculate_f1_at_k(precision, recall)
    ndcg = calculate_ndcg_at_k(recommendations, relevant_items, 5)
    coverage = calculate_coverage(recommendations, set(train_matrix.columns))

    return {
        'rmse': rmse,
        'mae': mae,
        'precision@5': precision,
        'recall@5': recall,
        'f1@5': f1,
        'ndcg@5': ndcg,
        'coverage': coverage
    }
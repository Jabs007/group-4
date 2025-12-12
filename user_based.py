"""
User-Based Collaborative Filtering module for CF Recommender Lab.

Implements UBCF logic including similarity computation, neighbor finding, and recommendations.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from typing import List, Tuple, Optional
import warnings
from utils import safe_divide

def compute_user_similarity(matrix: pd.DataFrame, metric: str = 'cosine') -> pd.DataFrame:
    """
    Compute user-user similarity matrix.

    Args:
        matrix: User-item rating matrix
        metric: Similarity metric ('cosine', 'pearson', 'jaccard')

    Returns:
        User-user similarity matrix
    """
    if metric == 'cosine':
        # Fill NaN with 0 for cosine similarity
        filled_matrix = matrix.fillna(0)
        similarity = cosine_similarity(filled_matrix)
    elif metric == 'pearson':
        # Compute pairwise Pearson correlation
        similarity = np.zeros((len(matrix), len(matrix)))
        users = matrix.index.tolist()
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                user1_ratings = matrix.loc[users[i]].dropna()
                user2_ratings = matrix.loc[users[j]].dropna()
                common_items = user1_ratings.index.intersection(user2_ratings.index)

                if len(common_items) > 1:
                    corr, _ = pearsonr(user1_ratings[common_items], user2_ratings[common_items])
                    similarity[i, j] = corr if not np.isnan(corr) else 0
                    similarity[j, i] = similarity[i, j]
                else:
                    similarity[i, j] = 0
                    similarity[j, i] = 0
    elif metric == 'jaccard':
        # Treat as binary (rated or not)
        binary_matrix = (~matrix.isna()).astype(int)
        similarity = np.zeros((len(matrix), len(matrix)))
        users = matrix.index.tolist()
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                user1_items = set(binary_matrix.loc[users[i]][binary_matrix.loc[users[i]] == 1].index)
                user2_items = set(binary_matrix.loc[users[j]][binary_matrix.loc[users[j]] == 1].index)
                intersection = len(user1_items.intersection(user2_items))
                union = len(user1_items.union(user2_items))
                jaccard = safe_divide(intersection, union)
                similarity[i, j] = jaccard
                similarity[j, i] = jaccard
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")

    # Create DataFrame
    similarity_df = pd.DataFrame(similarity, index=matrix.index, columns=matrix.index)

    # Set diagonal to 0 (no self-similarity)
    np.fill_diagonal(similarity_df.values, 0)

    return similarity_df

def find_user_neighbors(similarity_df: pd.DataFrame, user_id: int, k: int = 10) -> List[int]:
    """
    Find top-K similar users for a given user.

    Args:
        similarity_df: User-user similarity matrix
        user_id: Target user ID
        k: Number of neighbors

    Returns:
        List of neighbor user IDs
    """
    if user_id not in similarity_df.index:
        return []

    user_similarities = similarity_df.loc[user_id].sort_values(ascending=False)
    neighbors = user_similarities.head(k).index.tolist()
    return neighbors

def predict_user_rating(matrix: pd.DataFrame, similarity_df: pd.DataFrame,
                       user_id: int, item_id: int, neighbors: List[int]) -> Tuple[float, float]:
    """
    Predict rating for a user-item pair using UBCF.

    Args:
        matrix: User-item rating matrix
        similarity_df: User-user similarity matrix
        user_id: Target user ID
        item_id: Target item ID
        neighbors: List of neighbor user IDs

    Returns:
        Tuple of (predicted_rating, confidence)
    """
    if user_id not in matrix.index or item_id not in matrix.columns:
        return np.nan, 0.0

    # Get ratings from neighbors for this item
    neighbor_ratings = []
    neighbor_similarities = []

    for neighbor in neighbors:
        if neighbor in matrix.index and not np.isnan(matrix.loc[neighbor, item_id]):
            neighbor_ratings.append(matrix.loc[neighbor, item_id])
            neighbor_similarities.append(similarity_df.loc[user_id, neighbor])

    if not neighbor_ratings:
        return np.nan, 0.0

    # Weighted average with shrinkage
    weights = np.array(neighbor_similarities)
    ratings = np.array(neighbor_ratings)

    # Add small constant to avoid zero weights
    weights = weights + 1e-6

    predicted_rating = np.average(ratings, weights=weights)

    # Confidence as average similarity
    confidence = np.mean(neighbor_similarities)

    return predicted_rating, confidence

def get_user_recommendations(matrix: pd.DataFrame, similarity_df: pd.DataFrame,
                           user_id: int, k_neighbors: int = 10, n_recommendations: int = 5) -> pd.DataFrame:
    """
    Generate recommendations for a user using UBCF.

    Args:
        matrix: User-item rating matrix
        similarity_df: User-user similarity matrix
        user_id: Target user ID
        k_neighbors: Number of neighbors to use
        n_recommendations: Number of recommendations

    Returns:
        DataFrame with recommended items, predicted ratings, and confidence
    """
    if user_id not in matrix.index:
        return pd.DataFrame(columns=['itemId', 'predicted_rating', 'confidence'])

    # Find neighbors
    neighbors = find_user_neighbors(similarity_df, user_id, k_neighbors)

    if not neighbors:
        return pd.DataFrame(columns=['itemId', 'predicted_rating', 'confidence'])

    # Get unrated items
    user_ratings = matrix.loc[user_id]
    unrated_items = user_ratings[user_ratings.isna()].index.tolist()

    if not unrated_items:
        return pd.DataFrame(columns=['itemId', 'predicted_rating', 'confidence'])

    # Predict ratings for unrated items
    predictions = []
    for item_id in unrated_items:
        pred_rating, conf = predict_user_rating(matrix, similarity_df, user_id, item_id, neighbors)
        if not np.isnan(pred_rating):
            predictions.append({
                'itemId': item_id,
                'predicted_rating': pred_rating,
                'confidence': conf
            })

    # Sort by predicted rating descending
    recommendations = pd.DataFrame(predictions).sort_values('predicted_rating', ascending=False).head(n_recommendations)

    return recommendations
"""
Item-Based Collaborative Filtering module for CF Recommender Lab.

Implements IBCF logic including similarity computation and recommendations.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from typing import List, Tuple
from utils import safe_divide

def compute_item_similarity(matrix: pd.DataFrame, metric: str = 'cosine') -> pd.DataFrame:
    """
    Compute item-item similarity matrix.

    Args:
        matrix: User-item rating matrix
        metric: Similarity metric ('cosine', 'pearson', 'jaccard')

    Returns:
        Item-item similarity matrix
    """
    # Transpose matrix for item-based
    item_matrix = matrix.T

    if metric == 'cosine':
        # Fill NaN with 0
        filled_matrix = item_matrix.fillna(0)
        similarity = cosine_similarity(filled_matrix)
    elif metric == 'pearson':
        similarity = np.zeros((len(item_matrix), len(item_matrix)))
        items = item_matrix.index.tolist()
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                item1_ratings = item_matrix.loc[items[i]].dropna()
                item2_ratings = item_matrix.loc[items[j]].dropna()
                common_users = item1_ratings.index.intersection(item2_ratings.index)

                if len(common_users) > 1:
                    corr, _ = pearsonr(item1_ratings[common_users], item2_ratings[common_users])
                    similarity[i, j] = corr if not np.isnan(corr) else 0
                    similarity[j, i] = similarity[i, j]
                else:
                    similarity[i, j] = 0
                    similarity[j, i] = 0
    elif metric == 'jaccard':
        # Binary: rated or not
        binary_matrix = (~item_matrix.isna()).astype(int)
        similarity = np.zeros((len(item_matrix), len(item_matrix)))
        items = item_matrix.index.tolist()
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                item1_users = set(binary_matrix.loc[items[i]][binary_matrix.loc[items[i]] == 1].index)
                item2_users = set(binary_matrix.loc[items[j]][binary_matrix.loc[items[j]] == 1].index)
                intersection = len(item1_users.intersection(item2_users))
                union = len(item1_users.union(item2_users))
                jaccard = safe_divide(intersection, union)
                similarity[i, j] = jaccard
                similarity[j, i] = jaccard
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")

    # Create DataFrame
    similarity_df = pd.DataFrame(similarity, index=item_matrix.index, columns=item_matrix.index)

    # Set diagonal to 0
    np.fill_diagonal(similarity_df.values, 0)

    return similarity_df

def predict_item_rating(matrix: pd.DataFrame, similarity_df: pd.DataFrame,
                       user_id: int, item_id: int) -> Tuple[float, float]:
    """
    Predict rating for a user-item pair using IBCF.

    Args:
        matrix: User-item rating matrix
        similarity_df: Item-item similarity matrix
        user_id: Target user ID
        item_id: Target item ID

    Returns:
        Tuple of (predicted_rating, confidence)
    """
    if user_id not in matrix.index or item_id not in matrix.columns:
        return np.nan, 0.0

    # Get user's rated items
    user_ratings = matrix.loc[user_id].dropna()

    if len(user_ratings) == 0:
        return np.nan, 0.0

    # Find similar items that user has rated
    if item_id not in similarity_df.index:
        return np.nan, 0.0

    item_similarities = similarity_df.loc[item_id]
    rated_similar_items = item_similarities[user_ratings.index].dropna()

    if len(rated_similar_items) == 0:
        return np.nan, 0.0

    # Weighted average
    similarities = rated_similar_items.values
    ratings = user_ratings[rated_similar_items.index].values

    # Add small constant
    similarities = similarities + 1e-6

    predicted_rating = np.average(ratings, weights=similarities)

    # Confidence as average similarity
    confidence = np.mean(similarities)

    return predicted_rating, confidence

def get_item_recommendations(matrix: pd.DataFrame, similarity_df: pd.DataFrame,
                           user_id: int, n_recommendations: int = 5) -> pd.DataFrame:
    """
    Generate recommendations for a user using IBCF.

    Args:
        matrix: User-item rating matrix
        similarity_df: Item-item similarity matrix
        user_id: Target user ID
        n_recommendations: Number of recommendations

    Returns:
        DataFrame with recommended items, predicted ratings, and confidence
    """
    if user_id not in matrix.index:
        return pd.DataFrame(columns=['itemId', 'predicted_rating', 'confidence'])

    # Get unrated items
    user_ratings = matrix.loc[user_id]
    unrated_items = user_ratings[user_ratings.isna()].index.tolist()

    if not unrated_items:
        return pd.DataFrame(columns=['itemId', 'predicted_rating', 'confidence'])

    # Predict ratings for unrated items
    predictions = []
    for item_id in unrated_items:
        pred_rating, conf = predict_item_rating(matrix, similarity_df, user_id, item_id)
        if not np.isnan(pred_rating):
            predictions.append({
                'itemId': item_id,
                'predicted_rating': pred_rating,
                'confidence': conf
            })

    # Sort by predicted rating descending
    recommendations = pd.DataFrame(predictions).sort_values('predicted_rating', ascending=False).head(n_recommendations)

    return recommendations
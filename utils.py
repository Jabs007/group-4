"""
Utils module for CF Recommender Lab.

Contains helper functions, constants, and dummy data generation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional

# Constants
SIMILARITY_METRICS = ['cosine', 'pearson', 'jaccard']
DEFAULT_MIN_RATINGS_USER = 1
DEFAULT_MIN_RATINGS_ITEM = 1
DEFAULT_K_NEIGHBORS = 10
DEFAULT_N_RECOMMENDATIONS = 5
DEFAULT_TRAIN_SPLIT = 0.8
DEFAULT_RANDOM_SEED = 42
MIN_RATING = 0.0
MAX_RATING = 5.0

def generate_dummy_data(n_users: int = 100, n_items: int = 200, n_ratings: int = 1000,
                       min_rating: float = MIN_RATING, max_rating: float = MAX_RATING,
                       seed: int = DEFAULT_RANDOM_SEED) -> pd.DataFrame:
    """
    Generate dummy MovieLens-like dataset.

    Args:
        n_users: Number of users
        n_items: Number of items
        n_ratings: Number of ratings
        min_rating: Minimum rating value
        max_rating: Maximum rating value
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: userId, itemId, rating
    """
    np.random.seed(seed)

    # Generate random ratings
    user_ids = np.random.randint(1, n_users + 1, n_ratings)
    item_ids = np.random.randint(1, n_items + 1, n_ratings)
    ratings = np.random.uniform(min_rating, max_rating, n_ratings)

    # Round to nearest 0.5 for realism
    ratings = np.round(ratings * 2) / 2

    df = pd.DataFrame({
        'userId': user_ids,
        'itemId': item_ids,
        'rating': ratings
    })

    # Remove duplicates (same user-item pairs)
    df = df.drop_duplicates(subset=['userId', 'itemId'])

    return df

def validate_rating_columns(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate that DataFrame has required columns with correct types.

    Args:
        df: Input DataFrame

    Returns:
        Tuple of (is_valid, error_message)
    """
    required_cols = ['userId', 'itemId', 'rating']
    for col in required_cols:
        if col not in df.columns:
            return False, f"Missing required column: {col}"

    # Check data types
    if not pd.api.types.is_integer_dtype(df['userId']):
        return False, "userId must be integer"
    if not pd.api.types.is_integer_dtype(df['itemId']):
        return False, "itemId must be integer"
    if not pd.api.types.is_numeric_dtype(df['rating']):
        return False, "rating must be numeric"

    # Check rating range
    if df['rating'].min() < MIN_RATING or df['rating'].max() > MAX_RATING:
        return False, f"Rating values must be between {MIN_RATING} and {MAX_RATING}"

    return True, ""

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset by removing invalid rows and duplicates.

    Args:
        df: Input DataFrame

    Returns:
        Cleaned DataFrame
    """
    # Drop rows with NaN in required columns
    df = df.dropna(subset=['userId', 'itemId', 'rating'])

    # Remove duplicate user-item pairs (keep first occurrence)
    df = df.drop_duplicates(subset=['userId', 'itemId'], keep='first')

    # Ensure integer types
    df['userId'] = df['userId'].astype(int)
    df['itemId'] = df['itemId'].astype(int)

    return df

def calculate_sparsity(matrix: pd.DataFrame) -> float:
    """
    Calculate sparsity of the rating matrix.

    Args:
        matrix: User-item rating matrix

    Returns:
        Sparsity percentage (0-100)
    """
    total_elements = matrix.shape[0] * matrix.shape[1]
    non_null_elements = matrix.notna().sum().sum()
    sparsity = (1 - non_null_elements / total_elements) * 100
    return sparsity

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """
    Safe division to avoid division by zero.

    Args:
        a: Numerator
        b: Denominator
        default: Default value if b == 0

    Returns:
        Division result or default
    """
    return a / b if b != 0 else default
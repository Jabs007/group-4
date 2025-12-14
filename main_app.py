"""
CF Recommender Lab - Main Streamlit Application

A comprehensive collaborative filtering recommender system with UBCF and IBCF.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict
import io

# Import modules
from preprocessing import load_and_validate_data, build_rating_matrix, display_dataset_overview
from user_based import compute_user_similarity, get_user_recommendations
from item_based import compute_item_similarity, get_item_recommendations
from evaluation import evaluate_ubcf, evaluate_ibcf, train_test_split_ratings
from visualization import (plot_similarity_heatmap, plot_metrics_comparison,
                           plot_precision_recall_vs_k, plot_predicted_vs_actual,
                           plot_rating_distributions, plot_user_item_heatmap,
                           plot_coverage_analysis)
from utils import SIMILARITY_METRICS, DEFAULT_K_NEIGHBORS, DEFAULT_N_RECOMMENDATIONS, DEFAULT_TRAIN_SPLIT

# Page configuration
st.set_page_config(
    page_title="CF Recommender Lab",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-item {
        background-color: #e8f4f8;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<div class="main-header">🎬 CF Recommender Lab</div>', unsafe_allow_html=True)
    st.markdown("### Collaborative Filtering Recommender System with UBCF & IBCF")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Dataset selection
        st.subheader("📊 Dataset")
        st.info("Upload your dataset (CSV format with columns: userId, itemId, rating)")
        uploaded_file = st.file_uploader("Upload CSV Dataset", type=['csv'])

        # Preprocessing parameters
        st.subheader("🔧 Preprocessing")
        normalize = st.checkbox("Normalize Ratings (Mean-centering)", value=False)
        min_user_ratings = st.slider("Min Ratings per User", 1, 50, 1, key="min_user")
        min_item_ratings = st.slider("Min Ratings per Item", 1, 50, 1, key="min_item")

        # Algorithm parameters
        st.subheader("🤖 Algorithm Settings")
        algorithm = st.selectbox("Algorithm", ["UBCF", "IBCF"])
        similarity_metric = st.selectbox("Similarity Metric", SIMILARITY_METRICS, index=0)
        k_neighbors = st.slider("K Neighbors (UBCF)", 2, 50, DEFAULT_K_NEIGHBORS) if algorithm == "UBCF" else None
        n_recommendations = st.slider("N Recommendations", 1, 20, DEFAULT_N_RECOMMENDATIONS)

        # Evaluation parameters
        st.subheader("📈 Evaluation")
        train_split = st.slider("Train/Test Split", 0.6, 0.9, DEFAULT_TRAIN_SPLIT, 0.05)
        random_seed = st.number_input("Random Seed", value=42, min_value=0)

        # User selection for recommendations
        st.subheader("👤 User Selection")
        user_id = st.number_input("Select User ID", value=1, min_value=1)

        # Reset button
        if st.button("🔄 Reset App"):
            st.rerun()

    # Load and validate data
    @st.cache_data
    def load_data_cached(uploaded_file):
        return load_and_validate_data(uploaded_file, use_dummy=False)

    df, error_msg = load_data_cached(uploaded_file)

    if error_msg:
        st.error(f"Data loading error: {error_msg}")
        return

    # Build rating matrix
    @st.cache_data
    def build_matrix_cached(df, normalize, min_user_ratings, min_item_ratings):
        return build_rating_matrix(df, normalize, min_user_ratings, min_item_ratings)

    matrix = build_matrix_cached(df, normalize, min_user_ratings, min_item_ratings)

    if matrix.empty:
        st.error("Rating matrix is empty after preprocessing. Adjust filtering parameters.")
        return

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dataset Overview",
        "👥 UBCF",
        "🎬 IBCF",
        "📈 Evaluation",
        "📉 Visualizations",
        "🏆 Final Comparison"
    ])

    # Tab 1: Dataset Overview
    with tab1:
        with st.expander("ℹ️ What is the Dataset Overview?", expanded=False):
            st.markdown("""
            This tab provides a comprehensive overview of your dataset, including key statistics, data quality metrics,
            and preprocessing details. Understanding your data is crucial for interpreting recommendation results.
            """)

        st.markdown("### Understanding Your Dataset")
        st.markdown("""
        Before diving into recommendations, it's important to understand your data's characteristics.
        The statistics below show the scale and sparsity of your dataset, which affect recommendation quality.
        """)

        display_dataset_overview(df, matrix)

        # Preprocessing explanations
        st.markdown("### Preprocessing Details")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Filtering Applied:**")
            st.markdown(f"- Minimum {min_user_ratings} ratings per user")
            st.markdown(f"- Minimum {min_item_ratings} ratings per item")
            st.markdown("- Removed users/items with insufficient data")
        with col2:
            st.markdown("**Normalization:**")
            if normalize:
                st.markdown("- Mean-centering applied to user ratings")
                st.markdown("- Helps focus on preference deviations")
            else:
                st.markdown("- Raw ratings used")
                st.markdown("- Preserves absolute rating scales")

        # Matrix preview
        st.subheader("User-Item Rating Matrix Preview")
        st.markdown("This heatmap shows a small sample of the rating matrix. White cells indicate missing ratings (sparsity).")
        plot_user_item_heatmap(matrix, max_users=10, max_items=10)

    # Tab 2: UBCF
    with tab2:
        with st.expander("ℹ️ What is User-Based Collaborative Filtering?", expanded=False):
            st.markdown("""
            UBCF recommends items by finding users similar to the target user and suggesting items they liked.
            It works by: 1) Computing user-user similarities, 2) Finding nearest neighbors, 3) Predicting ratings
            based on neighbors' preferences. Best for datasets with many users and clear user clusters.
            """)

        st.header("User-Based Collaborative Filtering")

        if user_id not in matrix.index:
            st.warning(f"User {user_id} not found in the dataset.")
        else:
            # Compute similarity
            with st.spinner("Computing user-user similarity..."):
                @st.cache_data
                def compute_ubcf_similarity_cached(matrix, similarity_metric):
                    return compute_user_similarity(matrix, similarity_metric)

                ubcf_similarity = compute_ubcf_similarity_cached(matrix, similarity_metric)

            # Similarity heatmap
            st.subheader("User-User Similarity Matrix (Top 20 Users)")
            plot_similarity_heatmap(ubcf_similarity, "User-User Similarity Heatmap", max_users=20)

            # Recommendations
            st.subheader(f"Recommendations for User {user_id}")
            with st.spinner("Generating recommendations..."):
                recommendations = get_user_recommendations(
                    matrix, ubcf_similarity, user_id, k_neighbors, n_recommendations
                )

            if recommendations.empty:
                st.info("No recommendations available for this user.")
            else:
                for _, row in recommendations.iterrows():
                    st.markdown(f"""
                    <div class="recommendation-item">
                        <strong>Item {int(row['itemId'])}</strong><br>
                        Predicted Rating: {row['predicted_rating']:.2f}<br>
                        Confidence: {row['confidence']:.3f}
                    </div>
                    """, unsafe_allow_html=True)

                # Export recommendations
                csv = recommendations.to_csv(index=False)
                st.download_button(
                    label="📥 Download Recommendations as CSV",
                    data=csv,
                    file_name=f'ubcf_recommendations_user_{user_id}.csv',
                    mime='text/csv'
                )

            # User history
            st.subheader(f"User {user_id} Rating History")
            user_ratings = matrix.loc[user_id].dropna().reset_index()
            user_ratings.columns = ['Item ID', 'Rating']
            st.dataframe(user_ratings)

    # Tab 3: IBCF
    with tab3:
        with st.expander("ℹ️ What is Item-Based Collaborative Filtering?", expanded=False):
            st.markdown("""
            IBCF recommends items similar to those the user has already rated highly. It works by:
            1) Computing item-item similarities, 2) Finding items similar to user's liked items,
            3) Recommending the most similar unrated items. Effective when items have stable characteristics.
            """)

        st.header("Item-Based Collaborative Filtering")

        if user_id not in matrix.index:
            st.warning(f"User {user_id} not found in the dataset.")
        else:
            # Compute similarity
            with st.spinner("Computing item-item similarity..."):
                @st.cache_data
                def compute_ibcf_similarity_cached(matrix, similarity_metric):
                    return compute_item_similarity(matrix, similarity_metric)

                ibcf_similarity = compute_ibcf_similarity_cached(matrix, similarity_metric)

            # Similarity heatmap
            st.subheader("Item-Item Similarity Matrix (Top 20 Items)")
            plot_similarity_heatmap(ibcf_similarity, "Item-Item Similarity Heatmap", max_users=20)

            # Recommendations
            st.subheader(f"Recommendations for User {user_id}")
            with st.spinner("Generating recommendations..."):
                recommendations = get_item_recommendations(
                    matrix, ibcf_similarity, user_id, n_recommendations
                )

            if recommendations.empty:
                st.info("No recommendations available for this user.")
            else:
                for _, row in recommendations.iterrows():
                    st.markdown(f"""
                    <div class="recommendation-item">
                        <strong>Item {int(row['itemId'])}</strong><br>
                        Predicted Rating: {row['predicted_rating']:.2f}<br>
                        Confidence: {row['confidence']:.3f}
                    </div>
                    """, unsafe_allow_html=True)

                # Export recommendations
                csv = recommendations.to_csv(index=False)
                st.download_button(
                    label="📥 Download Recommendations as CSV",
                    data=csv,
                    file_name=f'ibcf_recommendations_user_{user_id}.csv',
                    mime='text/csv'
                )

            # User history (same as UBCF)
            st.subheader(f"User {user_id} Rating History")
            user_ratings = matrix.loc[user_id].dropna().reset_index()
            user_ratings.columns = ['Item ID', 'Rating']
            st.dataframe(user_ratings)

    # Tab 4: Evaluation
    with tab4:
        with st.expander("ℹ️ What is Model Evaluation?", expanded=False):
            st.markdown("""
            This tab evaluates UBCF and IBCF performance using standard recommender system metrics.
            Models are trained on a subset of data and tested on held-out ratings to measure accuracy and effectiveness.
            """)

        st.header("Model Evaluation")

        with st.spinner("Evaluating models..."):
            # Split data
            train_df, test_df = train_test_split_ratings(df, test_size=1-train_split, random_state=random_seed)

            # Evaluate UBCF
            ubcf_metrics = evaluate_ubcf(train_df, test_df, similarity_metric, k_neighbors,
                                       min_user_ratings, min_item_ratings)

            # Evaluate IBCF
            ibcf_metrics = evaluate_ibcf(train_df, test_df, similarity_metric,
                                       min_user_ratings, min_item_ratings)

        # Display metrics table
        st.subheader("Performance Metrics Comparison")
        metrics_df = pd.DataFrame({
            'UBCF': ubcf_metrics,
            'IBCF': ibcf_metrics
        }).T

        # Color coding
        def color_metrics(val):
            if pd.isna(val):
                return ''
            if val < 0.5:  # Lower is better for RMSE/MAE
                return 'color: green'
            elif val > 0.7:  # Higher is better for others
                return 'color: green'
            else:
                return 'color: orange'

        st.dataframe(metrics_df.style.applymap(color_metrics))

        # Metric explanations
        st.subheader("Understanding the Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**RMSE (Root Mean Square Error):**")
            st.markdown("Measures prediction accuracy. Lower is better. Represents average prediction error in rating units.")
            st.markdown("**MAE (Mean Absolute Error):**")
            st.markdown("Average absolute prediction error. Less sensitive to outliers than RMSE.")
        with col2:
            st.markdown("**Precision@K:**")
            st.markdown("Fraction of recommended items that are relevant. Higher is better.")
            st.markdown("**Recall@K:**")
            st.markdown("Fraction of relevant items that are recommended. Higher is better.")
            st.markdown("**F1@K:**")
            st.markdown("Harmonic mean of precision and recall. Balances both metrics.")
        st.markdown("**NDCG@K:**")
        st.markdown("Normalized Discounted Cumulative Gain. Rewards relevant items at top positions. Higher is better.")
        st.markdown("**Coverage:**")
        st.markdown("Percentage of items that can be recommended. Higher coverage means more diverse suggestions.")

    # Tab 5: Visualizations
    with tab5:
        with st.expander("ℹ️ What are the Visualizations?", expanded=False):
            st.markdown("""
            This tab provides interactive charts to help understand model performance, data characteristics,
            and algorithm comparisons. Visualizations make it easier to interpret complex metrics and identify patterns.
            """)

        st.header("Advanced Visualizations")

        # Metrics comparison
        st.subheader("Metrics Comparison")
        st.markdown("Bar chart comparing all evaluation metrics between UBCF and IBCF.")
        plot_metrics_comparison(ubcf_metrics, ibcf_metrics)

        # Coverage analysis
        st.subheader("Coverage Analysis")
        st.markdown("Shows how many items each algorithm can recommend. Higher coverage means more diverse suggestions.")
        plot_coverage_analysis(
            ubcf_metrics.get('coverage', 0),
            ibcf_metrics.get('coverage', 0),
            len(matrix.columns)
        )

        # Rating distributions
        st.subheader("Rating Distributions")
        st.markdown("Histogram of rating values in your dataset. Understanding rating patterns helps interpret recommendations.")
        plot_rating_distributions(df)

    # Tab 6: Final Comparison
    with tab6:
        with st.expander("ℹ️ What is the Final Comparison?", expanded=False):
            st.markdown("""
            This tab provides a comprehensive summary of UBCF vs IBCF performance across all metrics,
            with visualizations and export options. Use this to decide which algorithm works best for your data.
            """)

        st.header("🏆 Final Comparison & Summary")

        # Overall metrics table
        st.subheader("Complete Metrics Comparison")
        metrics_df = pd.DataFrame({
            'UBCF': ubcf_metrics,
            'IBCF': ibcf_metrics
        }).T
        st.dataframe(metrics_df.style.highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='lightcoral'))

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("UBCF Summary")
            st.metric("RMSE", ".3f")
            st.metric("MAE", ".3f")
            st.metric("Precision@5", ".3f")
            st.metric("Coverage", ".1f")

        with col2:
            st.subheader("IBCF Summary")
            st.metric("RMSE", ".3f")
            st.metric("MAE", ".3f")
            st.metric("Precision@5", ".3f")
            st.metric("Coverage", ".1f")

        # Winner determination with more criteria
        rmse_ubcf = ubcf_metrics.get('rmse', float('inf'))
        rmse_ibcf = ibcf_metrics.get('rmse', float('inf'))
        prec_ubcf = ubcf_metrics.get('precision@5', 0)
        prec_ibcf = ibcf_metrics.get('precision@5', 0)

        # Determine winner based on multiple criteria
        if rmse_ubcf < rmse_ibcf and prec_ubcf > prec_ibcf:
            winner = "UBCF"
            reason = "better accuracy and precision"
        elif rmse_ibcf < rmse_ubcf and prec_ibcf > prec_ubcf:
            winner = "IBCF"
            reason = "better accuracy and precision"
        elif rmse_ubcf < rmse_ibcf:
            winner = "UBCF"
            reason = "lower RMSE"
        elif rmse_ibcf < rmse_ubcf:
            winner = "IBCF"
            reason = "lower RMSE"
        else:
            winner = "Tie"
            reason = "similar performance"

        st.success(f"🏆 Winner: {winner} ({reason})")

        # Recommendations
        st.subheader("Recommendations")
        if winner == "UBCF":
            st.info("UBCF performs better. Consider using it for datasets with clear user communities or when user behavior is more predictive than item similarities.")
        elif winner == "IBCF":
            st.info("IBCF performs better. Consider using it for stable item characteristics or when items have consistent appeal patterns.")
        else:
            st.info("Both algorithms perform similarly. Choose based on computational efficiency (IBCF is often faster) or business requirements.")

        # Export comparison
        comparison_data = {
            'Algorithm': ['UBCF', 'IBCF'],
            'RMSE': [rmse_ubcf, rmse_ibcf],
            'MAE': [ubcf_metrics.get('mae', 0), ibcf_metrics.get('mae', 0)],
            'Precision@5': [ubcf_metrics.get('precision@5', 0), ibcf_metrics.get('precision@5', 0)],
            'Coverage': [ubcf_metrics.get('coverage', 0), ibcf_metrics.get('coverage', 0)]
        }
        comparison_df = pd.DataFrame(comparison_data)
        csv = comparison_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Comparison Results",
            data=csv,
            file_name='cf_comparison_results.csv',
            mime='text/csv'
        )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>CF Recommender Lab</strong> - Built with Streamlit</p>
        <p>Supporting UBCF & IBCF with comprehensive evaluation and visualization</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
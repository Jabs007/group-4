# CF Recommender Lab

A comprehensive Streamlit web application implementing User-Based and Item-Based Collaborative Filtering for recommender systems.

## Features

- **Dataset Upload**: Upload your own CSV dataset or use the built-in dummy dataset
- **Data Preprocessing**: Validate, clean, and build user-item rating matrices with optional normalization and filtering
- **User-Based CF (UBCF)**: Compute user-user similarities, find neighbors, and generate recommendations
- **Item-Based CF (IBCF)**: Compute item-item similarities and generate recommendations
- **Evaluation**: Comprehensive metrics including RMSE, MAE, Precision@K, Recall@K, F1@K, NDCG@K, and Coverage
- **Visualizations**: Interactive plots for similarity matrices, metrics comparison, and rating distributions
- **Comparison**: Side-by-side performance comparison between UBCF and IBCF

## Installation

1. Clone or download this repository
2. Navigate to the `cf_recommender_lab` directory
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the App

```bash
streamlit run main_app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Dataset Format

Your CSV file must contain exactly these columns:
- `userId`: Integer user identifiers
- `itemId`: Integer item identifiers
- `rating`: Float ratings (typically 0.0 to 5.0)

Example:
```csv
userId,itemId,rating
1,1,4.5
1,2,3.0
2,1,5.0
...
```

## Usage

1. **Dataset Selection**: Choose between dummy data or upload your own CSV
2. **Preprocessing**: Adjust filtering parameters and normalization options
3. **Algorithm Configuration**: Select UBCF or IBCF, similarity metric, and other parameters
4. **Explore Tabs**:
   - Dataset Overview: View statistics and matrix preview
   - UBCF/IBCF: See similarity matrices and get recommendations for selected users
   - Evaluation: Compare model performance metrics
   - Visualizations: Interactive charts and plots
   - Final Comparison: Summary and export options

## Modules

- `utils.py`: Helper functions, constants, and dummy data generation
- `preprocessing.py`: Data loading, validation, and matrix building
- `user_based.py`: UBCF implementation
- `item_based.py`: IBCF implementation
- `evaluation.py`: Metrics calculation and model evaluation
- `visualization.py`: Plotting functions
- `main_app.py`: Streamlit UI and app logic

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

## Example Dataset

An example dataset (`example_dataset.csv`) is provided with 50 users, 100 items, and 500 ratings.

## Notes

- The app handles cold start users/items gracefully
- All computations are cached for performance
- Error handling and validation are implemented throughout
- Export options available for recommendations and comparison results
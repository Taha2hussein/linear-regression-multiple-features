"""
model_comparison.py
===================
A reusable module for comparing multiple supervised learning regression models.
Import and use in your main.py file.

Usage:
------
    from model_comparison import compare_models

    results = compare_models(X_train, X_test, y_train, y_test)
    print(results)
"""

from sklearn.linear_model import LinearRegression, SGDRegressor, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score
)
import pandas as pd


def get_models(random_state=33):
    """
    Returns a dictionary of regression models.

    Parameters:
    -----------
    random_state : int
        Random seed for reproducibility.

    Returns:
    --------
    dict : Dictionary with model names as keys and model instances as values.
    """
    models = {
        'LinearRegression': LinearRegression(),
        'SGDRegressor': SGDRegressor(alpha=0.1, random_state=random_state, penalty='l2', loss='huber'),
        'Lasso': Lasso(alpha=1.0, random_state=random_state),
        'Ridge': Ridge(alpha=1.0, random_state=random_state),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=8, random_state=random_state, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=500, max_depth=7, learning_rate=0.1, random_state=random_state),
        'SVR': SVR(C=1.0, epsilon=0.1, kernel='rbf'),
        'DecisionTree': DecisionTreeRegressor(max_depth=3, random_state=random_state),
        'KNeighbors': KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto'),
        'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=random_state),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=100, max_depth=8, random_state=random_state, n_jobs=-1),
    }
    return models


def compare_models(X_train, X_test, y_train, y_test, models=None, verbose=True):
    """
    Trains multiple regression models and compares their performance.

    Parameters:
    -----------
    X_train, X_test : array-like
        Training and testing feature matrices.
    y_train, y_test : array-like
        Training and testing target vectors.
    models : dict, optional
        Dictionary of models to compare. If None, uses default models from get_models().
    verbose : bool
        If True, prints results for each model.

    Returns:
    --------
    pd.DataFrame : DataFrame containing all models' metrics, sorted by R2 (descending).
    dict : Dictionary containing trained model instances.
    """
    if models is None:
        models = get_models()

    results = []
    trained_models = {}

    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        mdse = median_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append({
            'Model': name,
            'Train_R2': round(train_score, 4),
            'Test_R2': round(test_score, 4),
            'R2': round(r2, 4),
            'MAE': round(mae, 4),
            'MSE': round(mse, 4),
            'RMSE': round(rmse, 4),
            'MdSE': round(mdse, 4),
        })

        if verbose:
            print(f'\n{"="*55}')
            print(f'  Model: {name}')
            print(f'{"="*55}')
            print(f'Train Score (R²) : {train_score:.4f}')
            print(f'Test Score  (R²) : {test_score:.4f}')
            print(f'R² Score         : {r2:.4f}')
            print(f'MAE  value       : {mae:.4f}')
            print(f'MSE  value       : {mse:.4f}')
            print(f'RMSE value       : {rmse:.4f}')
            print(f'MdSE value       : {mdse:.4f}')

    results_df = pd.DataFrame(results).sort_values('R2', ascending=False).reset_index(drop=True)

    if verbose:
        print(f'\n{"="*55}')
        print('  SUMMARY (Sorted by R² Score)')
        print(f'{"="*55}')
        print(results_df.to_string(index=False))
        print(f'\n🏆 Best Model: {results_df.iloc[0]["Model"]} (R² = {results_df.iloc[0]["R2"]})')

    return results_df, trained_models


def get_best_model(results_df, trained_models):
    """
    Returns the best performing model based on R2 score.

    Parameters:
    -----------
    results_df : pd.DataFrame
        Results DataFrame from compare_models().
    trained_models : dict
        Dictionary of trained models from compare_models().

    Returns:
    --------
    str : Name of the best model.
    object : The trained model instance.
    """
    best_name = results_df.iloc[0]['Model']
    return best_name, trained_models[best_name]


# ─────────────────────────────────────────────────────────────
# Example usage (run this file directly to test)
# ─────────────────────────────────────────────────────────────
# if __name__ == '__main__':
#     from sklearn.model_selection import train_test_split

#     # Load sample data
#     data = fetch_california_housing()
#     X, y = data.data, data.target
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Run comparison
#     results, models = compare_models(X_train, X_test, y_train, y_test)

#     # Get best model
#     best_name, best_model = get_best_model(results, models)
#     print(f"\n✅ Best Model: {best_name}")
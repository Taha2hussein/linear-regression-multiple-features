
# Model Evaluation Summary

After training multiple regression models, Gradient Boosting Regressor achieved the best overall performance.

## Best Model

GradientBoostingRegressor

## Performance

- Train R²: 0.9915
- Test R²: 0.8775
- MAE: 305,513
- RMSE: 925,043

## Why Gradient Boosting?

Gradient Boosting performed better because the dataset contains non-linear relationships between the features and the target variable (Price).

Unlike Linear Regression, Gradient Boosting builds decision trees sequentially, where each new tree learns from the errors of the previous one. This allows the model to capture complex patterns and improve prediction accuracy.

## Comparison

- Linear Regression worked as a good baseline model but could not capture non-linear patterns.
- Random Forest improved the performance significantly.
- Gradient Boosting achieved the highest R² score and the lowest prediction error.

## Current Limitation

Categorical features were removed before training because encoding has not been applied yet.

The following features were dropped:

- Make
- Model
- Fuel Type
- Transmission
- Location
- Color
- Owner
- Seller Type
- Drivetrain

These features may improve model performance after applying proper encoding techniques.

## Future Work

- Learn and apply Encoding.
- Perform Feature Engineering.
- Apply Feature Scaling where required.
- Use Cross Validation.
- Tune Hyperparameters.
- Compare with XGBoost, LightGBM, and CatBoost.


### Conclusion

At this stage, Gradient Boosting is the best model for this dataset because it captures non-linear relationships better than the other tested algorithms. However, the pipeline is still incomplete, and further improvements are expected after applying categorical encoding and feature engineering.

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression


def train_model(X_train, y_train):
    """
    Train a linear regression model using the provided training data.

    Parameters:
    X_train (DataFrame): The training features.
    y_train (Series): The target variable for training.

    Returns:
    model: The trained linear regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    """
    Use the trained model to make predictions on the test data.

    Parameters:
    model: The trained linear regression model.
    X_test (DataFrame): The test features.

    Returns:
    predictions: The predicted values for the test set.
    """
    return model.predict(X_test)
    
def evaluate_model(y_test, y_pred):
    """
    Evaluate the performance of the model using Mean Squared Error and R-squared metrics.

    Parameters:
    y_test (Series): The true target values for the test set.
    y_pred (array): The predicted values from the model.

    Returns:
    mse: Mean Squared Error of the predictions.
    r2: R-squared value of the predictions.
    """
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, r2, mae
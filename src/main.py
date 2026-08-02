from pathlib import Path
from numpy import log1p
from data.loader import load_data
from data.correlation import get_top_correlated_features
from visualization.scatter import draw_scatter
from visualization.histogram import draw_histogram
from visualization.boxplot import draw_box_plot
from data.transform import transform_column_types
from data.cleaning import remove_duplicates, fill_missing_values
from data.analyzer import dataset_summary, missing_values_report, duplicate_report, numeric_columns, categorical_columns
from models.split import split_data
from models.linear_train import train_model, predict, evaluate_model
from sklearn.metrics import r2_score
from models.random_forest_train import random_forest_train_model, random_forest_predict, random_forest_evaluate_model 
from models.model_comparison import compare_models, get_best_model


def main():
    """
    Load the dataset and print its summary, missing values report, duplicate report, and numeric columns.
    """
    
    DATA_PATH = Path("data/raw/car details v4.csv")
    df = load_data(DATA_PATH)
           
    print("Dataset Summary:")
    print(dataset_summary(df))
    
    print("\nMissing Values Report:")
    print(missing_values_report(df))
    
    print("\nDuplicate Report:")
    print(duplicate_report(df))
    
    print("\nNumeric Columns Before Transformation:")
    print(numeric_columns(df))
    
    print("\ncategorical columns Before Transformation:")
    print(categorical_columns(df))
    
    print("\ncolumn Data Types:")
    df = transform_column_types(df)
    print(df.dtypes)
    
    numeric_columns_list = numeric_columns(df)
    print("\nnumeric columns After Transformation:")
    print(numeric_columns_list)
        
    categorical_columns_list = categorical_columns(df)
    print("\ncategorical columns After Transformation:")
    print(categorical_columns_list)
    
    remove_duplicates(df)
    
    fill_missing_values(df)
        
    df = df.drop(columns=["Make","Model","Kilometer","Fuel Type","Transmission","Location","Color","Owner","Seller Type","Drivetrain"])  

    correlation_values = get_top_correlated_features(df, "Price", count=7)
    
    print("\nTopCorrelation")
    print(f"{correlation_values}")
    
    # IFwe need to log price for better visualization, we can do it here
    target= log1p(df["Price"])
    
    for correlation in correlation_values.index:
        draw_scatter(df[correlation], target)
        draw_histogram(df[correlation])
        draw_box_plot(df[correlation])
        
   
    X_train, X_test, y_train, y_test =  split_data(df, "Price", test_size=0.2, random_state=42)
   
    model = train_model(X_train, y_train)
    
    # Evaluate the model on the training set # Linear Regression
    train_pred = model.predict(X_train)

    train_r2 = r2_score(y_train, train_pred)

    test_pred = model.predict(X_test)

    test_r2 = r2_score(y_test, test_pred)

    print("Train R²:", train_r2)
    print("Test R² :", test_r2)
    
    ## Evaluate the model on the test set # Linear Regression
    y_pred = predict(model, X_test)
    
    mse, r2, mae = evaluate_model(y_test, y_pred)
    print("\nlinear regreesion mse:")
    print(mse)
    
    print("\nliner regression R2:")
    print(r2)
    
    print("\nlinear regressionmae:")
    print(mae)
    
    # Random Forest Model
    model = random_forest_train_model(X_train, y_train)
    y_pred = random_forest_predict(model, X_test)
    mse, r2, mae = random_forest_evaluate_model(y_test, y_pred)
    print("\nrandom forset mse:")
    print(mse)
    
    print("\nrandom foresr R2:")
    print(r2)
    
    print("\nrandom forest mae:")
    print(mae)

   ## Compare models
    
    results, models = compare_models(X_train, X_test, y_train, y_test)

    # Get best model
    best_name, best_model = get_best_model(results, models)
    print(f"\n✅ Best Model: {best_name} , {best_model}")
    
if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import make_pipeline

def load_data(file_path):
    """
    Read Excel data file
    :param file_path: Excel file path
    :return: DataFrame data
    """
    try:
        df = pd.read_excel(file_path)
        print(f"Successfully loaded data, shape: {df.shape}")
        print(f"Column names: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Error reading data: {e}")
        return None

def perform_regression_analysis(df):
    """
    Perform multiple regression analyses
    :param df: Data DataFrame
    :return: Dictionary containing results from various models
    """
    # Separate features and target variable
    # First 3 columns as independent variables, last column as dependent variable
    X = df.iloc[:, :3]  # First 3 columns: T1 chromaticity, T2 chromaticity, T3 chromaticity
    y = df.iloc[:, -1]  # Last column: MLSS actual value
    
    # Split training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create model dictionary
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Quadratic Polynomial Regression': make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
        'Cubic Polynomial Regression': make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
    }
    
    results = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Predict
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Predict on entire dataset to get all predicted values
        y_all_pred = model.predict(X)
        
        # Calculate R² score and Mean Squared Error
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        
        print(f"\n{name} Results:")
        print(f"  Training Set R² Score: {train_r2:.4f}")
        print(f"  Testing Set R² Score: {test_r2:.4f}")
        print(f"  Training Set MSE: {train_mse:.4f}")
        print(f"  Testing Set MSE: {test_mse:.4f}")
        
        # Save results
        results[name] = {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'y_test': y_test,
            'y_test_pred': y_test_pred,
            'y_all_pred': y_all_pred,  # Add all predicted values
            'X_test': X_test  # Save test set features for visualization
        }
        
        # Output fitting function formula
        if name == 'Linear Regression' or name == 'Ridge Regression' or name == 'Lasso Regression':
            # For linear models, output coefficients
            coef = model.coef_
            intercept = model.intercept_
            feature_names = X.columns.tolist() if hasattr(X, 'columns') else ['T1 Chromaticity', 'T2 Chromaticity', 'T3 Chromaticity']
            
            print(f"  Fitting Function: MLSS Actual = {intercept:.4f}", end="")
            for i, c in enumerate(coef):
                print(f" + ({c:.4f} * {feature_names[i]})", end="")
            print()
        elif name == 'Quadratic Polynomial Regression':
            # Get polynomial feature transformer
            poly_model = model.named_steps['polynomialfeatures']
            linear_model = model.named_steps['linearregression']
            
            # Get feature names - use correct feature names
            feature_names = X.columns.tolist() if hasattr(X, 'columns') else ['T1', 'T2', 'T3']
            try:
                poly_feature_names = poly_model.get_feature_names_out(feature_names)
            except:
                # If above method fails, use old method
                poly_feature_names = poly_model.get_feature_names(input_features=feature_names)
            
            print(f"  Fitting Function: MLSS Actual = {linear_model.intercept_:.4f}", end="")
            for i, coef in enumerate(linear_model.coef_):
                if abs(coef) > 1e-6:  # Only display terms with coefficient absolute value greater than threshold
                    print(f" + ({coef:.4f} * {poly_feature_names[i]})", end="")
            print(f"\n  (Note: Quadratic polynomial regression includes original features and all quadratic terms and interactions)")
        elif name == 'Cubic Polynomial Regression':
            # Get polynomial feature transformer
            poly_model = model.named_steps['polynomialfeatures']
            linear_model = model.named_steps['linearregression']
            
            # Get feature names - use correct feature names
            feature_names = X.columns.tolist() if hasattr(X, 'columns') else ['T1', 'T2', 'T3']
            try:
                poly_feature_names = poly_model.get_feature_names_out(feature_names)
            except:
                # If above method fails, use old method
                poly_feature_names = poly_model.get_feature_names(input_features=feature_names)
            
            print(f"  Fitting Function: MLSS Actual = {linear_model.intercept_:.4f}", end="")
            for i, coef in enumerate(linear_model.coef_):
                if abs(coef) > 1e-6:  # Only display terms with coefficient absolute value greater than threshold
                    print(f" + ({coef:.4f} * {poly_feature_names[i]})", end="")
            print(f"\n  (Note: Cubic polynomial regression includes original features and all cubic terms and interactions)")
    
    return results



def print_all_predictions(results, df):
    """
    Print predicted values for all models
    :param results: Model results dictionary
    :param df: Original data
    """
    # Generate all model predictions for entire dataset
    y_all = df.iloc[:, -1]  # Actual values for entire dataset
    
    # Create DataFrame containing actual values and all predicted values
    predictions_df = pd.DataFrame({'Actual': y_all.values})
    
    for name, result in results.items():
        predictions_df[f'{name}_Predicted'] = result['y_all_pred']
    
    print("\nPredicted Values for All Models (Entire Dataset):")
    print(predictions_df.head(10))  # Display first 10 rows
    
    # Save prediction results to CSV file
    try:
        predictions_df.to_csv('all_predictions.csv', encoding='utf-8-sig')
        print(f"\nPrediction results saved to 'all_predictions.csv' file, {len(predictions_df)} rows of data")
    except PermissionError:
        print("Warning: Unable to save to all_predictions.csv, file may be open in another program. Please close Excel or other programs and try again.")
        # Save to backup filename
        predictions_df.to_csv('all_predictions_backup.csv', encoding='utf-8-sig')
        print(f"Prediction results saved to 'all_predictions_backup.csv' file, {len(predictions_df)} rows of data")
    
    return predictions_df

def generate_summary_report(results, df):
    """
    Generate model comparison summary report
    :param results: Model results dictionary
    :param df: Original data
    """
    best_r2_model = max(results.keys(), key=lambda x: results[x]['test_r2'])
    best_mse_model = min(results.keys(), key=lambda x: results[x]['test_mse'])
    
    report = f"""
# Regression Analysis Summary Report

## Data Information
- Dataset Size: {df.shape[0]} rows, {df.shape[1]} columns
- Independent Variables: T1 Chromaticity, T2 Chromaticity, T3 Chromaticity
- Dependent Variable: MLSS Actual

## Model Performance Comparison

| Model Name | Test Set R² | Test Set MSE |
|-----------|-------------|--------------|
"""
    
    for name, result in results.items():
        report += f"| {name} | {result['test_r2']:.4f} | {result['test_mse']:.4f} |\n"
    
    report += f"""

## Best Model
- Highest R² Score: {best_r2_model} (R² = {results[best_r2_model]['test_r2']:.4f})
- Lowest MSE: {best_mse_model} (MSE = {results[best_mse_model]['test_mse']:.4f})

## Conclusion
- Higher R² score indicates stronger model ability to explain data variation
- Lower MSE indicates smaller prediction error
"""
    
    with open('regression_analysis_summary.md', 'w', encoding='utf-8-sig') as f:
        f.write(report)
    
    print("\nDetailed summary report saved to 'regression_analysis_summary.md'")

def main():
    """
    Main function
    """
    print("Starting regression analysis...")
    
    # Load data - use file in current directory
    df = load_data('SPSS-hg.xlsx')
    if df is None:
        print("Attempting to load data using relative path...")
        df = load_data('./SPSS-hg.xlsx')
        if df is None:
            print("Cannot find data file. Please ensure 'SPSS-hg.xlsx' file exists in current directory.")
            return
    
    # Perform regression analysis
    results = perform_regression_analysis(df)
    
    # Print all predicted values
    predictions_df = print_all_predictions(results, df)
    
    # Add fitting formula to last row of prediction results CSV file
    formulas = {}
    # Get feature names
    X = df.iloc[:, :3]  # First 3 columns: T1 chromaticity, T2 chromaticity, T3 chromaticity
    feature_names = X.columns.tolist() if hasattr(X, 'columns') else ['T1', 'T2', 'T3']
    
    for name, result in results.items():
        if name == 'Linear Regression' or name == 'Ridge Regression' or name == 'Lasso Regression':
            # For linear models, output coefficients
            model = result['model']
            coef = model.coef_
            intercept = model.intercept_
            # Use correct feature names
            actual_feature_names = X.columns.tolist() if hasattr(X, 'columns') else ['T1', 'T2', 'T3']
            
            formula = f"MLSS Actual = {intercept:.4f}"
            for i, c in enumerate(coef):
                formula += f" + ({c:.4f} * {actual_feature_names[i]})"
            formulas[f'{name}_Predicted'] = formula
        elif name == 'Quadratic Polynomial Regression':
            # Get polynomial feature transformer and linear model
            model = result['model']
            poly_model = model.named_steps['polynomialfeatures']
            linear_model = model.named_steps['linearregression']
            
            # Get feature names - use correct feature names
            try:
                poly_feature_names = poly_model.get_feature_names_out(feature_names)
            except:
                # If above method fails, use old method
                poly_feature_names = poly_model.get_feature_names(input_features=feature_names)
            
            # Build formula string
            formula = f"MLSS Actual = {linear_model.intercept_:.4f}"
            for i, coef in enumerate(linear_model.coef_):
                if abs(coef) > 1e-6:  # Only include terms with coefficient absolute value greater than threshold
                    formula += f" + ({coef:.4f} * {poly_feature_names[i]})"
            formulas[f'{name}_Predicted'] = formula
        elif name == 'Cubic Polynomial Regression':
            # Get polynomial feature transformer and linear model
            model = result['model']
            poly_model = model.named_steps['polynomialfeatures']
            linear_model = model.named_steps['linearregression']
            
            # Get feature names - use correct feature names
            try:
                poly_feature_names = poly_model.get_feature_names_out(feature_names)
            except:
                # If above method fails, use old method
                poly_feature_names = poly_model.get_feature_names(input_features=feature_names)
            
            # Build formula string
            formula = f"MLSS Actual = {linear_model.intercept_:.4f}"
            for i, coef in enumerate(linear_model.coef_):
                if abs(coef) > 1e-6:  # Only include terms with coefficient absolute value greater than threshold
                    formula += f" + ({coef:.4f} * {poly_feature_names[i]})"
            formulas[f'{name}_Predicted'] = formula
    
    # Add a row containing formulas to DataFrame
    formula_row = {'Actual': 'Fitting Formula'}
    for col, formula in formulas.items():
        formula_row[col] = formula
    
    # Add formula row to end of DataFrame
    all_predictions = pd.read_csv('all_predictions.csv', index_col=0)
    all_predictions.loc['Formula'] = pd.Series(formula_row)
    
    # Re-save CSV file containing formulas
    try:
        all_predictions.to_csv('all_predictions.csv', encoding='utf-8-sig')
    except PermissionError:
        print("Warning: Unable to save to all_predictions.csv, file may be open in another program. Please close Excel or other programs and try again.")
        # Save to backup filename
        all_predictions.to_csv('all_predictions_updated.csv', encoding='utf-8-sig')
        print("Updated prediction results saved to 'all_predictions_updated.csv'")
    
    # Generate summary report
    generate_summary_report(results, df)
    
    print("\nRegression analysis completed!")
    print("- All predicted values saved to 'all_predictions.csv' (with fitting formula in last row)")
    print("- Detailed report saved to 'regression_analysis_summary.md'")

if __name__ == "__main__":
    main()

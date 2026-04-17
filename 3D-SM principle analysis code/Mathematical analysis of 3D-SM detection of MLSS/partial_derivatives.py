import pandas as pd
import numpy as np
from sympy import symbols, diff, lambdify
import os

def calculate_partial_derivatives():
    """
    Calculate partial derivatives of cubic polynomial regression function with respect to three input variables
    """
    print("Calculating partial derivatives...")
    
    # Define symbolic variables
    T1, T2, T3 = symbols('T1 T2 T3')
    
    # Cubic polynomial regression function expression (based on previous regression analysis results)
    # MLSS actual = 3.5959 + (0.1179 * T1) + (-0.2513 * T2) + (0.1853 * T3) + 
    #              (-0.0019 * T1^2) + (0.0012 * T1*T2) + (0.0004 * T1*T3) + 
    #              (-0.0046 * T2^2) + (0.0131 * T2*T3) + (-0.0104 * T3^2) + 
    #              (0.0001 * T1^2*T2) + (-0.0000 * T1^2*T3) + (-0.0002 * T1*T2^2) + 
    #              (0.0002 * T1*T2*T3) + (-0.0001 * T1*T3^2) + (0.0002 * T2^3) + 
    #              (-0.0005 * T2^2*T3) + (0.0004 * T2*T3^2) + (-0.0001 * T3^3)
    
    function_expr = (3.5959 + 
                     0.1179 * T1 + 
                     (-0.2513) * T2 + 
                     0.1853 * T3 + 
                     (-0.0019) * T1**2 + 
                     0.0012 * T1 * T2 + 
                     0.0004 * T1 * T3 + 
                     (-0.0046) * T2**2 + 
                     0.0131 * T2 * T3 + 
                     (-0.0104) * T3**2 + 
                     0.0001 * T1**2 * T2 + 
                     0 * T1**2 * T3 +  # This term coefficient is 0
                     (-0.0002) * T1 * T2**2 + 
                     0.0002 * T1 * T2 * T3 + 
                     (-0.0001) * T1 * T3**2 + 
                     0.0002 * T2**3 + 
                     (-0.0005) * T2**2 * T3 + 
                     0.0004 * T2 * T3**2 + 
                     (-0.0001) * T3**3)
    
    # Calculate partial derivatives
    partial_T1 = diff(function_expr, T1)
    partial_T2 = diff(function_expr, T2)
    partial_T3 = diff(function_expr, T3)
    
    print("Partial derivative functions calculated:")
    print(f"∂f/∂T1 = {partial_T1}")
    print(f"∂f/∂T2 = {partial_T2}")
    print(f"∂f/∂T3 = {partial_T3}")
    
    # Convert symbolic expressions to callable numerical functions
    func_T1 = lambdify((T1, T2, T3), partial_T1, 'numpy')
    func_T2 = lambdify((T1, T2, T3), partial_T2, 'numpy')
    func_T3 = lambdify((T1, T2, T3), partial_T3, 'numpy')
    
    # Read input data
    excel_file = 'SPSS-hg.xlsx'
    if not os.path.exists(excel_file):
        print(f"Error: File {excel_file} not found")
        return
    
    print(f"Reading data file: {excel_file}")
    df = pd.read_excel(excel_file)
    
    print(f"Data column names: {list(df.columns)}")
    
    # Based on previous analysis, data column names are ['T1', 'T2', 'T3', 'MLSS']
    # First three columns are T1, T2, T3
    if 'T1' in df.columns and 'T2' in df.columns and 'T3' in df.columns:
        T1_vals = df['T1'].values
        T2_vals = df['T2'].values
        T3_vals = df['T3'].values
        print("Using column names T1, T2, T3 as input variables")
    else:
        # Use first three columns
        T1_vals = df.iloc[:, 0].values
        T2_vals = df.iloc[:, 1].values
        T3_vals = df.iloc[:, 2].values
        print(f"Using first three columns as input variables: {df.columns[0]}, {df.columns[1]}, {df.columns[2]}")
    
    print(f"Data reading completed, {len(T1_vals)} samples")
    
    # Calculate partial derivatives for each sample point
    partial_T1_values = func_T1(T1_vals, T2_vals, T3_vals)
    partial_T2_values = func_T2(T1_vals, T2_vals, T3_vals)
    partial_T3_values = func_T3(T1_vals, T2_vals, T3_vals)
    
    # Create result DataFrame
    result_df = pd.DataFrame({
        'T1': T1_vals,
        'T2': T2_vals,
        'T3': T3_vals,
        '∂f/∂T1': partial_T1_values,
        '∂f/∂T2': partial_T2_values,
        '∂f/∂T3': partial_T3_values
    })
    
    # Save results to Excel file
    output_file = 'partial_derivative_results.xlsx'
    result_df.to_excel(output_file, index=False)
    print(f"Partial derivative calculation results saved to: {output_file}")
    
    # Display first few rows of results
    print("\nPartial derivative calculation results preview:")
    print(result_df.head(10))
    
    print("\nCalculation completed!")


if __name__ == "__main__":
    calculate_partial_derivatives()


# Regression Analysis Summary Report

## Data Information
- Dataset Size: 1387 rows, 4 columns
- Independent Variables: T1 Chromaticity, T2 Chromaticity, T3 Chromaticity
- Dependent Variable: MLSS Actual

## Model Performance Comparison

| Model Name | Test Set R² | Test Set MSE |
|-----------|-------------|--------------|
| Linear Regression | 0.4355 | 2.3529 |
| Ridge Regression | 0.4355 | 2.3529 |
| Lasso Regression | 0.4357 | 2.3517 |
| Quadratic Polynomial Regression | 0.6375 | 1.5109 |
| Cubic Polynomial Regression | 0.6799 | 1.3342 |


## Best Model
- Highest R² Score: Cubic Polynomial Regression (R² = 0.6799)
- Lowest MSE: Cubic Polynomial Regression (MSE = 1.3342)

## Conclusion
- Higher R² score indicates stronger model ability to explain data variation
- Lower MSE indicates smaller prediction error

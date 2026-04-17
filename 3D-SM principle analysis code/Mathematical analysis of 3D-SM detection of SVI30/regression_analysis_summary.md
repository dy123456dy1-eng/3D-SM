
# Regression Analysis Summary Report

## Data Information
- Dataset Size: 1377 rows, 4 columns
- Independent Variables: T1 Chromaticity, T2 Chromaticity, T3 Chromaticity
- Dependent Variable: SVI30 Actual

## Model Performance Comparison

| Model Name | Test Set R² | Test Set MSE |
|-----------|-------------|--------------|
| Linear Regression | 0.1851 | 26796.1797 |
| Ridge Regression | 0.1851 | 26796.1520 |
| Lasso Regression | 0.1853 | 26791.5670 |
| Quadratic Polynomial Regression | 0.5788 | 13850.3616 |
| Cubic Polynomial Regression | 0.6606 | 11160.7799 |


## Best Model
- Highest R² Score: Cubic Polynomial Regression (R² = 0.6606)
- Lowest MSE: Cubic Polynomial Regression (MSE = 11160.7799)

## Conclusion
- Higher R² score indicates stronger model ability to explain data variation
- Lower MSE indicates smaller prediction error

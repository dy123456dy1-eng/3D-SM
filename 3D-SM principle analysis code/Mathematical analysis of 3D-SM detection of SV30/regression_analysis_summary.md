
# Regression Analysis Summary Report

## Data Information
- Dataset Size: 1387 rows, 4 columns
- Independent Variables: T1 Chromaticity, T2 Chromaticity, T3 Chromaticity
- Dependent Variable: SV30 Actual

## Model Performance Comparison

| Model Name | Test Set R² | Test Set MSE |
|-----------|-------------|--------------|
| Linear Regression | 0.5902 | 155.1637 |
| Ridge Regression | 0.5902 | 155.1631 |
| Lasso Regression | 0.5919 | 154.5087 |
| Quadratic Polynomial Regression | 0.6881 | 118.0754 |
| Cubic Polynomial Regression | 0.7397 | 98.5432 |


## Best Model
- Highest R² Score: Cubic Polynomial Regression (R² = 0.7397)
- Lowest MSE: Cubic Polynomial Regression (MSE = 98.5432)

## Conclusion
- Higher R² score indicates stronger model ability to explain data variation
- Lower MSE indicates smaller prediction error

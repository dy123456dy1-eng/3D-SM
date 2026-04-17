# Differential Testing Analysis Report

## 1. Data Basic Information

- **Data Source**: Differential testing.xlsx
- **Data Shape**: (398 rows, 3 columns)
- **Column Names**: T1, T2, T3
- **Data Types**: All float64
- **Missing Values**:
  - T1: 0 missing values
  - T2: 9 missing values
  - T3: 17 missing values

## 2. Descriptive Statistics

| Statistical Indicator | T1 | T2 | T3 |
|----------------------|----|----|----|
| Sample Size | 398 | 389 | 381 |
| Mean | 82.61 | 77.23 | 69.86 |
| Standard Deviation | 43.23 | 44.13 | 43.66 |
| Minimum Value | 0.00 | 0.00 | 0.00 |
| 25th Percentile | 49.23 | 35.10 | 26.33 |
| 50th Percentile (Median) | 95.50 | 89.16 | 76.33 |
| 75th Percentile | 119.70 | 115.74 | 109.53 |
| Maximum Value | 139.76 | 143.17 | 141.42 |

## 3. Normality Test (Shapiro-Wilk Test)

- **T1**: Statistic = 0.9070, p-value = 0.0000 → Data does not follow normal distribution (p ≤ 0.05)
- **T2**: Statistic = 0.9180, p-value = 0.0000 → Data does not follow normal distribution (p ≤ 0.05)
- **T3**: Statistic = 0.9212, p-value = 0.0000 → Data does not follow normal distribution (p ≤ 0.05)

**Conclusion**: All column data does not meet the normal distribution assumption.

## 4. Homogeneity of Variance Test (Levene Test)

- **Statistic**: 1.6959
- **p-value**: 0.1839
- **Conclusion**: Homogeneity of variance (p > 0.05), parametric tests can be used

## 5. Parametric Test (One-way ANOVA)

- **F-statistic**: 8.3390
- **p-value**: 0.0003
- **Conclusion**: At least two groups have significant differences (p < 0.05)

## 6. Non-parametric Test (Kruskal-Wallis H Test)

- **H-statistic**: 18.0693
- **p-value**: 0.0001
- **Conclusion**: At least two groups have significant differences (p < 0.05)

## 7. Post-hoc Multiple Comparisons Test (Mann-Whitney U Test)

- **T1 vs T2**:
  - U-statistic: 82874.5000
  - p-value: 0.0866
  - Conclusion: No significant difference between the two groups (p ≥ 0.05)

- **T1 vs T3**:
  - U-statistic: 88966.0000
  - p-value: 0.0000
  - Conclusion: Significant difference between the two groups (p < 0.05)

- **T2 vs T3**:
  - U-statistic: 82064.0000
  - p-value: 0.0099
  - Conclusion: Significant difference between the two groups (p < 0.05)

## 8. Comprehensive Conclusions

### 8.1 Data Characteristics
- The sample sizes of the three groups (T1, T2, T3) are similar, with 398, 389, and 381 samples respectively
- Data means: T1 (82.61) > T2 (77.23) > T3 (69.86)
- The standard deviations of the three groups are similar (approximately 43-44), indicating similar dispersion
- All data values are non-negative, with minimum value of 0

### 8.2 Statistical Test Results
1. **Normality**: Data from all three groups does not follow normal distribution
2. **Homogeneity of Variance**: Data from all three groups meets homogeneity of variance assumption
3. **Inter-group Differences**: Significant differences exist among the three groups

### 8.3 Specific Difference Analysis
- **T1 vs T3**: Significant difference (p < 0.05), T1 values are significantly higher than T3
- **T2 vs T3**: Significant difference (p < 0.05), T2 values are significantly higher than T3
- **T1 vs T2**: No significant difference (p ≥ 0.05), the two groups are statistically similar

### 8.4 Practical Significance
- T3 group data values are generally lower than T1 and T2 groups, which may indicate that T3 group performs worse or has lower values under certain conditions
- No significant difference between T1 and T2 groups suggests that these two groups may have similar experimental conditions or treatment methods
- From the means: T1 (82.61) > T2 (77.23) > T3 (69.86)

## 9. Recommendations

1. **Further Analysis**: Consider using other statistical methods to further explore potential differences between T1 and T2
2. **Data Exploration**: Check for outliers or special patterns in the data
3. **Experimental Design**: If this is experimental data, check whether experimental conditions affected the results
4. **Follow-up Analysis**: Consider calculating effect sizes to assess the practical importance of differences

## 10. Notes

- Since data does not follow normal distribution, non-parametric test results are more reliable
- Although homogeneity of variance test shows homogeneity, non-parametric methods are more robust when data is non-normal
- Post-hoc test results indicate that although overall ANOVA shows significant differences, the specific difference between T1 and T2 is not significant

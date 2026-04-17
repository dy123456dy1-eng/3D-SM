# -*- coding: utf-8 -*-
"""
Differential testing analysis script
This script is used to analyze the differences in 3 columns of data in an Excel file
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, levene, f_oneway, kruskal
import warnings
warnings.filterwarnings('ignore')

# Set Chinese font support
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # To display Chinese labels normally
plt.rcParams['axes.unicode_minus'] = False  # To display negative signs normally

def read_excel_data(file_path):
    """
    Read Excel file data
    Parameters:
        file_path: Excel file path
    Returns:
        DataFrame: DataFrame containing data
    """
    try:
        # Read Excel file
        df = pd.read_excel(file_path)
        print(f"Successfully read Excel file: {file_path}")
        print(f"Data shape: {df.shape}")
        print(f"Column names: {list(df.columns)}")
        print("\nData preview:")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

def descriptive_statistics(df):
    """
    Calculate descriptive statistics
    Parameters:
        df: DataFrame
    Returns:
        Descriptive statistics results
    """
    print("\n" + "="*50)
    print("Descriptive Statistical Analysis")
    print("="*50)
    
    # Basic statistical information
    desc_stats = df.describe()
    print("\nBasic Statistical Information:")
    print(desc_stats)
    
    # Number of missing values for each column
    missing_values = df.isnull().sum()
    print(f"\nMissing Values Statistics:")
    print(missing_values)
    
    # Number of non-null data for each column
    non_null_counts = df.count()
    print(f"\nNon-null Data Count:")
    print(non_null_counts)
    
    # Data type for each column
    print(f"\nData Types:")
    print(df.dtypes)
    
    return desc_stats

def normality_test(df):
    """
    Normality test
    Parameters:
        df: DataFrame
    Returns:
        Normality test results
    """
    print("\n" + "="*50)
    print("Normality Test (Shapiro-Wilk Test)")
    print("="*50)
    
    results = {}
    for col in df.columns:
        # Remove missing values
        data = df[col].dropna()
        if len(data) >= 3 and len(data) <= 5000:  # Shapiro test applicable range
            stat, p_value = shapiro(data)
            results[col] = {'statistic': stat, 'p_value': p_value}
            print(f"\n{col}:")
            print(f"  Statistic: {stat:.4f}")
            print(f"  p-value: {p_value:.4f}")
            if p_value > 0.05:
                print(f"  Conclusion: Data follows normal distribution (p > 0.05)")
            else:
                print(f"  Conclusion: Data does not follow normal distribution (p ≤ 0.05)")
        else:
            print(f"\n{col}: Data size is outside the applicable range for Shapiro test, skipping test")
            results[col] = {'statistic': None, 'p_value': None}
    
    return results

def homogeneity_test(df):
    """
    Homogeneity of variance test
    Parameters:
        df: DataFrame
    Returns:
        Homogeneity of variance test results
    """
    print("\n" + "="*50)
    print("Homogeneity of Variance Test (Levene Test)")
    print("="*50)
    
    # Prepare data
    data_to_test = []
    labels = []
    for col in df.columns:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            data_to_test.append(col_data.values)
            labels.append(col)
    
    if len(data_to_test) >= 2:
        stat, p_value = levene(*data_to_test)
        print(f"Statistic: {stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        if p_value > 0.05:
            print("Conclusion: Homogeneity of variance (p > 0.05), parametric tests can be used")
        else:
            print("Conclusion: Heterogeneity of variance (p ≤ 0.05), non-parametric tests are recommended")
        
        return {'statistic': stat, 'p_value': p_value}
    else:
        print("Insufficient number of data columns, cannot perform homogeneity of variance test")
        return None

def parametric_test(df):
    """
    Parametric test (ANOVA)
    Parameters:
        df: DataFrame
    Returns:
        Parametric test results
    """
    print("\n" + "="*50)
    print("Parametric Test (One-way ANOVA)")
    print("="*50)
    
    # Prepare data
    data_to_test = []
    labels = []
    for col in df.columns:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            data_to_test.append(col_data.values)
            labels.append(col)
    
    if len(data_to_test) >= 2:
        # Perform one-way ANOVA
        f_stat, p_value = f_oneway(*data_to_test)
        print(f"F-statistic: {f_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print("Conclusion: At least two groups have significant differences (p < 0.05)")
            print("Post-hoc multiple comparisons test is needed to determine which specific groups differ")
        else:
            print("Conclusion: No significant differences between groups (p ≥ 0.05)")
        
        return {'statistic': f_stat, 'p_value': p_value}
    else:
        print("Insufficient number of data columns, cannot perform ANOVA")
        return None

def non_parametric_test(df):
    """
    Non-parametric test (Kruskal-Wallis test)
    Parameters:
        df: DataFrame
    Returns:
        Non-parametric test results
    """
    print("\n" + "="*50)
    print("Non-parametric Test (Kruskal-Wallis H Test)")
    print("="*50)
    
    # Prepare data
    data_to_test = []
    labels = []
    for col in df.columns:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            data_to_test.append(col_data.values)
            labels.append(col)
    
    if len(data_to_test) >= 2:
        # Perform Kruskal-Wallis test
        h_stat, p_value = kruskal(*data_to_test)
        print(f"H-statistic: {h_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print("Conclusion: At least two groups have significant differences (p < 0.05)")
        else:
            print("Conclusion: No significant differences between groups (p ≥ 0.05)")
        
        return {'statistic': h_stat, 'p_value': p_value}
    else:
        print("Insufficient number of data columns, cannot perform Kruskal-Wallis test")
        return None

def post_hoc_test(df):
    """
    Post-hoc multiple comparisons test (if ANOVA is significant)
    Parameters:
        df: DataFrame
    """
    print("\n" + "="*50)
    print("Post-hoc Multiple Comparisons Test (Tukey HSD or Mann-Whitney U)")
    print("="*50)
    
    from itertools import combinations
    
    # Get all column names
    cols = df.columns.tolist()
    
    # If data meets normality and homogeneity of variance, use Tukey HSD
    # Otherwise use Mann-Whitney U test
    print("Since it cannot be determined whether the data meets the assumptions of normality and homogeneity of variance, Mann-Whitney U test results will be displayed:")
    
    for col1, col2 in combinations(cols, 2):
        data1 = df[col1].dropna()
        data2 = df[col2].dropna()
        
        # Perform Mann-Whitney U test
        u_stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        
        print(f"\n{col1} vs {col2}:")
        print(f"  U-statistic: {u_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        if p_value < 0.05:
            print(f"  Conclusion: Significant difference between the two groups (p < 0.05)")
        else:
            print(f"  Conclusion: No significant difference between the two groups (p ≥ 0.05)")

def visualize_data(df):
    """
    Visualize data distribution
    Parameters:
        df: DataFrame
    """
    print("\n" + "="*50)
    print("Data Visualization")
    print("="*50)
    
    # Set graph style
    plt.style.use('default')
    
    # Create figure
    n_cols = len(df.columns)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Data Distribution Visualization Analysis', fontsize=16, fontweight='bold')
    
    # 1. Boxplot - Show data distribution and outliers
    df.boxplot(ax=axes[0,0])
    axes[0,0].set_title('Boxplot - Data Distribution')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Violin plot - Show data distribution density
    df_melted = df.melt(var_name='Variable', value_name='Value')
    sns.violinplot(data=df_melted, x='Variable', y='Value', ax=axes[0,1])
    axes[0,1].set_title('Violin Plot - Data Distribution Density')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Histogram - Show data distribution for each column
    axes[1,0].clear()
    for i, col in enumerate(df.columns):
        data = df[col].dropna()
        axes[1,0].hist(data, alpha=0.5, label=col, bins=min(20, len(data)//2+1))
    axes[1,0].set_title('Histogram - Data Distribution for Each Column')
    axes[1,0].legend()
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 4. Scatter plot matrix simplified version - Show correlation
    if n_cols >= 2:
        # If 2 or 3 columns, plot scatter plot
        if n_cols == 2:
            axes[1,1].scatter(df.iloc[:, 0], df.iloc[:, 1], alpha=0.6)
            axes[1,1].set_xlabel(df.columns[0])
            axes[1,1].set_ylabel(df.columns[1])
            axes[1,1].set_title(f'{df.columns[0]} vs {df.columns[1]}')
        elif n_cols >= 3:
            axes[1,1].scatter(df.iloc[:, 0], df.iloc[:, 1], alpha=0.6, label=df.columns[2])
            axes[1,1].set_xlabel(df.columns[0])
            axes[1,1].set_ylabel(df.columns[1])
            axes[1,1].set_title(f'{df.columns[0]} vs {df.columns[1]} (Grouped by {df.columns[2]})')
            axes[1,1].legend()
    else:
        axes[1,1].text(0.5, 0.5, 'Insufficient number of columns\nCannot plot scatter plot', 
                      horizontalalignment='center', verticalalignment='center',
                      transform=axes[1,1].transAxes, fontsize=14)
        axes[1,1].set_title('Scatter Plot')
    
    plt.tight_layout()
    plt.show()

def generate_report(df, normality_results, homogeneity_result, parametric_result, non_parametric_result):
    """
    Generate analysis report
    Parameters:
        df: DataFrame
        normality_results: Normality test results
        homogeneity_result: Homogeneity of variance test results
        parametric_result: Parametric test results
        non_parametric_result: Non-parametric test results
    """
    print("\n" + "="*60)
    print("Comprehensive Analysis Report of Differential Testing")
    print("="*60)
    
    print(f"\nData Basic Information:")
    print(f"- Data shape: {df.shape}")
    print(f"- Column names: {list(df.columns)}")
    
    print(f"\nNormality Test Summary:")
    for col, result in normality_results.items():
        if result['p_value'] is not None:
            status = "Follows normal distribution" if result['p_value'] > 0.05 else "Does not follow normal distribution"
            print(f"- {col}: {status} (p={result['p_value']:.4f})")
        else:
            print(f"- {col}: Cannot be tested (data size outside applicable range)")
    
    print(f"\nHomogeneity of Variance Test Summary:")
    if homogeneity_result:
        status = "Homogeneity of variance" if homogeneity_result['p_value'] > 0.05 else "Heterogeneity of variance"
        print(f"- {status} (p={homogeneity_result['p_value']:.4f})")
    
    print(f"\nParametric Test (ANOVA) Summary:")
    if parametric_result:
        status = "Significant differences exist" if parametric_result['p_value'] < 0.05 else "No significant differences"
        print(f"- {status} (p={parametric_result['p_value']:.4f})")
    
    print(f"\nNon-parametric Test (Kruskal-Wallis) Summary:")
    if non_parametric_result:
        status = "Significant differences exist" if non_parametric_result['p_value'] < 0.05 else "No significant differences"
        print(f"- {status} (p={non_parametric_result['p_value']:.4f})")
    
    print(f"\nConclusions and Recommendations:")
    if parametric_result and parametric_result['p_value'] < 0.05:
        print("- Parametric test results show significant differences between groups")
        print("- It is recommended to further perform post-hoc multiple comparisons test")
    elif non_parametric_result and non_parametric_result['p_value'] < 0.05:
        print("- Non-parametric test results show significant differences between groups")
        print("- Data may not meet normality or homogeneity of variance assumptions")
    else:
        print("- No significant differences found between groups")
        print("- Data among groups are not statistically different")
    
    print("\n" + "="*60)

def main():
    """
    Main function
    """
    # Excel file path - use current script directory
    import os
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'case analysis.xlsx')
    
    print(f"Analyzing data in file {file_path}...")
    
    # 1. Read Excel data
    df = read_excel_data(file_path)
    if df is None:
        return
    
    # Check if there are at least 3 columns of data
    if len(df.columns) < 3:
        print(f"Warning: Data has only {len(df.columns)} columns, expected at least 3")
        print("Continuing analysis with available columns...")
    
    # 2. Descriptive statistics
    desc_stats = descriptive_statistics(df)
    
    # 3. Normality test
    normality_results = normality_test(df)
    
    # 4. Homogeneity of variance test
    homogeneity_result = homogeneity_test(df)
    
    # 5. Parametric test (ANOVA)
    parametric_result = parametric_test(df)
    
    # 6. Non-parametric test (Kruskal-Wallis)
    non_parametric_result = non_parametric_test(df)
    
    # 7. Post-hoc multiple comparisons test
    if parametric_result and parametric_result['p_value'] < 0.05:
        post_hoc_test(df)
    
    # 8. Generate comprehensive report
    generate_report(df, normality_results, homogeneity_result, 
                   parametric_result, non_parametric_result)
    
    print("\nAnalysis completed!")

if __name__ == "__main__":
    main()

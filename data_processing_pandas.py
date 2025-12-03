#!/usr/bin/env python3
"""
Pandas-based Data Processing Script for Bank Customer Churn Prediction
This script loads, explores, cleans, and processes the bank customer data without PySpark.
"""

import pandas as pd
import numpy as np
import os

def main():
    print("Loading dataset...")
    
    # Load the raw data
    df = pd.read_csv("dataset/Churn_Modelling.csv")
    
    print(f"Dataset loaded successfully!")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    
    print("\n=== Dataset Schema ===")
    print(df.dtypes)
    
    print("\n=== First 5 Rows ===")
    print(df.head())
    
    print("\n=== Column Names ===")
    print(df.columns.tolist())
    
    print("\n=== Statistical Summary ===")
    print(df.describe())
    
    # Check for missing values
    print("\n=== Missing Values Check ===")
    print(df.isnull().sum())
    
    # Analyze target variable (Churn)
    print("\n=== Churn Distribution ===")
    print(df['Exited'].value_counts())
    
    total_customers = len(df)
    churned_customers = (df['Exited'] == 1).sum()
    churn_rate = (churned_customers / total_customers) * 100
    print(f"Churn Rate: {churn_rate:.2f}%")
    
    # Feature analysis
    print("\n=== Geography Distribution ===")
    print(df['Geography'].value_counts())
    
    print("\n=== Gender Distribution ===")
    print(df['Gender'].value_counts())
    
    print("\n=== Number of Products Distribution ===")
    print(df['NumOfProducts'].value_counts())
    
    print("\n=== Churn by Geography ===")
    print(pd.crosstab(df['Geography'], df['Exited']))
    
    print("\n=== Churn by Gender ===")
    print(pd.crosstab(df['Gender'], df['Exited']))
    
    # Data cleaning
    print("\nCleaning data...")
    df_clean = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    
    print("\n=== Cleaned Dataset Schema ===")
    print(df_clean.dtypes)
    
    # Feature engineering
    print("\nEngineering features...")
    
    # Age groups
    df_clean['AgeGroup'] = pd.cut(df_clean['Age'], 
                                   bins=[0, 30, 45, 100], 
                                   labels=['Young', 'Middle', 'Senior'])
    
    # Balance category
    df_clean['BalanceCategory'] = pd.cut(df_clean['Balance'],
                                          bins=[-1, 0, 50000, 100000, 300000],
                                          labels=['Zero', 'Low', 'Medium', 'High'])
    
    # Credit Score Category
    df_clean['CreditScoreCategory'] = pd.cut(df_clean['CreditScore'],
                                              bins=[0, 600, 700, 800, 900],
                                              labels=['Poor', 'Fair', 'Good', 'Excellent'])
    
    print("\n=== Dataset with Engineered Features ===")
    print(df_clean.head())
    
    # Save processed data
    print("\nSaving processed data...")
    
    # Create dataset directory if it doesn't exist
    os.makedirs("dataset", exist_ok=True)
    
    # Save as CSV
    df_clean.to_csv("dataset/processed_churn_data.csv", index=False)
    print("Processed data saved to 'dataset/processed_churn_data.csv'")
    
    # Final statistics
    print("\n=== Final Dataset Statistics ===")
    print(f"Total Records: {len(df_clean)}")
    print(f"Total Features: {len(df_clean.columns)}")
    print(f"\nFeature List: {df_clean.columns.tolist()}")
    
    # Feature correlations with churn
    print("\n=== Feature Correlations with Churn ===")
    numeric_features = ["CreditScore", "Age", "Tenure", "Balance", 
                       "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
    
    for feature in numeric_features:
        correlation = df_clean[feature].corr(df_clean['Exited'])
        print(f"{feature}: {correlation:.4f}")
    
    print("\n=== Data Processing Complete! ===")

if __name__ == "__main__":
    main()

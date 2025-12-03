#!/usr/bin/env python3
"""
PySpark Data Processing Script for Bank Customer Churn Prediction
This script loads, explores, cleans, and processes the bank customer data.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, isnan, isnull
from pyspark.sql.types import *

def main():
    # ============================================================================
    # STEP 1: Initialize PySpark Session
    # ============================================================================
    
    print("Initializing Spark Session...")
    spark = SparkSession.builder \
        .appName("BankCustomerChurnPrediction") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    print("Spark Session Created Successfully!")
    
    # ============================================================================
    # STEP 2: Load Dataset
    # ============================================================================
    
    print("\nLoading dataset...")
    df = spark.read.csv("dataset/Churn_Modelling.csv", header=True, inferSchema=True)
    
    print("Dataset loaded successfully!")
    print(f"Number of rows: {df.count()}")
    print(f"Number of columns: {len(df.columns)}")
    
    # ============================================================================
    # STEP 3: Initial Data Exploration
    # ============================================================================
    
    print("\n=== Dataset Schema ===")
    df.printSchema()
    
    print("\n=== First 5 Rows ===")
    df.show(5)
    
    print("\n=== Column Names ===")
    print(df.columns)
    
    print("\n=== Statistical Summary ===")
    df.describe().show()
    
    # ============================================================================
    # STEP 4: Check for Missing Values
    # ============================================================================
    
    print("\n=== Missing Values Check ===")
    # Only check isnan for numeric columns
    numeric_cols = [c for c in df.columns if df.schema[c].dataType.simpleString() in ['int', 'bigint', 'double', 'float']]
    string_cols = [c for c in df.columns if c not in numeric_cols]
    
    missing_counts = []
    for c in numeric_cols:
        missing_counts.append(count(when(isnan(c) | col(c).isNull(), c)).alias(c))
    for c in string_cols:
        missing_counts.append(count(when(col(c).isNull(), c)).alias(c))
    
    df.select(missing_counts).show()
    
    # ============================================================================
    # STEP 5: Analyze Target Variable (Churn)
    # ============================================================================
    
    print("\n=== Churn Distribution ===")
    df.groupBy("Exited").count().show()
    
    total_customers = df.count()
    churned_customers = df.filter(col("Exited") == 1).count()
    churn_rate = (churned_customers / total_customers) * 100
    print(f"Churn Rate: {churn_rate:.2f}%")
    
    # ============================================================================
    # STEP 6: Feature Analysis
    # ============================================================================
    
    print("\n=== Geography Distribution ===")
    df.groupBy("Geography").count().show()
    
    print("\n=== Gender Distribution ===")
    df.groupBy("Gender").count().show()
    
    print("\n=== Number of Products Distribution ===")
    df.groupBy("NumOfProducts").count().show()
    
    print("\n=== Churn by Geography ===")
    df.groupBy("Geography", "Exited").count().show()
    
    print("\n=== Churn by Gender ===")
    df.groupBy("Gender", "Exited").count().show()
    
    # ============================================================================
    # STEP 7: Data Cleaning and Preprocessing
    # ============================================================================
    
    print("\nCleaning data...")
    df_clean = df.drop("RowNumber", "CustomerId", "Surname")
    
    print("\n=== Cleaned Dataset Schema ===")
    df_clean.printSchema()
    
    # ============================================================================
    # STEP 8: Feature Engineering
    # ============================================================================
    
    print("\nEngineering features...")
    
    # Age groups
    df_clean = df_clean.withColumn("AgeGroup", 
        when(col("Age") < 30, "Young")
        .when((col("Age") >= 30) & (col("Age") < 45), "Middle")
        .when(col("Age") >= 45, "Senior")
    )
    
    # Balance category
    df_clean = df_clean.withColumn("BalanceCategory",
        when(col("Balance") == 0, "Zero")
        .when((col("Balance") > 0) & (col("Balance") <= 50000), "Low")
        .when((col("Balance") > 50000) & (col("Balance") <= 100000), "Medium")
        .when(col("Balance") > 100000, "High")
    )
    
    # Credit Score Category
    df_clean = df_clean.withColumn("CreditScoreCategory",
        when(col("CreditScore") < 600, "Poor")
        .when((col("CreditScore") >= 600) & (col("CreditScore") < 700), "Fair")
        .when((col("CreditScore") >= 700) & (col("CreditScore") < 800), "Good")
        .when(col("CreditScore") >= 800, "Excellent")
    )
    
    print("\n=== Dataset with Engineered Features ===")
    df_clean.show(5)
    
    # ============================================================================
    # STEP 9: Save Processed Data
    # ============================================================================
    
    print("\nSaving processed data...")
    
    # Save as CSV only (Parquet has Hadoop issues on Windows)
    df_clean.coalesce(1).write.mode("overwrite") \
        .option("header", "true") \
        .csv("dataset/processed_churn_data_csv")
    
    print("\n=== Processed data saved successfully! ===")
    
    # ============================================================================
    # STEP 10: Final Statistics
    # ============================================================================
    
    print("\n=== Final Dataset Statistics ===")
    print(f"Total Records: {df_clean.count()}")
    print(f"Total Features: {len(df_clean.columns)}")
    print(f"\nFeature List: {df_clean.columns}")
    
    print("\n=== Feature Correlations with Churn ===")
    numeric_features = ["CreditScore", "Age", "Tenure", "Balance", 
                       "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
    
    for feature in numeric_features:
        correlation = df_clean.stat.corr(feature, "Exited")
        print(f"{feature}: {correlation:.4f}")
    
    print("\n=== Data Processing Complete! ===")
    
    spark.stop()

if __name__ == "__main__":
    main()

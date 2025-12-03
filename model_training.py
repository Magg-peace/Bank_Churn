#!/usr/bin/env python3
"""
PySpark MLlib Model Training Script for Bank Customer Churn Prediction
This script trains multiple classification models and saves the best one.
"""

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col

def main():
    # ============================================================================
    # STEP 1: Initialize PySpark Session
    # ============================================================================
    
    print("Initializing Spark Session...")
    spark = SparkSession.builder \
        .appName("BankChurnModelTraining") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    
    print("Spark Session Created Successfully!")
    
    # ============================================================================
    # STEP 2: Load Preprocessed Data
    # ============================================================================
    
    print("\nLoading preprocessed data...")
    # Read from CSV (replaces Parquet due to Windows Hadoop issues)
    df = spark.read.csv("dataset/processed_churn_data_csv", header=True, inferSchema=True)
    
    print(f"Dataset loaded: {df.count()} rows, {len(df.columns)} columns")
    df.printSchema()
    
    # ============================================================================
    # STEP 3: Prepare Features for ML
    # ============================================================================
    
    print("\nPreparing features...")
    
    categorical_cols = ["Geography", "Gender", "AgeGroup", "BalanceCategory", "CreditScoreCategory"]
    numerical_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", 
                     "HasCrCard", "IsActiveMember", "EstimatedSalary"]
    
    # String Indexing
    indexers = [StringIndexer(inputCol=col, outputCol=col+"_Index", handleInvalid="keep") 
                for col in categorical_cols]
    
    feature_cols = [col+"_Index" for col in categorical_cols] + numerical_cols
    print(f"\nFeature columns: {feature_cols}")
    
    # Vector Assembler
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="assembled_features")
    
    # Standard Scaler
    scaler = StandardScaler(inputCol="assembled_features", outputCol="features", 
                           withStd=True, withMean=True)
    
    # ============================================================================
    # STEP 4: Split Data
    # ============================================================================
    
    print("\nSplitting data...")
    train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)
    
    print(f"Training set size: {train_data.count()}")
    print(f"Testing set size: {test_data.count()}")
    
    print("\n=== Training Set Churn Distribution ===")
    train_data.groupBy("Exited").count().show()
    
    print("\n=== Testing Set Churn Distribution ===")
    test_data.groupBy("Exited").count().show()
    
    # ============================================================================
    # STEP 5: Model 1 - Logistic Regression
    # ============================================================================
    
    print("\n" + "="*80)
    print("MODEL 1: LOGISTIC REGRESSION")
    print("="*80)
    
    lr = LogisticRegression(labelCol="Exited", featuresCol="features", maxIter=100)
    lr_pipeline = Pipeline(stages=indexers + [assembler, scaler, lr])
    
    print("\nTraining Logistic Regression model...")
    lr_model = lr_pipeline.fit(train_data)
    lr_predictions = lr_model.transform(test_data)
    
    # Evaluate
    binary_evaluator = BinaryClassificationEvaluator(labelCol="Exited", metricName="areaUnderROC")
    multiclass_evaluator = MulticlassClassificationEvaluator(labelCol="Exited", predictionCol="prediction")
    
    lr_auc = binary_evaluator.evaluate(lr_predictions)
    lr_accuracy = multiclass_evaluator.evaluate(lr_predictions, {multiclass_evaluator.metricName: "accuracy"})
    lr_precision = multiclass_evaluator.evaluate(lr_predictions, {multiclass_evaluator.metricName: "weightedPrecision"})
    lr_recall = multiclass_evaluator.evaluate(lr_predictions, {multiclass_evaluator.metricName: "weightedRecall"})
    lr_f1 = multiclass_evaluator.evaluate(lr_predictions, {multiclass_evaluator.metricName: "f1"})
    
    print("\n=== Logistic Regression Results ===")
    print(f"AUC-ROC: {lr_auc:.4f}")
    print(f"Accuracy: {lr_accuracy:.4f}")
    print(f"Precision: {lr_precision:.4f}")
    print(f"Recall: {lr_recall:.4f}")
    print(f"F1-Score: {lr_f1:.4f}")
    
    print("\n=== Confusion Matrix ===")
    lr_predictions.groupBy("Exited", "prediction").count().show()
    
    # ============================================================================
    # STEP 6: Model 2 - Random Forest
    # ============================================================================
    
    print("\n" + "="*80)
    print("MODEL 2: RANDOM FOREST CLASSIFIER")
    print("="*80)
    
    rf = RandomForestClassifier(labelCol="Exited", featuresCol="features", 
                                numTrees=100, maxDepth=10, seed=42)
    rf_pipeline = Pipeline(stages=indexers + [assembler, scaler, rf])
    
    print("\nTraining Random Forest model...")
    rf_model = rf_pipeline.fit(train_data)
    rf_predictions = rf_model.transform(test_data)
    
    # Evaluate
    rf_auc = binary_evaluator.evaluate(rf_predictions)
    rf_accuracy = multiclass_evaluator.evaluate(rf_predictions, {multiclass_evaluator.metricName: "accuracy"})
    rf_precision = multiclass_evaluator.evaluate(rf_predictions, {multiclass_evaluator.metricName: "weightedPrecision"})
    rf_recall = multiclass_evaluator.evaluate(rf_predictions, {multiclass_evaluator.metricName: "weightedRecall"})
    rf_f1 = multiclass_evaluator.evaluate(rf_predictions, {multiclass_evaluator.metricName: "f1"})
    
    print("\n=== Random Forest Results ===")
    print(f"AUC-ROC: {rf_auc:.4f}")
    print(f"Accuracy: {rf_accuracy:.4f}")
    print(f"Precision: {rf_precision:.4f}")
    print(f"Recall: {rf_recall:.4f}")
    print(f"F1-Score: {rf_f1:.4f}")
    
    # Feature Importance
    rf_model_stage = rf_model.stages[-1]
    feature_importance = rf_model_stage.featureImportances
    print("\n=== Top Feature Importances ===")
    importance_dict = {feature_cols[i]: float(importance) 
                      for i, importance in enumerate(feature_importance) if importance > 0}
    for feature, importance in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{feature}: {importance:.4f}")
    
    # ============================================================================
    # STEP 7: Model 3 - Gradient Boosted Trees
    # ============================================================================
    
    print("\n" + "="*80)
    print("MODEL 3: GRADIENT BOOSTED TREES")
    print("="*80)
    
    gbt = GBTClassifier(labelCol="Exited", featuresCol="features", 
                        maxIter=100, maxDepth=5, seed=42)
    gbt_pipeline = Pipeline(stages=indexers + [assembler, scaler, gbt])
    
    print("\nTraining Gradient Boosted Trees model...")
    gbt_model = gbt_pipeline.fit(train_data)
    gbt_predictions = gbt_model.transform(test_data)
    
    # Evaluate
    gbt_auc = binary_evaluator.evaluate(gbt_predictions)
    gbt_accuracy = multiclass_evaluator.evaluate(gbt_predictions, {multiclass_evaluator.metricName: "accuracy"})
    gbt_precision = multiclass_evaluator.evaluate(gbt_predictions, {multiclass_evaluator.metricName: "weightedPrecision"})
    gbt_recall = multiclass_evaluator.evaluate(gbt_predictions, {multiclass_evaluator.metricName: "weightedRecall"})
    gbt_f1 = multiclass_evaluator.evaluate(gbt_predictions, {multiclass_evaluator.metricName: "f1"})
    
    print("\n=== Gradient Boosted Trees Results ===")
    print(f"AUC-ROC: {gbt_auc:.4f}")
    print(f"Accuracy: {gbt_accuracy:.4f}")
    print(f"Precision: {gbt_precision:.4f}")
    print(f"Recall: {gbt_recall:.4f}")
    print(f"F1-Score: {gbt_f1:.4f}")
    
    # ============================================================================
    # STEP 8: Model Comparison
    # ============================================================================
    
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    print("\n{:<30} {:<12} {:<12} {:<12} {:<12} {:<12}".format(
        "Model", "AUC-ROC", "Accuracy", "Precision", "Recall", "F1-Score"))
    print("-" * 90)
    print("{:<30} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}".format(
        "Logistic Regression", lr_auc, lr_accuracy, lr_precision, lr_recall, lr_f1))
    print("{:<30} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}".format(
        "Random Forest", rf_auc, rf_accuracy, rf_precision, rf_recall, rf_f1))
    print("{:<30} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}".format(
        "Gradient Boosted Trees", gbt_auc, gbt_accuracy, gbt_precision, gbt_recall, gbt_f1))
    
    # ============================================================================
    # STEP 9: Save Models
    # ============================================================================
    
    print("\n\nSaving models...")
    
    models = {
        "Logistic Regression": (lr_model, lr_auc),
        "Random Forest": (rf_model, rf_auc),
        "Gradient Boosted Trees": (gbt_model, gbt_auc)
    }
    
    best_model_name = max(models, key=lambda x: models[x][1])
    best_model = models[best_model_name][0]
    best_auc = models[best_model_name][1]
    
    print(f"\n=== Best Model: {best_model_name} (AUC-ROC: {best_auc:.4f}) ===")
    
    # Save best model
    best_model.write().overwrite().save("models/best_churn_model")
    print("\nBest model saved to 'models/best_churn_model'")
    
    # Save all models
    lr_model.write().overwrite().save("models/logistic_regression_model")
    rf_model.write().overwrite().save("models/random_forest_model")
    gbt_model.write().overwrite().save("models/gradient_boosted_trees_model")
    print("All models saved successfully!")
    
    # ============================================================================
    # STEP 10: Analyze Predictions
    # ============================================================================
    
    print("\n" + "="*80)
    print("CHURN PREDICTION ANALYSIS")
    print("="*80)
    
    best_predictions = best_model.transform(test_data)
    high_risk = best_predictions.filter(col("prediction") == 1)
    
    print(f"\nTotal high-risk customers (predicted to churn): {high_risk.count()}")
    
    print("\n=== High-Risk Customer Characteristics ===")
    print("\nBy Geography:")
    high_risk.groupBy("Geography").count().orderBy(col("count").desc()).show()
    
    print("\nBy Age Group:")
    high_risk.groupBy("AgeGroup").count().orderBy(col("count").desc()).show()
    
    print("\nBy Active Membership:")
    high_risk.groupBy("IsActiveMember").count().show()
    
    print("\n=== Model Training Complete! ===")
    
    spark.stop()

if __name__ == "__main__":
    main()

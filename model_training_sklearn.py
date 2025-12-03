#!/usr/bin/env python3
"""
Scikit-learn Model Training Script for Bank Customer Churn Prediction
This script trains multiple classification models using scikit-learn.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle
import os

def main():
    print("Loading preprocessed data...")
    df = pd.read_csv("dataset/processed_churn_data.csv")
    
    print(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
    print(df.head())
    
    # Prepare features for ML
    print("\nPreparing features...")
    
    # Label encode categorical features
    label_encoders = {}
    categorical_cols = ['Geography', 'Gender', 'AgeGroup', 'BalanceCategory', 'CreditScoreCategory']
    
    df_encoded = df.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # Define feature and target columns
    numerical_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", 
                     "HasCrCard", "IsActiveMember", "EstimatedSalary"]
    
    # All features in consistent order
    all_features = categorical_cols + numerical_cols
    
    X = df_encoded[all_features]
    y = df_encoded['Exited']
    
    print(f"\nFeature columns: {X.columns.tolist()}")
    print(f"Number of features: {len(X.columns)}")
    
    # Split data
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    print(f"\nTraining set churn rate: {y_train.mean():.2%}")
    print(f"Testing set churn rate: {y_test.mean():.2%}")
    
    # Store results
    results = {}
    
    # ============================================================================
    # MODEL 1: LOGISTIC REGRESSION
    # ============================================================================
    
    print("\n" + "="*80)
    print("MODEL 1: LOGISTIC REGRESSION")
    print("="*80)
    
    print("\nTraining Logistic Regression model...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_pred_proba = lr_model.predict_proba(X_test)[:, 1]
    
    lr_accuracy = accuracy_score(y_test, lr_pred)
    lr_precision = precision_score(y_test, lr_pred)
    lr_recall = recall_score(y_test, lr_pred)
    lr_f1 = f1_score(y_test, lr_pred)
    lr_auc = roc_auc_score(y_test, lr_pred_proba)
    
    print("\n=== Logistic Regression Results ===")
    print(f"AUC-ROC: {lr_auc:.4f}")
    print(f"Accuracy: {lr_accuracy:.4f}")
    print(f"Precision: {lr_precision:.4f}")
    print(f"Recall: {lr_recall:.4f}")
    print(f"F1-Score: {lr_f1:.4f}")
    
    results['Logistic Regression'] = {
        'model': lr_model,
        'auc': lr_auc,
        'accuracy': lr_accuracy,
        'precision': lr_precision,
        'recall': lr_recall,
        'f1': lr_f1
    }
    
    # ============================================================================
    # MODEL 2: RANDOM FOREST
    # ============================================================================
    
    print("\n" + "="*80)
    print("MODEL 2: RANDOM FOREST CLASSIFIER")
    print("="*80)
    
    print("\nTraining Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_precision = precision_score(y_test, rf_pred)
    rf_recall = recall_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred)
    rf_auc = roc_auc_score(y_test, rf_pred_proba)
    
    print("\n=== Random Forest Results ===")
    print(f"AUC-ROC: {rf_auc:.4f}")
    print(f"Accuracy: {rf_accuracy:.4f}")
    print(f"Precision: {rf_precision:.4f}")
    print(f"Recall: {rf_recall:.4f}")
    print(f"F1-Score: {rf_f1:.4f}")
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== Top 10 Feature Importances ===")
    print(feature_importance.head(10).to_string(index=False))
    
    results['Random Forest'] = {
        'model': rf_model,
        'auc': rf_auc,
        'accuracy': rf_accuracy,
        'precision': rf_precision,
        'recall': rf_recall,
        'f1': rf_f1
    }
    
    # ============================================================================
    # MODEL 3: GRADIENT BOOSTING
    # ============================================================================
    
    print("\n" + "="*80)
    print("MODEL 3: GRADIENT BOOSTING CLASSIFIER")
    print("="*80)
    
    print("\nTraining Gradient Boosting model...")
    gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    gb_pred_proba = gb_model.predict_proba(X_test)[:, 1]
    
    gb_accuracy = accuracy_score(y_test, gb_pred)
    gb_precision = precision_score(y_test, gb_pred)
    gb_recall = recall_score(y_test, gb_pred)
    gb_f1 = f1_score(y_test, gb_pred)
    gb_auc = roc_auc_score(y_test, gb_pred_proba)
    
    print("\n=== Gradient Boosting Results ===")
    print(f"AUC-ROC: {gb_auc:.4f}")
    print(f"Accuracy: {gb_accuracy:.4f}")
    print(f"Precision: {gb_precision:.4f}")
    print(f"Recall: {gb_recall:.4f}")
    print(f"F1-Score: {gb_f1:.4f}")
    
    results['Gradient Boosting'] = {
        'model': gb_model,
        'auc': gb_auc,
        'accuracy': gb_accuracy,
        'precision': gb_precision,
        'recall': gb_recall,
        'f1': gb_f1
    }
    
    # ============================================================================
    # MODEL COMPARISON
    # ============================================================================
    
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    print("\n{:<30} {:<12} {:<12} {:<12} {:<12} {:<12}".format(
        "Model", "AUC-ROC", "Accuracy", "Precision", "Recall", "F1-Score"))
    print("-" * 90)
    
    for model_name, metrics in results.items():
        print("{:<30} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}".format(
            model_name, 
            metrics['auc'], 
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1']
        ))
    
    # ============================================================================
    # SAVE MODELS
    # ============================================================================
    
    print("\n\nSaving models...")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['auc'])
    best_model = results[best_model_name]['model']
    best_auc = results[best_model_name]['auc']
    
    print(f"\n=== Best Model: {best_model_name} (AUC-ROC: {best_auc:.4f}) ===")
    
    # Save best model
    with open('models/best_churn_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print("Best model saved to 'models/best_churn_model.pkl'")
    
    # Save all models
    with open('models/logistic_regression_model.pkl', 'wb') as f:
        pickle.dump(results['Logistic Regression']['model'], f)
    
    with open('models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(results['Random Forest']['model'], f)
    
    with open('models/gradient_boosting_model.pkl', 'wb') as f:
        pickle.dump(results['Gradient Boosting']['model'], f)
    
    # Save label encoders
    with open('models/label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    
    # Save feature order
    with open('models/feature_columns.pkl', 'wb') as f:
        pickle.dump(all_features, f)
    
    print("All models saved successfully!")
    
    # ============================================================================
    # PREDICTION ANALYSIS
    # ============================================================================
    
    print("\n" + "="*80)
    print("CHURN PREDICTION ANALYSIS")
    print("="*80)
    
    best_predictions = best_model.predict(X_test)
    high_risk_indices = X_test[best_predictions == 1].index
    
    print(f"\nTotal high-risk customers (predicted to churn): {len(high_risk_indices)}")
    print(f"High-risk rate: {len(high_risk_indices) / len(X_test) * 100:.2f}%")
    
    print("\n=== Model Training Complete! ===")

if __name__ == "__main__":
    main()

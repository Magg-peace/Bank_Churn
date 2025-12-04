#!/usr/bin/env python3
"""
Visualization Module for Bank Customer Churn Prediction
Generates ROC curves, feature importance, histograms, scatter plots, and model comparisons
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

def load_data_and_models():
    """Load preprocessed data and trained models"""
    print("Loading data and models...")
    
    # Load data
    df = pd.read_csv("dataset/processed_churn_data.csv")
    
    # Load encoders and feature columns
    with open('models/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    with open('models/feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    
    # Prepare features
    categorical_cols = ['Geography', 'Gender', 'AgeGroup', 'BalanceCategory', 'CreditScoreCategory']
    numerical_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", 
                     "HasCrCard", "IsActiveMember", "EstimatedSalary"]
    
    df_encoded = df.copy()
    for col in categorical_cols:
        df_encoded[col] = label_encoders[col].transform(df[col].astype(str))
    
    X = df_encoded[feature_columns]
    y = df_encoded['Exited']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Load models
    with open('models/best_churn_model.pkl', 'rb') as f:
        best_model = pickle.load(f)
    with open('models/logistic_regression_model.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    with open('models/random_forest_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    with open('models/gradient_boosting_model.pkl', 'rb') as f:
        gb_model = pickle.load(f)
    
    return {
        'df': df,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_columns': feature_columns,
        'models': {
            'Logistic Regression': lr_model,
            'Random Forest': rf_model,
            'Gradient Boosting': gb_model
        },
        'label_encoders': label_encoders
    }

def plot_roc_curves(data):
    """Plot ROC curves for all models"""
    print("Generating ROC Curves...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('ROC Curves - Model Comparison', fontsize=16, fontweight='bold')
    
    X_test = data['X_test']
    y_test = data['y_test']
    models = data['models']
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # Plot individual ROC curves
    axes_flat = axes.flatten()
    for idx, (model_name, model) in enumerate(models.items()):
        ax = axes_flat[idx]
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=colors[idx], lw=2.5, label=f'{model_name} (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        ax.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Plot all ROC curves together
    ax = axes_flat[3]
    for idx, (model_name, model) in enumerate(models.items()):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[idx], lw=2.5, label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    ax.set_title('All Models - Combined ROC Curves', fontsize=12, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/roc_curves.png', dpi=300, bbox_inches='tight')
    print("✓ ROC Curves saved: visualizations/roc_curves.png")
    plt.close()

def plot_model_comparison_bars(data):
    """Plot bar graphs comparing model performance"""
    print("Generating Model Comparison Bar Charts...")
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    X_test = data['X_test']
    y_test = data['y_test']
    models = data['models']
    
    metrics_data = {
        'Model': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': [],
        'AUC-ROC': []
    }
    
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics_data['Model'].append(model_name)
        metrics_data['Accuracy'].append(accuracy_score(y_test, y_pred))
        metrics_data['Precision'].append(precision_score(y_test, y_pred))
        metrics_data['Recall'].append(recall_score(y_test, y_pred))
        metrics_data['F1-Score'].append(f1_score(y_test, y_pred))
        metrics_data['AUC-ROC'].append(roc_auc_score(y_test, y_pred_proba))
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Create comparison bar charts
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for idx, metric in enumerate(metrics):
        ax = axes.flatten()[idx]
        bars = ax.bar(df_metrics['Model'], df_metrics[metric], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_ylabel(metric, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Rotate x labels
        ax.set_xticklabels(df_metrics['Model'], rotation=45, ha='right')
    
    # Remove extra subplot
    axes.flatten()[5].remove()
    
    plt.tight_layout()
    plt.savefig('visualizations/model_comparison_bars.png', dpi=300, bbox_inches='tight')
    print("✓ Model Comparison Bar Charts saved: visualizations/model_comparison_bars.png")
    plt.close()
    
    return df_metrics

def plot_confusion_matrices(data):
    """Plot confusion matrices for all models"""
    print("Generating Confusion Matrices...")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold')
    
    X_test = data['X_test']
    y_test = data['y_test']
    models = data['models']
    
    for idx, (model_name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   cbar_kws={'label': 'Count'}, annot_kws={'size': 14, 'weight': 'bold'})
        axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Actual', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Predicted', fontsize=11, fontweight='bold')
        axes[idx].set_xticklabels(['No Churn', 'Churn'])
        axes[idx].set_yticklabels(['No Churn', 'Churn'])
    
    plt.tight_layout()
    plt.savefig('visualizations/confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("✓ Confusion Matrices saved: visualizations/confusion_matrices.png")
    plt.close()

def plot_feature_importance(data):
    """Plot feature importance from Random Forest model"""
    print("Generating Feature Importance Plot...")
    
    X_test = data['X_test']
    rf_model = data['models']['Random Forest']
    feature_columns = data['feature_columns']
    
    # Get feature importances
    importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(12)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars = ax.barh(range(len(feature_importance_df)), feature_importance_df['Importance'], 
                   color='#45B7D1', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_yticks(range(len(feature_importance_df)))
    ax.set_yticklabels(feature_importance_df['Feature'])
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title('Top 12 Feature Importance - Random Forest Model', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f'{width:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Feature Importance saved: visualizations/feature_importance.png")
    plt.close()

def plot_churn_distribution(data):
    """Plot churn distribution histograms"""
    print("Generating Churn Distribution Plots...")
    
    df = data['df']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Churn Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Overall churn distribution
    ax = axes[0, 0]
    churn_counts = df['Exited'].value_counts()
    colors = ['#4ECDC4', '#FF6B6B']
    wedges, texts, autotexts = ax.pie(churn_counts.values, labels=['No Churn', 'Churn'],
                                       autopct='%1.2f%%', startangle=90, colors=colors,
                                       textprops={'fontsize': 11, 'weight': 'bold'})
    ax.set_title('Overall Churn Distribution', fontsize=12, fontweight='bold')
    
    # Churn by Geography
    ax = axes[0, 1]
    churn_by_geo = pd.crosstab(df['Geography'], df['Exited'])
    churn_by_geo.plot(kind='bar', ax=ax, color=['#4ECDC4', '#FF6B6B'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_title('Churn by Geography', fontsize=12, fontweight='bold')
    ax.set_xlabel('Geography', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.legend(['No Churn', 'Churn'], loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Churn by Gender
    ax = axes[1, 0]
    churn_by_gender = pd.crosstab(df['Gender'], df['Exited'])
    churn_by_gender.plot(kind='bar', ax=ax, color=['#4ECDC4', '#FF6B6B'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_title('Churn by Gender', fontsize=12, fontweight='bold')
    ax.set_xlabel('Gender', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.legend(['No Churn', 'Churn'], loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    # Churn by Active Member
    ax = axes[1, 1]
    churn_by_active = pd.crosstab(df['IsActiveMember'], df['Exited'])
    churn_by_active.plot(kind='bar', ax=ax, color=['#4ECDC4', '#FF6B6B'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_title('Churn by Active Membership', fontsize=12, fontweight='bold')
    ax.set_xlabel('Active Member (1=Yes, 0=No)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.legend(['No Churn', 'Churn'], loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticklabels(['Inactive', 'Active'], rotation=0)
    
    plt.tight_layout()
    plt.savefig('visualizations/churn_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Churn Distribution saved: visualizations/churn_distribution.png")
    plt.close()

def plot_numerical_features_distribution(data):
    """Plot histograms and scatter plots for numerical features"""
    print("Generating Numerical Features Distribution Plots...")
    
    df = data['df']
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    fig.suptitle('Numerical Features Distribution by Churn Status', fontsize=16, fontweight='bold')
    
    numerical_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", 
                     "HasCrCard", "IsActiveMember", "EstimatedSalary"]
    
    for idx, col in enumerate(numerical_cols):
        ax = axes.flatten()[idx]
        
        no_churn = df[df['Exited'] == 0][col]
        churn = df[df['Exited'] == 1][col]
        
        ax.hist(no_churn, bins=30, alpha=0.6, label='No Churn', color='#4ECDC4', edgecolor='black', linewidth=0.5)
        ax.hist(churn, bins=30, alpha=0.6, label='Churn', color='#FF6B6B', edgecolor='black', linewidth=0.5)
        
        ax.set_title(f'{col} Distribution', fontsize=11, fontweight='bold')
        ax.set_xlabel(col, fontsize=10, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('visualizations/numerical_features_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Numerical Features Distribution saved: visualizations/numerical_features_distribution.png")
    plt.close()

def plot_scatter_plots(data):
    """Plot scatter plots for key feature relationships"""
    print("Generating Scatter Plots...")
    
    df = data['df']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Key Feature Relationships with Churn', fontsize=16, fontweight='bold')
    
    # Age vs Balance colored by Churn
    ax = axes[0, 0]
    for churn in [0, 1]:
        mask = df['Exited'] == churn
        ax.scatter(df[mask]['Age'], df[mask]['Balance'], 
                  label='Churn' if churn == 1 else 'No Churn',
                  alpha=0.5, s=30, c='#FF6B6B' if churn == 1 else '#4ECDC4', edgecolor='black', linewidth=0.3)
    ax.set_xlabel('Age', fontsize=11, fontweight='bold')
    ax.set_ylabel('Balance', fontsize=11, fontweight='bold')
    ax.set_title('Age vs Balance', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Tenure vs Balance
    ax = axes[0, 1]
    for churn in [0, 1]:
        mask = df['Exited'] == churn
        ax.scatter(df[mask]['Tenure'], df[mask]['Balance'],
                  label='Churn' if churn == 1 else 'No Churn',
                  alpha=0.5, s=30, c='#FF6B6B' if churn == 1 else '#4ECDC4', edgecolor='black', linewidth=0.3)
    ax.set_xlabel('Tenure (years)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Balance', fontsize=11, fontweight='bold')
    ax.set_title('Tenure vs Balance', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Age vs CreditScore
    ax = axes[0, 2]
    for churn in [0, 1]:
        mask = df['Exited'] == churn
        ax.scatter(df[mask]['Age'], df[mask]['CreditScore'],
                  label='Churn' if churn == 1 else 'No Churn',
                  alpha=0.5, s=30, c='#FF6B6B' if churn == 1 else '#4ECDC4', edgecolor='black', linewidth=0.3)
    ax.set_xlabel('Age', fontsize=11, fontweight='bold')
    ax.set_ylabel('Credit Score', fontsize=11, fontweight='bold')
    ax.set_title('Age vs Credit Score', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # NumOfProducts vs Balance
    ax = axes[1, 0]
    for churn in [0, 1]:
        mask = df['Exited'] == churn
        ax.scatter(df[mask]['NumOfProducts'], df[mask]['Balance'],
                  label='Churn' if churn == 1 else 'No Churn',
                  alpha=0.5, s=30, c='#FF6B6B' if churn == 1 else '#4ECDC4', edgecolor='black', linewidth=0.3)
    ax.set_xlabel('Number of Products', fontsize=11, fontweight='bold')
    ax.set_ylabel('Balance', fontsize=11, fontweight='bold')
    ax.set_title('Number of Products vs Balance', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # EstimatedSalary vs Balance
    ax = axes[1, 1]
    for churn in [0, 1]:
        mask = df['Exited'] == churn
        ax.scatter(df[mask]['EstimatedSalary'], df[mask]['Balance'],
                  label='Churn' if churn == 1 else 'No Churn',
                  alpha=0.5, s=30, c='#FF6B6B' if churn == 1 else '#4ECDC4', edgecolor='black', linewidth=0.3)
    ax.set_xlabel('Estimated Salary', fontsize=11, fontweight='bold')
    ax.set_ylabel('Balance', fontsize=11, fontweight='bold')
    ax.set_title('Salary vs Balance', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Age vs Tenure
    ax = axes[1, 2]
    for churn in [0, 1]:
        mask = df['Exited'] == churn
        ax.scatter(df[mask]['Age'], df[mask]['Tenure'],
                  label='Churn' if churn == 1 else 'No Churn',
                  alpha=0.5, s=30, c='#FF6B6B' if churn == 1 else '#4ECDC4', edgecolor='black', linewidth=0.3)
    ax.set_xlabel('Age', fontsize=11, fontweight='bold')
    ax.set_ylabel('Tenure (years)', fontsize=11, fontweight='bold')
    ax.set_title('Age vs Tenure', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/scatter_plots.png', dpi=300, bbox_inches='tight')
    print("✓ Scatter Plots saved: visualizations/scatter_plots.png")
    plt.close()

def plot_model_predictions_comparison(data):
    """Plot prediction distributions from all models"""
    print("Generating Model Predictions Comparison...")
    
    X_test = data['X_test']
    y_test = data['y_test']
    models = data['models']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Model Predictions Distribution Comparison', fontsize=16, fontweight='bold')
    
    # Probability distributions
    ax = axes[0, 0]
    for model_name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        ax.hist(y_pred_proba, bins=50, alpha=0.6, label=model_name, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Predicted Probability of Churn', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('Churn Probability Distributions', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Actual vs Predicted for best model
    ax = axes[0, 1]
    best_model = data['models']['Random Forest']
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    no_churn_proba = y_pred_proba[y_test == 0]
    churn_proba = y_pred_proba[y_test == 1]
    
    ax.hist(no_churn_proba, bins=50, alpha=0.6, label='Actual: No Churn', color='#4ECDC4', edgecolor='black', linewidth=0.5)
    ax.hist(churn_proba, bins=50, alpha=0.6, label='Actual: Churn', color='#FF6B6B', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Predicted Probability of Churn', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('Random Forest - Actual vs Predicted', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Box plots for all models
    ax = axes[1, 0]
    proba_data = []
    model_names = []
    for model_name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        proba_data.append(y_pred_proba)
        model_names.append(model_name)
    
    bp = ax.boxplot(proba_data, labels=model_names, patch_artist=True)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Predicted Probability', fontsize=11, fontweight='bold')
    ax.set_title('Prediction Probability Box Plots', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Calibration curve
    ax = axes[1, 1]
    from sklearn.calibration import calibration_curve
    
    for model_name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
        ax.plot(prob_pred, prob_true, marker='o', linewidth=2, label=model_name, markersize=8)
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfectly Calibrated')
    ax.set_xlabel('Mean Predicted Probability', fontsize=11, fontweight='bold')
    ax.set_ylabel('Fraction of Positives', fontsize=11, fontweight='bold')
    ax.set_title('Calibration Curves', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/model_predictions_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Model Predictions Comparison saved: visualizations/model_predictions_comparison.png")
    plt.close()

def main():
    """Generate all visualizations"""
    
    # Create visualizations directory
    os.makedirs('visualizations', exist_ok=True)
    
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS FOR BANK CHURN PREDICTION")
    print("="*80 + "\n")
    
    # Load data and models
    data = load_data_and_models()
    
    # Generate all visualizations
    plot_roc_curves(data)
    plot_model_comparison_bars(data)
    plot_confusion_matrices(data)
    plot_feature_importance(data)
    plot_churn_distribution(data)
    plot_numerical_features_distribution(data)
    plot_scatter_plots(data)
    plot_model_predictions_comparison(data)
    
    print("\n" + "="*80)
    print("✓ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("="*80)
    print("\nVisualization files saved in 'visualizations/' directory:")
    print("1. roc_curves.png - ROC curves for all models")
    print("2. model_comparison_bars.png - Bar charts comparing model metrics")
    print("3. confusion_matrices.png - Confusion matrices for all models")
    print("4. feature_importance.png - Top 12 important features")
    print("5. churn_distribution.png - Churn distribution analysis")
    print("6. numerical_features_distribution.png - Histograms of numerical features")
    print("7. scatter_plots.png - Scatter plots of feature relationships")
    print("8. model_predictions_comparison.png - Model predictions comparison")
    print("\n")

if __name__ == "__main__":
    main()

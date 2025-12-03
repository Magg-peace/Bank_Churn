#!/usr/bin/env python3
"""
Generate 1 Lakh (100,000) Records Bank Customer Churn Dataset with proper churn patterns
"""

import pandas as pd
import numpy as np
import os

def generate_large_dataset(num_records=100000):
    """Generate a large dataset with 1 lakh records with realistic churn patterns"""
    
    print(f"Generating {num_records:,} customer records with realistic patterns...")
    
    np.random.seed(42)
    
    # Generate features
    geography = np.random.choice(['France', 'Germany', 'Spain'], num_records, p=[0.5, 0.25, 0.25])
    gender = np.random.choice(['Male', 'Female'], num_records, p=[0.545, 0.455])
    credit_score = np.random.normal(660, 100, num_records).astype(int)
    credit_score = np.clip(credit_score, 350, 850)
    age = np.random.normal(40, 15, num_records).astype(int)
    age = np.clip(age, 18, 92)
    tenure = np.random.randint(0, 11, num_records)
    balance = np.random.exponential(50000, num_records)
    balance = np.clip(balance, 0, 250000)
    num_products = np.random.choice([1, 2, 3, 4], num_records, p=[0.5, 0.46, 0.03, 0.01])
    has_cr_card = np.random.choice([0, 1], num_records, p=[0.30, 0.70])
    is_active = np.random.choice([0, 1], num_records, p=[0.485, 0.515])
    estimated_salary = np.random.uniform(11.58, 199992.48, num_records)
    
    # Generate churn based on realistic patterns
    churn = np.zeros(num_records, dtype=int)
    
    # Base churn rate: 20%
    base_churn_mask = np.random.rand(num_records) < 0.20
    
    # Factor 1: Age (older customers churn more)
    age_factor = (age - age.min()) / (age.max() - age.min())
    churn_prob = 0.15 + (age_factor * 0.15)  # 15-30%
    
    # Factor 2: Tenure (longer tenure = less churn)
    tenure_factor = (tenure.max() - tenure) / (tenure.max() - tenure.min())
    churn_prob += (tenure_factor * 0.10)  # Add up to 10%
    
    # Factor 3: Active membership (inactive members churn more)
    is_inactive = is_active == 0
    churn_prob[is_inactive] += 0.15
    
    # Factor 4: Number of products (few products = more churn)
    few_products = num_products <= 2
    churn_prob[few_products] += 0.10
    
    # Factor 5: Balance (very low balance = more churn)
    zero_balance = balance == 0
    churn_prob[zero_balance] += 0.10
    
    # Factor 6: Geography (Germany has higher churn)
    germany = geography == 'Germany'
    churn_prob[germany] += 0.05
    
    # Clip probability to [0, 1]
    churn_prob = np.clip(churn_prob, 0, 1)
    
    # Generate churn based on probability
    churn = (np.random.rand(num_records) < churn_prob).astype(int)
    
    # Create DataFrame
    data = {
        'RowNumber': np.arange(1, num_records + 1),
        'CustomerId': np.random.randint(15000000, 16000000, num_records),
        'Surname': np.random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 
                                     'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez',
                                     'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson',
                                     'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin'],
                                    num_records),
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active,
        'EstimatedSalary': estimated_salary,
        'Exited': churn
    }
    
    df = pd.DataFrame(data)
    
    print(f"\nDataset generated successfully!")
    print(f"Total records: {len(df):,}")
    print(f"Churn rate: {df['Exited'].mean():.2%}")
    
    return df

def main():
    # Create dataset directory
    os.makedirs("dataset", exist_ok=True)
    
    # Generate 1 lakh records
    df = generate_large_dataset(100000)
    
    print("\n=== Dataset Preview ===")
    print(df.head(10))
    
    print("\n=== Dataset Statistics ===")
    print(df.describe())
    
    print("\n=== Churn Distribution ===")
    print(df['Exited'].value_counts())
    print(f"Churn Rate: {df['Exited'].mean():.2%}")
    
    # Save the raw data
    output_path = "dataset/Churn_Modelling_100K.csv"
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")
    
    # Also save as the main dataset
    df.to_csv("dataset/Churn_Modelling.csv", index=False)
    print("Also saved as: dataset/Churn_Modelling.csv")

if __name__ == "__main__":
    main()

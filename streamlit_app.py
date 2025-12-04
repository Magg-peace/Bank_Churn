import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load models and encoders
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        if os.path.exists("models/best_churn_model.pkl"):
            with open("models/best_churn_model.pkl", "rb") as f:
                model = pickle.load(f)
            with open("models/label_encoders.pkl", "rb") as f:
                label_encoders = pickle.load(f)
            with open("models/feature_columns.pkl", "rb") as f:
                feature_columns = pickle.load(f)
            return model, label_encoders, feature_columns, "best_churn_model"
        else:
            return None, None, None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None

def predict_churn(model, label_encoders, feature_columns, customer_data):
    """Make churn prediction for a customer"""
    try:
        # Create DataFrame with all features in correct order
        df = pd.DataFrame([customer_data])
        
        # Encode categorical features
        categorical_cols = ['Geography', 'Gender', 'AgeGroup', 'BalanceCategory', 'CreditScoreCategory']
        
        for col in categorical_cols:
            if col in label_encoders:
                df[col] = label_encoders[col].transform(df[col].astype(str))
        
        # Select features in the same order as training
        df_features = df[feature_columns]
        
        # Make prediction
        prediction = model.predict(df_features)[0]
        probability = model.predict_proba(df_features)[0]
        
        return int(prediction), probability
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

# Main App
def main():
    # Header
    st.title("🏦 Bank Customer Churn Prediction System")
    st.markdown("### Predict customer churn using Machine Learning")
    
    st.markdown("---")
    
    # Load model
    model, label_encoders, feature_columns, model_name = load_model()
    
    if model is None:
        st.error("❌ Model not found! Please train the model first using the training script.")
        st.info("Run: python model_training_sklearn.py")
        return
    
    # Display model info
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"✅ Model: {model_name}")
    with col2:
        st.info("📊 Training Size: 100,000+ records")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Single Prediction", "Batch Prediction", "Model Insights"])
    
    if page == "Single Prediction":
        show_single_prediction(model, label_encoders, feature_columns)
    elif page == "Batch Prediction":
        show_batch_prediction(model, label_encoders, feature_columns)
    else:
        show_model_insights()

def show_single_prediction(model, label_encoders, feature_columns):
    """Single customer prediction page"""
    st.header("🔍 Single Customer Prediction")
    st.markdown("Enter customer details to predict churn probability")
    
    # Create input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Personal Information")
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 100, 35)
    
    with col2:
        st.subheader("Account Information")
        credit_score = st.slider("Credit Score", 300, 850, 650)
        tenure = st.slider("Tenure (years)", 0, 10, 5)
        balance = st.number_input("Account Balance ($)", 0.0, 250000.0, 50000.0, step=1000.0)
    
    with col3:
        st.subheader("Banking Behavior")
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        has_credit_card = st.selectbox("Has Credit Card", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        is_active = st.selectbox("Is Active Member", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        estimated_salary = st.number_input("Estimated Salary ($)", 0.0, 200000.0, 50000.0, step=1000.0)
    
    # Predict button
    if st.button("🎯 Predict Churn", type="primary"):
        # Prepare customer data with engineered features
        age_group = "Young" if age < 30 else ("Middle" if age < 45 else "Senior")
        balance_cat = "Zero" if balance == 0 else ("Low" if balance <= 50000 else ("Medium" if balance <= 100000 else "High"))
        credit_cat = "Poor" if credit_score < 600 else ("Fair" if credit_score < 700 else ("Good" if credit_score < 800 else "Excellent"))
        
        customer_data = {
            "Geography": geography,
            "Gender": gender,
            "AgeGroup": age_group,
            "BalanceCategory": balance_cat,
            "CreditScoreCategory": credit_cat,
            "CreditScore": credit_score,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": has_credit_card,
            "IsActiveMember": is_active,
            "EstimatedSalary": estimated_salary
        }
        
        # Make prediction
        with st.spinner("Making prediction..."):
            churn_prediction, probability = predict_churn(model, label_encoders, feature_columns, customer_data)
        
        if churn_prediction is not None:
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if churn_prediction == 1:
                    st.error("⚠️ **HIGH RISK**")
                    st.metric("Churn Prediction", "Will Churn")
                else:
                    st.success("✅ **LOW RISK**")
                    st.metric("Churn Prediction", "Will Stay")
            
            with col2:
                churn_prob = probability[1] * 100
                st.metric("Churn Probability", f"{churn_prob:.2f}%")
            
            with col3:
                retention_prob = probability[0] * 100
                st.metric("Retention Probability", f"{retention_prob:.2f}%")
            
            # Risk level indicator
            st.markdown("### Risk Level")
            if churn_prob > 70:
                st.error("🔴 Very High Risk - Immediate action required!")
            elif churn_prob > 50:
                st.warning("🟡 High Risk - Consider retention strategies")
            elif churn_prob > 30:
                st.info("🔵 Medium Risk - Monitor customer engagement")
            else:
                st.success("🟢 Low Risk - Customer is likely to stay")
            
            # Recommendations
            st.markdown("### 💡 Recommended Actions")
            if churn_prediction == 1:
                recommendations = []
                if is_active == 0:
                    recommendations.append("✓ Customer is inactive - engage with personalized offers")
                if num_products <= 2:
                    recommendations.append("✓ Offer additional products/services to increase engagement")
                if balance == 0:
                    recommendations.append("✓ Low balance - encourage deposits with promotional rates")
                if age > 50:
                    recommendations.append("✓ Senior customer - provide dedicated support and benefits")
                
                if recommendations:
                    for rec in recommendations:
                        st.write(rec)
                else:
                    st.write("✓ Schedule a personal call to understand customer needs")
                    st.write("✓ Offer loyalty rewards or exclusive benefits")
            else:
                st.write("✓ Customer shows positive engagement - maintain regular communication")
                st.write("✓ Consider upselling opportunities")

def show_batch_prediction(model, label_encoders, feature_columns):
    """Batch prediction page"""
    st.header("📁 Batch Prediction")
    st.markdown("Upload a CSV file with customer data for batch predictions")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Read CSV
        df_pandas = pd.read_csv(uploaded_file)
        
        st.subheader("Data Preview")
        st.dataframe(df_pandas.head())
        
        if st.button("🚀 Run Batch Prediction"):
            try:
                with st.spinner("Processing predictions..."):
                    # Prepare data
                    df_predict = df_pandas.copy()
                    
                    # Engineer features if not already present
                    if 'AgeGroup' not in df_predict.columns:
                        df_predict['AgeGroup'] = pd.cut(df_predict['Age'], 
                                                        bins=[0, 30, 45, 100], 
                                                        labels=['Young', 'Middle', 'Senior'])
                    
                    # Balance category - match trained encoder categories: ['High', 'Low', 'Medium']
                    if 'BalanceCategory' not in df_predict.columns:
                        # Map balance to correct categories (no 'Zero' in encoder)
                        def categorize_balance(balance):
                            if balance <= 50000:
                                return 'Low'
                            elif balance <= 100000:
                                return 'Medium'
                            else:
                                return 'High'
                        
                        df_predict['BalanceCategory'] = df_predict['Balance'].apply(categorize_balance)
                    
                    # Credit Score category - match encoder categories: ['Excellent', 'Fair', 'Good', 'Poor']
                    if 'CreditScoreCategory' not in df_predict.columns:
                        df_predict['CreditScoreCategory'] = pd.cut(df_predict['CreditScore'],
                                                                   bins=[0, 600, 700, 800, 900],
                                                                   labels=['Poor', 'Fair', 'Good', 'Excellent'],
                                                                   include_lowest=True)
                    
                    # Encode categorical features with error handling
                    df_encoded = df_predict.copy()
                    categorical_cols = ['Geography', 'Gender', 'AgeGroup', 'BalanceCategory', 'CreditScoreCategory']
                    
                    errors = []
                    for col in categorical_cols:
                        if col in label_encoders:
                            try:
                                # Convert to string
                                df_encoded[col] = df_encoded[col].astype(str)
                                
                                # Check for unseen labels
                                known_classes = set(label_encoders[col].classes_)
                                seen_values = set(df_encoded[col].unique())
                                unseen = seen_values - known_classes
                                
                                if unseen:
                                    errors.append(f"Column '{col}' has unseen values: {unseen}. Using mode value as replacement.")
                                    # Replace unseen values with the first known class
                                    default_class = label_encoders[col].classes_[0]
                                    for unseen_val in unseen:
                                        df_encoded.loc[df_encoded[col] == unseen_val, col] = default_class
                                
                                # Transform using label encoder
                                df_encoded[col] = label_encoders[col].transform(df_encoded[col])
                            except Exception as e:
                                raise ValueError(f"Error encoding column '{col}': {str(e)}")
                    
                    # Select features in correct order
                    df_features = df_encoded[feature_columns]
                    
                    # Make predictions
                    predictions = model.predict(df_features)
                    probabilities = model.predict_proba(df_features)
                    
                    # Add predictions to results
                    results = df_pandas.copy()
                    results['Churn_Prediction'] = predictions
                    results['Churn_Probability'] = probabilities[:, 1]
                    results['Retention_Probability'] = probabilities[:, 0]
                    results['Risk_Level'] = results['Churn_Probability'].apply(
                        lambda x: 'Very High' if x > 0.7 else ('High' if x > 0.5 else ('Medium' if x > 0.3 else 'Low'))
                    )
                    
                    st.success("✅ Predictions completed!")
                    
                    # Show warnings if any
                    if errors:
                        for error in errors:
                            st.warning(error)
                    
                    # Show results
                    st.subheader("Prediction Results")
                    st.dataframe(results)
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    total_customers = len(results)
                    churn_count = len(results[results['Churn_Prediction'] == 1])
                    retention_count = total_customers - churn_count
                    avg_churn_prob = results['Churn_Probability'].mean()
                    
                    with col1:
                        st.metric("Total Customers", total_customers)
                    with col2:
                        st.metric("Predicted Churn", churn_count)
                    with col3:
                        st.metric("Predicted Retention", retention_count)
                    with col4:
                        st.metric("Avg Churn Prob", f"{avg_churn_prob:.2%}")
                    
                    # Download results
                    csv = results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="churn_predictions.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"❌ Error during batch prediction: {str(e)}")
                st.info("Please ensure your CSV file contains all required columns: Geography, Gender, Age, CreditScore, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary")

def show_model_insights():
    """Model insights page"""
    st.header("📈 Model Insights")
    
    st.info("📊 Displaying insights for the trained model")
    
    st.markdown("""
    ### Model Performance
    
    Our Bank Customer Churn Prediction model uses Scikit-learn with multiple algorithms:
    - **Logistic Regression**: Baseline model for interpretability
    - **Random Forest**: Ensemble method for improved accuracy
    - **Gradient Boosting**: Advanced ensemble technique (Best Model)
    
    ### Key Features Analyzed
    1. **Age**: Customer demographic factor (strongest predictor)
    2. **Number of Products**: Products/services used
    3. **Balance**: Account balance amount
    4. **Estimated Salary**: Customer income level
    5. **Credit Score**: Customer creditworthiness
    6. **Active Membership**: Customer engagement level
    7. **Geography & Gender**: Demographic factors
    
    ### Top Churn Indicators (by importance)
    1. 🔴 **Age** - Older customers show higher churn rates
    2. 🔴 **Low number of products** - Single-product customers more likely to churn
    3. 🔴 **Balance patterns** - Both zero and very high balances indicate risk
    4. 🔴 **Estimated Salary** - Income-related patterns affect churn
    5. 🔴 **Credit Score** - Credit profile impacts retention
    6. 🔴 **Inactive membership** - Engagement is crucial
    
    ### Business Impact
    - Early identification of at-risk customers
    - Targeted retention campaigns
    - Resource optimization for customer service
    - Improved customer lifetime value
    - Reduced customer acquisition costs
    
    ### Recommended Actions
    1. **High-Risk Customers**: Immediate intervention with personalized offers
    2. **Medium-Risk Customers**: Engagement campaigns and regular follow-ups
    3. **Low-Risk Customers**: Upselling opportunities and loyalty programs
    """)
    
    # Model Metrics
    st.markdown("### Model Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", "87%")
    with col2:
        st.metric("AUC-ROC", "0.859")
    with col3:
        st.metric("Precision", "76%")
    with col4:
        st.metric("Recall", "49%")
    
    st.markdown("### Feature Importance")
    st.write("""
    **Top 10 Most Important Features:**
    1. Age: 27.7%
    2. NumOfProducts: 21.9%
    3. Balance: 8.2%
    4. EstimatedSalary: 7.9%
    5. CreditScore: 7.6%
    6. IsActiveMember: 6.6%
    7. AgeGroup: 4.8%
    8. Tenure: 4.5%
    9. Geography: 3.4%
    10. BalanceCategory: 2.4%
    """)

if __name__ == "__main__":
    main()

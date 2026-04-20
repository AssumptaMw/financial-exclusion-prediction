import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings

warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Financial Exclusion Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("Financial Exclusion Risk Predictor")
st.markdown("""
This application predicts the likelihood of financial exclusion for individuals in Kenya 
based on socio-demographic, economic, and technology-access indicators.
""")

# Load the trained model
@st.cache_resource
def load_model():
    """Load the pre-trained model"""
    try:
        # Try to load the model - if it doesn't exist, provide instructions
        model = joblib.load("financial_exclusion_model.pkl")
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'financial_exclusion_model.pkl' is in the current directory.")
        st.info("To create the model, run all cells in the Jupyter notebook first.")
        return None

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = load_model()

model = st.session_state.model

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Prediction", "About", "Instructions"])

if page == "Prediction":
    if model is None:
        st.error("Model not loaded. Please ensure the model file exists.")
    else:
        st.header("Make a Prediction")
        
        # Create two columns for input
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Demographic Information")
            age = st.slider("Age (years)", min_value=18, max_value=100, value=35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", 
                                         ["Married", "Single", "Divorced", "Widowed", "Unknown"])
            household_size = st.slider("Household Size", min_value=1, max_value=20, value=4)
            
        with col2:
            st.subheader("Economic & Technology Access")
            education_level = st.selectbox("Education Level", 
                                          ["No formal education", "Primary education", 
                                           "Secondary education", "Tertiary education", "Unknown"])
            income_source = st.selectbox("Income Source", 
                                        ["Farming", "Business", "Employment", 
                                         "Casual labor", "Other", "Unemployed"])
            location_type = st.selectbox("Location Type", ["Urban", "Rural"])
            county = st.selectbox("County", 
                                 ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Kericho", "Garissa", 
                                  "Mandera", "Turkana", "Samburu", "Marsabit", "West Pokot", 
                                  "Uasin Gishu", "Kiambu", "Murang'a", "Nyeri", "Laikipia",
                                  "Embu", "Meru", "Isiolo", "Kitui", "Machakos", "Makueni",
                                  "Kajiado", "Narok", "Bomet", "Elgeyo Marakwet", "Nandi",
                                  "Baringo", "Lamu", "Taita Taveta", "Migori", "Homabay",
                                  "Siana", "Kakamega", "Vihiga", "Bungoma", "Busia", "Nyanza",
                                  "Kisii", "Kilifi", "Tharaka Nithi", "Muranga", "Kwale", "Dungu"])
        
        with col1:
            st.subheader("Digital & Financial Assets")
            id_ownership = st.selectbox("National ID Ownership", 
                                       ["Has ID", "No ID"])
            mobile_phone_ownership = st.selectbox("Mobile Phone Ownership", 
                                                 ["Yes", "No"])
        
        with col2:
            st.subheader("Service Usage & Disability")
            internet_access = st.selectbox("Internet Access", ["Yes", "No"])
            disability_status = st.selectbox("Disability Status", 
                                            ["No disability", "Has disability", "Undisclosed"])
        
        # Prediction button
        if st.button("Predict Financial Exclusion Risk", use_container_width=True):
            # Prepare input data with proper type conversions
            input_data = pd.DataFrame({
                'age': [age],
                'gender': [gender],
                'education_level': [education_level],
                'marital_status': [marital_status],
                'household_size': [household_size],
                'income_source': [income_source],
                'location_type': [location_type],
                'county': [county],
                'id_ownership': [1 if 'Has ID' in id_ownership else 0],  # Convert to binary
                'mobile_phone_ownership': [1 if mobile_phone_ownership == "Yes" else 0],  # Convert to binary
                'internet_access': [1 if internet_access == "Yes" else 0],
                'disability_status': [disability_status]
            })
            
            try:
                # Make prediction
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0]
                
                # Display results
                st.markdown("---")
                st.subheader("Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    status = "EXCLUDED" if prediction == 1 else "INCLUDED"
                    st.metric("Financial Status", status)
                
                with col2:
                    exclusion_prob = probability[1] * 100
                    st.metric("Exclusion Probability", f"{exclusion_prob:.1f}%")
                
                with col3:
                    inclusion_prob = probability[0] * 100
                    st.metric("Inclusion Probability", f"{inclusion_prob:.1f}%")
                
                # Visualize probability
                fig, ax = plt.subplots(figsize=(10, 4))
                categories = ['Financially Included', 'Financially Excluded']
                colors = ['#2ecc71', '#e74c3c']
                probs = probability * 100
                
                bars = ax.barh(categories, probs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
                ax.set_xlabel('Probability (%)', fontsize=12, fontweight='bold')
                ax.set_xlim(0, 100)
                
                # Add value labels on bars
                for i, (bar, prob) in enumerate(zip(bars, probs)):
                    ax.text(prob + 1, i, f'{prob:.1f}%', va='center', fontweight='bold')
                
                ax.set_title('Financial Inclusion/Exclusion Risk Profile', fontsize=14, fontweight='bold', pad=20)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Interpretation
                st.markdown("---")
                st.subheader("Interpretation")
                
                if prediction == 1:
                    st.warning("""
                    **This individual is predicted to be FINANCIALLY EXCLUDED.**
                    
                    Financial exclusion means the individual does not currently use any formal or informal 
                    financial services (banking, mobile money, insurance, savings, credit).
                    
                    **Recommended Interventions:**
                    - Promote digital financial literacy programs
                    - Facilitate mobile money adoption
                    - Assist with national ID registration
                    - Enhance local financial infrastructure
                    """)
                else:
                    st.success("""
                    **This individual is predicted to be FINANCIALLY INCLUDED.**
                    
                    Financial inclusion means the individual uses at least one financial service. 
                    However, targeted retention and expansion programs can help deepen financial access.
                    """)
                
                # Key factors explanation
                st.markdown("---")
                st.subheader("Key Factors Affecting This Prediction")
                
                key_factors = {
                    'Mobile Phone Ownership': mobile_phone_ownership,
                    'National ID Ownership': id_ownership,
                    'Internet Access': internet_access,
                    'Location Type': location_type,
                    'Education Level': education_level,
                    'Income Source': income_source,
                    'County': county
                }
                
                factors_df = pd.DataFrame(list(key_factors.items()), columns=['Factor', 'Value'])
                st.table(factors_df)
                
                st.info("""
                **Note:** The most important factors for financial exclusion are:
                1. **Mobile Phone Ownership** - Essential for digital financial services
                2. **National ID Ownership** - Required for formal financial account opening
                3. **Internet Access** - Enables access to online banking and mobile money
                4. **Location Type** - Rural areas face infrastructure challenges
                """)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("Please check that all inputs are valid and the model is properly loaded.")

elif page == "About":
    st.header("About This Project")
    
    st.markdown("""
    ## Predicting Financial Exclusion in Kenya Using Machine Learning
    
    ### Background
    Financial exclusion remains a critical policy challenge in Kenya. Approximately 10% of adults 
    remain completely excluded from both formal and informal financial systems. This project applies 
    machine learning techniques to predict individuals at risk of financial exclusion.
    
    ### Data Source
    **2024 FinAccess Household Survey Dataset**
    - Over 3,800 variables covering demographic, geographic, and financial access information
    - Collected by Kenya National Bureau of Statistics (KNBS)
    - Survey period: January - March 2024
    - Coverage: All 47 counties in Kenya
    - Sampling method: Stratified random sampling for urban and rural representation
    
    ### Research Problem
    How can machine learning models predict financial exclusion among Kenyan adults using 
    socio-demographic, economic, and technology-access indicators?
    
    ### Key Features
    - **Demographic Factors**: Age, gender, education, marital status
    - **Economic Factors**: Household size, income source
    - **Technology Access**: Mobile phone ownership, internet access, ID ownership
    - **Geographic Factors**: Urban/rural location, county
    - **Health**: Disability status
    
    ### Models Used
    1. **Logistic Regression** - Highest interpretability and ROC-AUC (0.94)
    2. **Random Forest** - Captures non-linear relationships
    3. **XGBoost** - High performance on structured data
    4. **Support Vector Machine** - Effective in high-dimensional spaces
    
    ### Key Findings
    - **Mobile phone ownership** is the strongest predictor of financial inclusion
    - **National ID ownership** is critical for formal financial access
    - **Rural location** is significantly associated with exclusion
    - **Education level** correlates with financial service usage
    
    ### Model Performance
    - **ROC-AUC Score**: 0.94+ across all models
    - **Recall (Excluded)**: 87% - Captures most financially excluded individuals
    - **Precision (Excluded)**: 70% - Acceptable false positive rate for policy intervention
    """)

elif page == "Instructions":
    st.header("How to Use This Application")
    
    st.markdown("""
    ### Step-by-Step Guide
    
    **1. Navigate to Prediction Page**
    - Use the sidebar to select "Prediction"
    
    **2. Fill in Personal Information**
    - **Demographic Section**: Enter age, gender, marital status, household size
    - **Economic Section**: Select education level, income source, and location type
    - **Digital Assets**: Indicate ID and mobile phone ownership, internet access
    - **Health**: Specify disability status
    - **Geography**: Select county of residence
    
    **3. Click "Predict Financial Exclusion Risk"**
    - The model will process your inputs and generate a prediction
    
    **4. Review Results**
    - **Financial Status**: Whether the person is predicted to be included or excluded
    - **Probability Score**: The confidence level of the prediction
    - **Visualization**: Bar chart showing inclusion vs exclusion probability
    - **Interpretation**: Personalized explanation of the results
    
    ### Understanding the Outputs
    
    **Financial Exclusion** means:
    - No usage of banking services
    - No mobile money usage
    - No savings with SACCO or informal groups
    - No insurance coverage
    
    **Financial Inclusion** means:
    - Uses at least one of the above financial services
    
    ### Key Risk Indicators
    
    **High-Risk Factors for Exclusion:**
    - No mobile phone
    - No national ID
    - No internet access
    - Rural location
    - No formal education
    - Unemployed or casual labor income
    
    **Protective Factors for Inclusion:**
    - Mobile phone ownership
    - National ID ownership
    - Internet access
    - Higher education level
    - Formal employment
    - Urban location
    
    ### Limitations & Disclaimer
    
    - **Predictions are probabilistic**: Individual outcomes may vary
    - **Based on historical data**: 2024 FinAccess survey
    - **Kenya-specific**: Model trained on Kenyan population
    - **Policy tool**: Intended to support targeted financial inclusion programs
    
    ### For Questions or Feedback
    Please refer to the dissertation documentation or contact the project team.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px; padding: 20px;'>
    <p>Financial Exclusion Predictor | Dissertation CAT 1 | Admission Number: 134022</p>
</div>
""", unsafe_allow_html=True)

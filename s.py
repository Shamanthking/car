import streamlit as st
import pandas as pd
import numpy as np
import pickle as pk
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from openpyxl import Workbook
from sklearn.preprocessing import LabelEncoder
import os
import statsmodels.api as sm

# ---- HOME PAGE ----
def show_home(df):
    st.title("Welcome to the Car Price Prediction App 🚗")

    # ---- CUSTOM CSS FOR BACKGROUND ----
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("https://img.freepik.com/free-vector/lightened-luxury-sedan-car-against-night-city-with-headlamps-rear-tail-lights-lit_1284-28804.jpg");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        color: white;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.write("""
    This application leverages the power of **machine learning** to analyze car features, uncover insights, 
    and predict car prices with ease. Whether you're a car dealer, buyer, or data enthusiast, this tool 
    is designed to provide you with actionable insights and accurate predictions.
    """)
    st.subheader("📖 How to Use This App:")
    st.markdown("""
    1. **Explore and Analyze Data**  
       - Dive into the dataset with **interactive visualizations** and **metrics**:
       - Understand trends in car features like mileage, engine size, and brand popularity.
       - Identify key factors that influence car prices.
    
    2. **Predict Selling Prices**  
       - Provide the required details such as car age, mileage, and engine specifications.  
       - Instantly predict the expected selling price using powerful machine learning models.
    
    3. **Compare Machine Learning Models**  
       - Evaluate multiple models, including Random Forest, Gradient Boosting, and Linear Regression, 
         to see which performs best on your data.
    
    4. **Leave Feedback**  
       - Share your experience with the app to help us improve!
    """)

    # Display initial insights
    st.subheader("Dataset Overview")
    st.write(df.head())
    st.write(f"Number of records: {df.shape[0]} | Number of features: {df.shape[1]}")

df=pd.read_csv(r"c1.csv")
df1=pd.read_csv(r"c2.csv")
st.set_page_config(
    page_title='CarDheko price pridection',
    layout='wide'
)

# ---- PREDICTION PAGE FUNCTION ----
def show_prediction(df):
    st.title("Car Price Prediction 🚗")
    
        col1,col2=st.columns(2)
        
        with col1:
            Fuel_type=st.selectbox('Select the fuel type',df['fuel_type'].unique())
            df_1=df[df['fuel_type']==Fuel_type]
            body_type=st.selectbox('Select the body type',df['body_type'].unique())
            df_1=df[df['body_type']==body_type]
            owner_type=st.selectbox('Select the owner type',df['owner_type'].unique())
            df_1=df[df['owner_type']==owner_type]
            transmission_type=st.selectbox('Select the transmission type',df['transmission_type'].unique())
            df_1=df[df['body_type']==transmission_type]
            manufacture_year=st.selectbox('Select the manufacture year',df['manufacture'].unique())
            df_1=df[df['manufacture']==manufacture_year]
            kilometer=st.selectbox('Select the kilometer in log',df1['kilometers_log'].unique())
            df_1=df1[df1['kilometers_log']==kilometer]
        with col2:
              seat=st.selectbox('Select the seat type',df['seat'].unique())
              df_1=df[df['seat']==seat]
              car_model=st.selectbox('Select the number of owner',df['oem'].unique())
              df_1=df[df['oem']==car_model]
              Mileage=st.selectbox('Select the Mileage',df['mileage'].unique())
              df_2=df[df['mileage']==Mileage]
              Engine_Capacity=st.selectbox('Select the Engine Capacity',df['engine_capacity'].unique())
              df1=df[df['engine_capacity']==Engine_Capacity]
              city=st.selectbox('Select the city',df['city'].unique())
              df_2=df[df['city']==city]

if select=='prediction':
    input_data=df.drop(columns=['price'])
    output_data=df['price']

# ---- DATA ANALYSIS ----
def show_analysis(df):
    st.header("📊 Detailed Data Analysis")
    if df is not None:
        st.write("""
        Explore the dataset through a variety of visualizations and analyses to gain deeper insights into 
        the factors affecting car prices. Each visualization is explained for better interpretation.
        """)

        # 1. Brand Distribution
        st.subheader("🔍 Brand Distribution")
        st.write("This bar chart shows the count of cars available for each brand in the dataset.")
        brand_counts = df['brand'].value_counts()
        fig = px.bar(brand_counts, x=brand_counts.index, y=brand_counts.values, 
                     labels={'x': 'Brand', 'y': 'Count'}, title="Brand Distribution")
        st.plotly_chart(fig)

        # Other visualizations would go here (e.g., Fuel Type, Price Distribution, etc.)

# ---- MODEL COMPARISON ----
def show_model_comparison(df):
    st.header("Model Comparison")
    if df is not None:
        X = df.drop(columns=['selling_price'])
        y = df['selling_price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "Linear Regression": LinearRegression(),
            "K-Neighbors Regressor": KNeighborsRegressor(n_neighbors=5),
            "Decision Tree": DecisionTreeRegressor(random_state=42)
        }

        metrics = {"Model": [], "MSE": [], "RMSE": [], "R²": []}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            metrics["Model"].append(model_name)
            metrics["MSE"].append(mse)
            metrics["RMSE"].append(rmse)
            metrics["R²"].append(r2)

        metrics_df = pd.DataFrame(metrics)
        st.dataframe(metrics_df)

# ---- TEAM PAGE ----
def show_team():
    st.title("Meet the Team")
    st.write(""" 
    - *Deekshith N:* 4AD22CI009 
    - *Prashanth Singh H S:* 4AD22CI040 
    - *Shamanth M:* 4AD22CI047 
    - *Akash A S:* 4AD22CI400 
    """)
    st.balloons()

# ---- FEEDBACK & CONTACT PAGE ----
def show_feedback_and_contact():
    st.title("Feedback & Contact")

    # Feedback Form
    st.subheader("We'd love to hear your feedback!")
    rating = st.selectbox("Rate Us:", ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"], index=4)
    feedback = st.text_area("Share your suggestions or comments:")

    if st.button("Submit Feedback"):
        try:
            # Load existing feedback data
            feedback_file = 'feedback.xlsx'
            if os.path.exists(feedback_file):
                feedback_df = pd.read_excel(feedback_file, engine="openpyxl")
            else:
                feedback_df = pd.DataFrame(columns=["rating", "comments"])

            # Append new feedback
            new_feedback = pd.DataFrame([[rating, feedback]], columns=["rating", "comments"])
            with pd.ExcelWriter(feedback_file, engine="openpyxl", mode="a") as writer:
                feedback_df = pd.concat([feedback_df, new_feedback], ignore_index=True)
                feedback_df.to_excel(writer, index=False)

            st.success("Thank you for your feedback!")
        except Exception as e:
            st.error(f"Error saving feedback: {e}")

    # Contact Information
    st.subheader("Contact Information")
    st.write("For inquiries, reach out to us at:")
    st.write("Email: example@example.com")
    st.write("Phone: 123-456-7890")

# ---- MAIN FUNCTION ----
def main():
    df = load_data()
    if df is not None:
        menu = ["Home", "Prediction", "Data Analysis", "Model Comparison", "Team", "Feedback & Contact"]
        choice = st.sidebar.selectbox("Menu", menu)

        if choice == "Home":
            show_home(df)
        elif choice == "Prediction":
            show_prediction(df)
        elif choice == "Data Analysis":
            show_analysis(df)
        elif choice == "Model Comparison":
            show_model_comparison(df)
        elif choice == "Team":
            show_team()
        elif choice == "Feedback & Contact":
            show_feedback_and_contact()

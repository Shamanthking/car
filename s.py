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
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from openpyxl import Workbook
import os

# ---- Load Data ----
def load_data():
    try:
        df = pd.read_csv("c1.csv")
        df1 = pd.read_csv("c2.csv")
        return df, df1
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}")
        return None, None

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
    and predict car prices with ease.
    """)
    st.subheader("Dataset Overview")
    st.write(df.head())
    st.write(f"Number of records: {df.shape[0]} | Number of features: {df.shape[1]}")

# ---- PREDICTION PAGE FUNCTION ----
def show_prediction(df, df1, choice):
    st.title("Car Price Prediction 🚗")

    if choice == 'Create the Model':
        col1, col2 = st.columns(2)
        with col1:
            Fuel_type = st.selectbox('Select the fuel type', df['fuel_type'].unique())
            df_1 = df[df['fuel_type'] == Fuel_type]
            body_type = st.selectbox('Select the body type', df_1['body_type'].unique())
            df_1 = df_1[df_1['body_type'] == body_type]
            owner_type = st.selectbox('Select the owner type', df_1['owner_type'].unique())
            df_1 = df_1[df_1['owner_type'] == owner_type]
            transmission_type = st.selectbox('Select the transmission type', df_1['transmission_type'].unique())
            df_1 = df_1[df_1['transmission_type'] == transmission_type]
        with col2:
            manufacture_year = st.selectbox('Select the manufacture year', df_1['manufacture'].unique())
            df_1 = df_1[df_1['manufacture'] == manufacture_year]
            kilometer = st.selectbox('Select the kilometer log', df1['kilometers_log'].unique())
            df_1 = df1[df1['kilometers_log'] == kilometer]

    elif choice == 'Prediction':
        input_data = df.drop(columns=['price'])
        output_data = df['price']
        st.write("Prediction functionality will be added here.")

# ---- DATA ANALYSIS ----
def show_analysis(df):
    st.header("📊 Detailed Data Analysis")
    if df is not None:
        st.subheader("Brand Distribution")
        brand_counts = df['brand'].value_counts()
        fig = px.bar(brand_counts, x=brand_counts.index, y=brand_counts.values, 
                     labels={'x': 'Brand', 'y': 'Count'}, title="Brand Distribution")
        st.plotly_chart(fig)

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
    st.write("Team Member Information")

# ---- FEEDBACK PAGE ----
def show_feedback_and_contact():
    st.title("Feedback & Contact")
    rating = st.selectbox("Rate Us:", ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"], index=4)
    feedback = st.text_area("Share your suggestions or comments:")

    if st.button("Submit Feedback"):
        feedback_file = 'feedback.xlsx'
        try:
            if os.path.exists(feedback_file):
                feedback_df = pd.read_excel(feedback_file, engine="openpyxl")
            else:
                feedback_df = pd.DataFrame(columns=["Rating", "Comments"])
            new_feedback = pd.DataFrame([[rating, feedback]], columns=["Rating", "Comments"])
            feedback_df = pd.concat([feedback_df, new_feedback], ignore_index=True)
            feedback_df.to_excel(feedback_file, index=False, engine="openpyxl")
            st.success("Thank you for your feedback!")
        except Exception as e:
            st.error(f"Error: {e}")

# ---- MAIN FUNCTION ----
def main():
    df, df1 = load_data()
    if df is not None:
        menu = ["Home", "Create the Model", "Prediction", "Data Analysis", "Model Comparison", "Team", "Feedback & Contact"]
        choice = st.sidebar.selectbox("Menu", menu)

        if choice == "Home":
            show_home(df)
        elif choice in ["Create the Model", "Prediction"]:
            show_prediction(df, df1, choice)
        elif choice == "Data Analysis":
            show_analysis(df)
        elif choice == "Model Comparison":
            show_model_comparison(df)
        elif choice == "Team":
            show_team()
        elif choice == "Feedback & Contact":
            show_feedback_and_contact()

if __name__ == "__main__":
    st.set_page_config(page_title='CarDekho Price Prediction', layout='wide')
    main()

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

# ---- PAGE CONFIGURATION ----
st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="wide")

# ---- CUSTOM CSS ----
page_bg_img = '''
<style>
.stApp {
    background-image: url("https://i.pinimg.com/originals/65/3a/b9/653ab9dd1ef121f163c484d03322f1a9.jpg");
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
    color: white;
}
.sidebar .sidebar-content {
    background-color: #333333;
    color: white;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# ---- LOAD DATA ----
@st.cache_data
def load_data():
    try:
        file_path = 'data/carr.csv'
        df = pd.read_csv(file_path)
        df['car_age'] = 2024 - df['Model_Year']  # Calculate car age
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# ---- HOMEPAGE ----
def show_home():
    st.title("🚗 Car Price Prediction Web Application")
    st.subheader("Predict the price of used cars based on various features")
    st.write("""
        This application uses machine learning models to estimate used car prices. 
        You can input various features such as age, kilometers driven, and fuel type to get an accurate prediction.
    """)

# ---- TEAM SECTION ----
def show_team():
    st.title("👥 Our Team")
    st.write("""
        - **Deekshith N:** 4AD22CI009  
        - **Prashanth Singh H S:** 4AD22CI040  
        - **Shamanth M:** 4AD22CI047  
        - **Akash A S:** 4AD22CI400
    """)

# ---- PREDICTION PAGE ----
def show_prediction():
    st.title("🔍 Car Price Prediction")
    df = load_data()
    if df is not None:
        st.sidebar.header("Input Features")
        fuel_type = st.sidebar.selectbox("Fuel Type", df['Fuel_Type'].unique())
        body_type = st.sidebar.selectbox("Body Type", df['Body_Type'].unique())
        km_driven = st.sidebar.number_input("Kilometers Driven", int(df['Kilometers_Driven'].min()), int(df['Kilometers_Driven'].max()), 50000)
        transmission = st.sidebar.selectbox("Transmission", df['Transmission'].unique())
        owner_count = st.sidebar.number_input("Number of Owners", int(df['Owner_Count'].min()), int(df['Owner_Count'].max()), 1)
        brand = st.sidebar.selectbox("Brand", df['Brand'].unique())
        model = st.sidebar.selectbox("Model", df['model'].unique())
        model_year = st.sidebar.number_input("Year of Manufacture", min_value=1980, max_value=2024, value=2015)
        location = st.sidebar.selectbox("Location", df['Location'].unique())
        mileage = st.sidebar.slider("Mileage (kmpl)", float(df['mileage'].min()), float(df['mileage'].max()), 20.0)
        seats = st.sidebar.selectbox("Number of Seats", sorted(df['Number_of_Seats'].dropna().unique()))

        input_data = {
            'Fuel_Type': fuel_type,
            'Body_Type': body_type,
            'Kilometers_Driven': km_driven,
            'Transmission': transmission,
            'Owner_Count': owner_count,
            'Brand': brand,
            'model': model,
            'Model_Year': model_year,
            'Location': location,
            'mileage': mileage,
            'Number_of_Seats': seats,
            'car_age': 2024 - model_year
        }
        user_df = pd.DataFrame([input_data])
        user_df = pd.get_dummies(user_df).reindex(columns=df.drop(columns=['selling_price']).columns, fill_value=0)

        # Train and predict using Random Forest model
        X = df.drop(columns=['selling_price'])
        y = df['selling_price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predicted_price = model.predict(user_df)
        st.write(f"### Predicted Selling Price: ₹{predicted_price[0]:,.2f}")

# ---- DATA ANALYSIS ----
def show_analysis():
    st.title("📊 Data Analysis")
    df = load_data()
    if df is not None:
        st.subheader("Brand Distribution")
        brand_counts = df['Brand'].value_counts()
        fig = px.bar(brand_counts, x=brand_counts.index, y=brand_counts.values, labels={'x': 'Brand', 'y': 'Count'})
        st.plotly_chart(fig)

        st.subheader("Fuel Type Distribution")
        fuel_counts = df['Fuel_Type'].value_counts()
        fig = px.pie(fuel_counts, names=fuel_counts.index, values=fuel_counts.values)
        st.plotly_chart(fig)

# ---- NAVIGATION ----
menu_options = {
    "Home": show_home,
    "Car Price Prediction": show_prediction,
    "Data Analysis": show_analysis,
    "Team": show_team
}
selected_menu = st.sidebar.selectbox("Main Menu", list(menu_options.keys()))
menu_options[selected_menu]()

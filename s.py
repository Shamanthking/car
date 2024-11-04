import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

# ---- PAGE CONFIGURATION ----
st.set_page_config(page_title="Car Price Prediction & Analysis Dashboard", page_icon=":car:", layout="wide")

# ---- CUSTOM CSS FOR BACKGROUND ----
page_bg_img = '''
<style>
.stApp {
    background-image: url("https://i.pinimg.com/originals/65/3a/b9/653ab9dd1ef121f163c484d03322f1a9.jpg");
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
    color: white;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# ---- LOAD DATA ----
@st.cache_data
def load_data():
    """Loads and preprocesses the car dataset."""
    try:
        df = pd.read_csv('data/used_cars.csv', on_bad_lines='skip')

        # Check for missing values and handle them
        if df.isnull().sum().any():
            st.warning("Data contains missing values. Handling missing values...")
            df.fillna(0, inplace=True)

        # Create car_age from Year and drop the original Year column
        df['car_age'] = 2024 - df['Year']
        df.drop(columns=['Year'], inplace=True)

        # Rename target column
        df.rename(columns={'Selling_Price': 'price', 'Present_Price': 'current_price',
                           'Kms_Driven': 'km_driven'}, inplace=True)

        # One-Hot Encoding for categorical features
        categorical_cols = ['Car_Name', 'Fuel_Type', 'Seller_Type', 'Transmission']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        return df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# ---- INITIALIZE SESSION STATE FOR PAGE TRACKING ----
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Function to switch page
def switch_page(page_name):
    st.session_state.page = page_name

# Sidebar Navigation buttons
st.sidebar.title("Navigation")
st.sidebar.button("Home", on_click=switch_page, args=('home',))
st.sidebar.button("Prediction", on_click=switch_page, args=('prediction',))
st.sidebar.button("Data Analysis", on_click=switch_page, args=('analysis',))
st.sidebar.button("Model Comparison", on_click=switch_page, args=('model_comparison',))
st.sidebar.button("Contact", on_click=switch_page, args=('contact',))

# ---- PAGE SECTIONS ----
def show_home():
    st.title("Car Price Prediction")
    st.subheader("Get accurate predictions on car prices and explore data insights.")

def show_prediction():
    st.header("Car Price Prediction")

    # Load data
    df = load_data()
    if df is not None:
        # Train Random Forest model
        X = df.drop(columns=['price'])
        y = df['price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Extract unique car names before one-hot encoding
        unique_car_names = [col.split('_', 1)[-1] for col in df.columns if col.startswith("Car_Name_")]

        # Prediction input fields
        car_name = st.selectbox("Car Name", sorted(unique_car_names))
        fuel_type = st.selectbox("Fuel Type", sorted(df['Fuel_Type_Petrol'].unique()))
        seller_type = st.selectbox("Seller Type", sorted(df['Seller_Type_Individual'].unique()))
        transmission = st.selectbox("Transmission", sorted(df['Transmission_Manual'].unique()))
        car_age = st.slider("Car Age", 0, 20, 10)
        km_driven = st.number_input("Kilometers Driven", 0, 300000, 50000)
        current_price = st.number_input("Current Price (in Lakh)", 0.0, 50.0, 5.0)
        owner = st.selectbox("Owner", [0, 1, 2, 3])

        # Prepare input for prediction
        input_data = pd.DataFrame({
            f'Car_Name_{car_name}': [1],
            f'Fuel_Type_Petrol' if fuel_type == "Petrol" else "Fuel_Type_Diesel": [1],
            f'Seller_Type_Individual' if seller_type == "Individual" else "Seller_Type_Dealer": [1],
            f'Transmission_Manual' if transmission == "Manual" else "Transmission_Automatic": [1],
            'car_age': [car_age],
            'km_driven': [km_driven],
            'current_price': [current_price],
            'Owner': [owner]
        })

        # Add missing columns for prediction
        missing_cols = set(X_train.columns) - set(input_data.columns)
        for col in missing_cols:
            input_data[col] = 0
        input_data = input_data[X_train.columns]

        # Prediction
        try:
            prediction = model.predict(input_data)
            st.write(f"Predicted Selling Price: ₹ {prediction[0]:,.2f} Lakh")
        except Exception as e:
            st.error(f"Prediction error: {e}")

def show_analysis():
    # (Analysis code remains unchanged)
    pass

def show_model_comparison():
    # (Model comparison code remains unchanged)
    pass

def show_contact():
    st.header("Contact Us")
    st.markdown("""
        - [LinkedIn](https://www.linkedin.com/in/shamanth-m-05537b264)
        - [Instagram](https://www.instagram.com/shamanth_m_)
        - [Email](mailto:shamanth2626@gmail.com)
    """)

# ---- DISPLAY SELECTED PAGE ----
if st.session_state.page == 'home':
    show_home()
elif st.session_state.page == 'prediction':
    show_prediction()
elif st.session_state.page == 'analysis':
    show_analysis()
elif st.session_state.page == 'model_comparison':
    show_model_comparison()
elif st.session_state.page == 'contact':
    show_contact()

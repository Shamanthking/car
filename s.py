import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# ---- PAGE CONFIGURATION ----
st.set_page_config(page_title="Car Price Prediction & Analysis Dashboard", page_icon="🚗", layout="wide")

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
def load_data(uploaded_file=None):
    """Loads and preprocesses the car dataset, either from uploaded file or fallback path."""
    try:
        # Load data from the uploaded file
        df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')

        # Display initial columns and data types
        st.write("Initial Columns:", df.columns)
        st.write("Data Types:", df.dtypes)
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
        
        # Impute missing values for numeric columns
        imputer = SimpleImputer(strategy="mean")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

        # Feature Engineering - derive car age if model_year available
        if 'model_year' in df.columns:
            df['car_age'] = 2024 - df['model_year']
            df.drop(columns=['model_year'], inplace=True)

        # Handle categorical data
        categorical_cols = ['brand', 'fuel_type', 'transmission_type', 'seller_type']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# ---- PAGE SECTIONS ----
def show_home():
    st.title("Car Price Prediction & Analysis")
    st.subheader("Get predictions and insights into car price data with multiple model comparisons.")

def show_prediction(uploaded_file=None):
    st.header("Car Price Prediction")

    df = load_data(uploaded_file)
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

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            st.write(f"{model_name}:")
            st.write(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

# ---- ADDITIONAL SECTIONS ----
# Define functions for analysis, model comparison, and contact as in the original code

# ---- DISPLAY SELECTED PAGE ----
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Navigation logic to switch between pages
if 'page' not in st.session_state:
    st.session_state.page = 'home'

def switch_page(page_name):
    st.session_state.page = page_name

st.sidebar.title("Navigation")
st.sidebar.button("Home", on_click=switch_page, args=('home',))
st.sidebar.button("Prediction", on_click=switch_page, args=('prediction',))
st.sidebar.button("Data Analysis", on_click=switch_page, args=('analysis',))
st.sidebar.button("Model Comparison", on_click=switch_page, args=('model_comparison',))
st.sidebar.button("Contact", on_click=switch_page, args=('contact',))

# Display the appropriate page
if st.session_state.page == 'home':
    show_home()
elif st.session_state.page == 'prediction':
    show_prediction(uploaded_file)
elif st.session_state.page == 'analysis':
    show_analysis(uploaded_file)
elif st.session_state.page == 'model_comparison':
    show_model_comparison(uploaded_file)
elif st.session_state.page == 'contact':
    show_contact()

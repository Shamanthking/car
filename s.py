import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
from openpyxl import Workbook
import os

# ---- PAGE CONFIGURATION ----
st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="wide")

# ---- EXCEL FILE SETUP ----
users_file = 'users_data.xlsx'
feedback_file = 'feedback_data.xlsx'

# Helper function to create an empty Excel file with specified columns
def create_empty_excel(file_name, columns):
    wb = Workbook()
    ws = wb.active
    ws.append(columns)
    wb.save(file_name)

# Create users and feedback Excel files if they don't exist or are not valid Excel files
for file, columns in [(users_file, ["username", "email", "password"]), 
                      (feedback_file, ["rating", "comments"])]:
    if not os.path.exists(file):
        create_empty_excel(file, columns)
    else:
        try:
            # Try to read the file to check if it is a valid Excel file
            pd.read_excel(file, engine="openpyxl")
        except Exception:
            # If not a valid Excel file, recreate it
            os.remove(file)
            create_empty_excel(file, columns)

# ---- AUTHENTICATION ----
def add_user(username, email, password):
    users_df = pd.read_excel(users_file, engine="openpyxl")
    if (users_df['username'] == username).any() or (users_df['email'] == email).any():
        st.sidebar.error("Username or email already exists.")
    else:
        new_user = pd.DataFrame([[username, email, password]], columns=["username", "email", "password"])
        with pd.ExcelWriter(users_file, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
            new_user.to_excel(writer, index=False, header=False, startrow=len(users_df) + 1)
        st.sidebar.success("User registered successfully. Please login.")

def authenticate_user():
    st.sidebar.title("Authentication")
    auth_option = st.sidebar.radio("Choose Option", ["Login", "Register"], key="auth_option")

    if auth_option == "Login":
        username = st.sidebar.text_input("Username", key="login_username")
        password = st.sidebar.text_input("Password", type="password", key="login_password")
        if st.sidebar.button("Login"):
            users_df = pd.read_excel(users_file, engine="openpyxl")
            user = users_df[(users_df['username'] == username) & (users_df['password'] == password)]
            if not user.empty:
                st.sidebar.success(f"Welcome, {username}!")
                return True
            else:
                st.sidebar.error("Invalid username or password.")
                return False

    elif auth_option == "Register":
        new_username = st.sidebar.text_input("Create Username", key="register_username")
        email = st.sidebar.text_input("Email", key="register_email")
        new_password = st.sidebar.text_input("Create Password", type="password", key="register_password")
        confirm_password = st.sidebar.text_input("Confirm Password", type="password", key="confirm_password")
        if st.sidebar.button("Register"):
            if new_password == confirm_password:
                add_user(new_username, email, new_password)
            else:
                st.sidebar.error("Passwords do not match.")
    return False

# ---- LOAD DATA ----
@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
        
        # Encode categorical features and handle missing values
        cat_cols = df.select_dtypes(include=['object']).columns
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy="mean")
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# ---- TRAIN MODEL ----
@st.cache_resource
def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# ---- HOMEPAGE ----
def show_home():
    st.title("Car Price Prediction Application 🚗")
    st.write("Welcome to the Car Price Prediction app! This tool helps predict car prices, explore data insights, and compare machine learning models.")
    st.image("https://images.pexels.com/photos/10287567/pexels-photo-10287567.jpeg", caption="Predict Car Prices Instantly", use_column_width=True)

# ---- CAR PRICE PREDICTION ----
def show_prediction(df):
    st.title("Car Price Prediction")
    car_age = st.slider("Car Age", 0, 20, 10)
    km_driven = st.number_input("Kilometers Driven", 0, 300000, 50000)
    seats = st.selectbox("Seats", [2, 4, 5, 7])
    max_power = st.number_input("Max Power (in bhp)", 50, 500, 100)
    mileage = st.number_input("Mileage (kmpl)", 5.0, 35.0, 20.0)
    engine_cc = st.number_input("Engine Capacity (CC)", 500, 5000, 1200)
    
    # Gather categories dynamically
    brand = st.selectbox("Brand", df.filter(regex='^brand_').columns)
    fuel_type = st.selectbox("Fuel Type", df.filter(regex='^fuel_type_').columns)
    seller_type = st.selectbox("Seller Type", df.filter(regex='^seller_type_').columns)
    transmission = st.selectbox("Transmission", df.filter(regex='^transmission_type_').columns)

    # Prepare user input for model prediction
    X = df.drop(columns=['selling_price'])
    y = df['selling_price']
    user_data = pd.DataFrame({
        'vehicle_age': [car_age],
        'km_driven': [km_driven],
        'seats': [seats],
        'max_power': [max_power],
        'mileage': [mileage],
        'engine': [engine_cc],
        **{col: [0] for col in X.columns if col.startswith(('brand_', 'fuel_type_', 'seller_type_', 'transmission_type_'))}
    })
    user_data[brand] = 1
    user_data[fuel_type] = 1
    user_data[seller_type] = 1
    user_data[transmission] = 1

    # Train model and predict price
    model = train_model(X, y)
    predicted_price = model.predict(user_data)
    st.write(f"### Predicted Selling Price: ₹{predicted_price[0]:,.2f}")

# ---- DATA ANALYSIS ----
def show_analysis(df):
    st.title("Data Analysis")
    st.subheader("Brand Distribution")
    fig1 = px.bar(df['brand'].value_counts(), labels={'x': 'Brand', 'y': 'Count'})
    st.plotly_chart(fig1)

# ---- TEAM ----
def show_team():
    st.title("Meet the Team")
    st.write("""
    - **John Doe**: Data Scientist  
    - **Jane Smith**: Backend Developer  
    - **Alan Turing**: ML Specialist  
    """)

# ---- FEEDBACK ----
def show_feedback():
    st.title("Feedback & Contact")
    rating = st.selectbox("Rate Us:", ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"], index=4)
    feedback = st.text_area("Suggestions or comments?")
    if st.button("Submit Feedback"):
        new_feedback = pd.DataFrame([[rating, feedback]], columns=["rating", "comments"])
        with pd.ExcelWriter(feedback_file, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
            new_feedback.to_excel(writer, index=False, header=False, startrow=len(pd.read_excel(feedback_file, engine="openpyxl")) + 1)
        st.success("Thank you for your feedback!")
    st.write("Contact Us: support@carpredictionapp.com | +123-456-7890")

# ---- NAVIGATION ----
if authenticate_user():
    menu_options = {
        "Home": show_home,
        "Car Price Prediction": show_prediction,
        "Data Analysis": show_analysis,
        "Team": show_team,
        "Feedback & Contact": show_feedback
    }
    selected_menu = st.sidebar.selectbox("Main Menu", list(menu_options.keys()))
    uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            menu_options[selected_menu](df)
    else:
        st.sidebar.info("Please upload a CSV dataset to proceed.")

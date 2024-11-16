# Required imports
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
from openpyxl import Workbook
import os

# ---- PAGE CONFIGURATION ----
st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="wide")

# ---- FILE SETUP ----
users_file = 'users.xlsx'
feedback_file = 'feedback.xlsx'

# Helper function to create Excel files if they do not exist
def create_empty_excel(file_name, columns):
    wb = Workbook()
    ws = wb.active
    ws.append(columns)
    wb.save(file_name)

# Ensure required files exist
for file, columns in [(users_file, ["username", "email", "password"]),
                      (feedback_file, ["rating", "comments"])]:
    if not os.path.exists(file):
        create_empty_excel(file, columns)
    else:
        try:
            pd.read_excel(file, engine="openpyxl")
        except Exception:
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

# ---- DATA LOADING ----
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
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

# ---- HOME PAGE ----
def show_home(df):
    st.title("Welcome to the Car Price Prediction App 🚗")
    st.image("https://images.pexels.com/photos/10287567/pexels-photo-10287567.jpeg", caption="Accurate Car Price Predictions", use_column_width=True)
    st.write("""
    This application provides insights into car prices using machine learning models.  
    You can upload your dataset, analyze car features, and predict selling prices instantly.
    """)

    # Display initial insights
    st.subheader("Dataset Overview")
    st.write(df.head())
    st.write(f"Number of records: {df.shape[0]} | Number of features: {df.shape[1]}")

# ---- PREDICTION PAGE ----
def show_prediction(df):
    st.title("Car Price Prediction")
    st.subheader("Input the car details below:")

    car_age = st.slider("Car Age", 0, 20, 5)
    km_driven = st.number_input("Kilometers Driven", 0, 300000, 50000)
    seats = st.selectbox("Seats", [2, 4, 5, 7])
    max_power = st.number_input("Max Power (in bhp)", 50, 500, 100)
    mileage = st.number_input("Mileage (kmpl)", 5.0, 35.0, 20.0)
    engine_cc = st.number_input("Engine Capacity (CC)", 500, 5000, 1200)
    
    # Prepare dynamic user input
    brand_cols = df.filter(regex='^brand_').columns
    fuel_cols = df.filter(regex='^fuel_type_').columns
    seller_cols = df.filter(regex='^seller_type_').columns
    transmission_cols = df.filter(regex='^transmission_type_').columns

    brand = st.selectbox("Brand", brand_cols)
    fuel_type = st.selectbox("Fuel Type", fuel_cols)
    seller_type = st.selectbox("Seller Type", seller_cols)
    transmission = st.selectbox("Transmission", transmission_cols)

    # Prepare input
    X = df.drop(columns=['selling_price'])
    y = df['selling_price']
    user_input = pd.DataFrame({
        'vehicle_age': [car_age],
        'km_driven': [km_driven],
        'seats': [seats],
        'max_power': [max_power],
        'mileage': [mileage],
        'engine': [engine_cc],
        **{col: [0] for col in brand_cols.append(fuel_cols).append(seller_cols).append(transmission_cols)}
    })
    user_input[brand] = 1
    user_input[fuel_type] = 1
    user_input[seller_type] = 1
    user_input[transmission] = 1

    # Train model and predict
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    predicted_price = model.predict(user_input)

    st.success(f"### Predicted Selling Price: ₹{predicted_price[0]:,.2f}")

# ---- DATA ANALYSIS ----
def show_analysis(df):
    st.title("Data Analysis")

    st.subheader("1. Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("2. Feature Importance")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    X = df.drop(columns=['selling_price'])
    y = df['selling_price']
    model.fit(X, y)
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x=importance, y=importance.index, ax=ax)
    st.pyplot(fig)

    # Additional analysis plots
    st.subheader("3. Scatter Plot: Mileage vs Selling Price")
    fig = px.scatter(df, x='mileage', y='selling_price', trendline='ols')
    st.plotly_chart(fig)

    # ... Add 7 more plots similarly ...

# ---- TEAM PAGE ----
def show_team():
    st.title("Meet the Team")
    st.write("""
    - **John Doe**: Data Scientist  
    - **Jane Smith**: Backend Developer  
    - **Alan Turing**: ML Specialist  
    """)

# ---- FEEDBACK & CONTACT PAGE ----
def show_feedback_and_contact():
    st.title("Feedback & Contact")

    # Feedback Form
    st.subheader("We'd love to hear your feedback!")
    rating = st.selectbox("Rate Us:", ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"], index=4)
    feedback = st.text_area("Share your suggestions or comments:")
    
    if st.button("Submit Feedback"):
        if feedback.strip():
            new_feedback = pd.DataFrame([[rating, feedback]], columns=["rating", "comments"])
            with pd.ExcelWriter(feedback_file, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
                new_feedback.to_excel(writer, index=False, header=False, startrow=len(pd.read_excel(feedback_file, engine="openpyxl")) + 1)
            st.success("Thank you for your feedback!")
        else:
            st.error("Please provide your feedback before submitting.")

    # Contact Information
    st.subheader("Contact Us")
    st.write("""
    If you have any questions or need support, feel free to reach out to us:

    - 📧 **Email**: support@carpriceprediction.com  
    - 📞 **Phone**: +1-800-123-4567  
    - 🌐 **Website**: [www.carpriceprediction.com](https://www.carpriceprediction.com)
    """)

    # Social Media Links
    st.write("Follow us on:")
    st.markdown("""
    - [LinkedIn](https://www.linkedin.com)  
    - [Twitter](https://twitter.com)  
    - [Facebook](https://facebook.com)  
    - [Instagram](https://instagram.com)
    """)


# ---- MAIN APP ----
if authenticate_user():
    st.sidebar.title("Menu")
    menu = st.sidebar.radio("Select a page:", ["Home", "Prediction", "Analysis", "Team", "Feedback"])

    df = load_data('data/carr.csv')
    if df is not None:
        if menu == "Home":
            show_home(df)
        elif menu == "Prediction":
            show_prediction(df)
        elif menu == "Analysis":
            show_analysis(df)
        elif menu == "Team":
            show_team()
        elif menu == "Feedback":
            show_feedback_and_contact()

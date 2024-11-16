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
from sklearn.preprocessing import LabelEncoder
import streamlit_authenticator as stauth
import sqlite3
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# ---- PAGE CONFIGURATION ----
st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="wide")

# ---- DATABASE SETUP FOR NEW USERS ----
conn = sqlite3.connect('users.db')
cursor = conn.cursor()

# Create Users Table if not exists
cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        email TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL
    )
""")
conn.commit()

# ---- HELPER FUNCTION TO ADD USERS ----
def add_user(username, email, password):
    cursor.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", (username, email, password))
    conn.commit()

# ---- AUTHENTICATION CONFIGURATION ----
def authenticate_user():
    st.sidebar.title("Authentication")
    auth_option = st.sidebar.radio("Choose Option", ["Login", "Register"], key="auth_option")

    if auth_option == "Login":
        username = st.sidebar.text_input("Username", key="login_username")
        password = st.sidebar.text_input("Password", type="password", key="login_password")
        if st.sidebar.button("Login"):
            cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
            user = cursor.fetchone()
            if user:
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
                try:
                    add_user(new_username, email, new_password)
                    st.sidebar.success("User registered successfully. Please login.")
                except sqlite3.IntegrityError:
                    st.sidebar.error("Username or email already exists.")
            else:
                st.sidebar.error("Passwords do not match.")
    return False
# ---- LOAD DATA ----
@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
        cat_cols = df.select_dtypes(include=['object']).columns.difference(['brand', 'model'])
        df[cat_cols] = df[cat_cols].apply(LabelEncoder().fit_transform)
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
    brand = st.selectbox("Brand", df['brand'].unique())
    fuel_type = st.selectbox("Fuel Type", df['fuel_type'].unique())
    seller_type = st.selectbox("Seller Type", df['seller_type'].unique())
    transmission = st.selectbox("Transmission", df['transmission_type'].unique())

    X = df.drop(columns=['selling_price'])
    y = df['selling_price']
    user_data = pd.DataFrame({
        'vehicle_age': [car_age],
        'km_driven': [km_driven],
        'seats': [seats],
        'max_power': [max_power],
        'mileage': [mileage],
        'engine': [engine_cc],
        'brand': [brand],
        'fuel_type': [fuel_type],
        'seller_type': [seller_type],
        'transmission_type': [transmission]
    })
    user_data = pd.get_dummies(user_data)
    user_data = user_data.reindex(columns=X.columns, fill_value=0)
    model = train_model(X, y)
    predicted_price = model.predict(user_data)
    st.write(f"### Predicted Selling Price: ₹{predicted_price[0]:,.2f}")

# ---- DATA ANALYSIS ----
def show_analysis(df):
    st.title("Data Analysis")

    st.subheader("Brand Distribution")
    fig1 = px.bar(df['brand'].value_counts(), labels={'x': 'Brand', 'y': 'Count'})
    st.plotly_chart(fig1)

    st.subheader("Fuel Type Distribution")
    fig2 = px.pie(df, names='fuel_type', title="Fuel Type Share")
    st.plotly_chart(fig2)

    st.subheader("Transmission Type Count")
    fig3 = px.histogram(df, x='transmission_type')
    st.plotly_chart(fig3)

    st.subheader("Selling Price vs. Engine Capacity")
    fig4 = px.scatter(df, x='engine', y='selling_price', color='brand', title="Price vs Engine")
    st.plotly_chart(fig4)

    st.subheader("Heatmap: Correlation")
    corr = df.corr()
    fig5, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig5)

# ---- MODEL COMPARISON ----
def show_model_comparison(df):
    st.title("Model Comparison")
    X = df.drop(columns=['selling_price'])
    y = df['selling_price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "Linear Regression": LinearRegression(),
        "K-Neighbors Regressor": KNeighborsRegressor(),
        "Decision Tree": DecisionTreeRegressor()
    }
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append((name, mse, r2))

    results_df = pd.DataFrame(results, columns=["Model", "MSE", "R²"])
    st.dataframe(results_df)

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
        cursor.execute("INSERT INTO feedback (rating, comments) VALUES (?, ?)", (rating, feedback))
        conn.commit()
        st.success("Thank you for your feedback!")
    st.write("Contact Us: support@carpredictionapp.com | +123-456-7890")

# ---- NAVIGATION ----
if authenticate_user():
    menu_options = {
        "Home": show_home,
        "Car Price Prediction": show_prediction,
        "Data Analysis": show_analysis,
        "Model Comparison": show_model_comparison,
        "Team": show_team,
        "Feedback & Contact": show_feedback
    }
    selected_menu = st.sidebar.selectbox("Main Menu", list(menu_options.keys()))
    uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv"])
    if uploaded_file:
        df = load_data(uploaded_file)
        if selected_menu in ["Car Price Prediction", "Data Analysis", "Model Comparison"] and df is not None:
            menu_options[selected_menu](df)
        else:
            menu_options[selected_menu]()
    else:
        st.warning("Please upload a dataset to proceed.")

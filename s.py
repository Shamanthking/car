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
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# ---- PAGE CONFIGURATION ----
st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="wide")

# ---- LOAD DATA ----
@st.cache_data
def load_data():
    try:
        file_path = 'data/carr.csv'
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

        # Encode categorical features
        cat_cols = df.select_dtypes(include=['object']).columns.difference(['brand', 'model'])
        df[cat_cols] = df[cat_cols].apply(LabelEncoder().fit_transform)

        # Impute missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy="mean")
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# ---- HOMEPAGE ----
def show_home():
    st.title("Car Price Prediction Web Application")
    st.write("Welcome to the Car Price Prediction app! This tool helps predict car prices, explore data insights, and compare machine learning models.")

# ---- PREDICTION PAGE ----
def show_prediction():
    st.title("Car Price Prediction")
    df = load_data()
    if df is not None:
        # Input fields for prediction
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

        # Prepare data for model prediction
        X = df.drop(columns=['selling_price'])
        y = df['selling_price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

        # One-hot encoding for the categorical features
        user_data = pd.get_dummies(user_data)
        user_data = user_data.reindex(columns=X.columns, fill_value=0)

        # Train and predict using Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predicted_price = model.predict(user_data)
        st.write(f"### Predicted Selling Price: ₹{predicted_price[0]:,.2f}")

# ---- DATA ANALYSIS ----
def show_analysis():
    st.title("Data Analysis")
    st.write("Explore the car dataset with insightful visualizations.")

    df = load_data()
    if df is not None:
        # (10 Analysis Charts as mentioned earlier)
        st.subheader("Brand Distribution")
        fig1 = px.bar(df['brand'].value_counts(), labels={'x': 'Brand', 'y': 'Count'})
        st.plotly_chart(fig1)

# ---- MODEL COMPARISON ----
def show_model_comparison():
    st.title("Model Comparison")
    st.write("Evaluate different machine learning models on their performance.")

    df = load_data()
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
        st.dataframe(metrics_df.style.highlight_min(subset=['MSE', 'RMSE'], color='lightgreen').highlight_max(subset=['R²'], color='lightblue'))

# ---- TEAM SECTION ----
def show_team():
    st.title("Meet the Team")
    st.write("""
    - **Deekshith N**: Data Scientist  
    - **Prashanth Singh H S**: Machine Learning Engineer  
    - **Shamanth M**: Backend Developer  
    - **Akash A S**: Frontend Developer
    """)

# ---- FEEDBACK & CONTACT ----
def show_feedback_contact():
    st.title("We Value Your Feedback")
    st.write("Rate your experience and leave your feedback.")

    rating = st.selectbox("Rate Us:", ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"], index=4)
    feedback = st.text_area("Suggestions or comments?")

    if st.button("Submit Feedback"):
        feedback_data = {"Rating": rating, "Feedback": feedback}
        feedback_df = pd.DataFrame([feedback_data])
        feedback_df.to_excel("feedback.xlsx", index=False, mode="a", header=False)
        st.write("Thank you for your feedback!")

    st.subheader("Contact Us")
    st.write("""
    - Email: support@carpredictionapp.com  
    - Phone: +123-456-7890
    """)

# ---- NAVIGATION ----
menu_options = {
    "Home": show_home,
    "Car Price Prediction": show_prediction,
    "Data Analysis": show_analysis,
    "Model Comparison": show_model_comparison,
    "Team": show_team,
    "Feedback & Contact": show_feedback_contact
}

selected_menu = st.sidebar.selectbox("Main Menu", list(menu_options.keys()))
menu_options[selected_menu]()

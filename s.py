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

# ---- DATASET DESCRIPTION ----
dataset_description = """
### Used Car Price Prediction Dataset
This dataset, sourced from [cars.com](https://www.cars.com), contains 4,009 records of used car listings with the following features:

- **Brand & Model**: The brand and model of each vehicle.
- **Model Year**: The manufacturing year, useful for estimating depreciation.
- **Mileage**: The mileage of the vehicle, indicating wear and tear.
- **Fuel Type**: The type of fuel used by the vehicle (gasoline, diesel, electric, hybrid).
- **Engine Type**: Specifications related to the vehicle's engine performance.
- **Transmission**: Whether the vehicle has automatic or manual transmission.
- **Exterior & Interior Colors**: Color details that may influence aesthetic preference.
- **Accident History**: Indicates whether the vehicle has a history of accidents.
- **Clean Title**: Shows if the vehicle has a clean title, impacting resale value.
- **Price**: The listing price of the vehicle.

This dataset is valuable for automotive enthusiasts, researchers, and anyone interested in analyzing market trends or making informed car purchasing decisions.
"""
st.sidebar.markdown(dataset_description)

# ---- LOAD DATA ----
@st.cache_data
def load_data():
    """Loads and preprocesses the car dataset."""
    try:
        df = pd.read_csv('data/used_cars.csv', on_bad_lines='skip')

        # Handle missing values
        if df.isnull().sum().any():
            df.fillna(0, inplace=True)

        # Calculate car age and drop model_year
        df['car_age'] = 2024 - df['model_year']
        df.drop(columns=['model_year'], inplace=True)

        # One-Hot Encoding for categorical features including 'brand'
        categorical_cols = ['brand', 'fuel_type', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# ---- PAGE NAVIGATION ----
if 'page' not in st.session_state:
    st.session_state.page = 'home'

def switch_page(page_name):
    st.session_state.page = page_name

# Sidebar Navigation
st.sidebar.title("Navigation")
st.sidebar.button("Home", on_click=switch_page, args=('home',))
st.sidebar.button("Prediction", on_click=switch_page, args=('prediction',))
st.sidebar.button("Data Analysis", on_click=switch_page, args=('analysis',))
st.sidebar.button("Model Comparison", on_click=switch_page, args=('model_comparison',))
st.sidebar.button("Contact", on_click=switch_page, args=('contact',))

# ---- PAGE FUNCTIONS ----
def show_home():
    st.title("Car Price Prediction & Analysis Dashboard")
    st.subheader("Use this dashboard to predict car prices and explore various trends in car features.")

def show_prediction():
    st.header("Car Price Prediction")

    df = load_data()
    if df is not None:
        model, X_train, X_test, y_train, y_test = train_random_forest_model(df)

        # Prediction input fields
        brand = st.selectbox("Brand", sorted([col.replace('brand_', '') for col in X_train.columns if 'brand_' in col]))
        car_age = st.slider("Car Age", 0, 20, 10)
        km_driven = st.number_input("Kilometers Driven", 0, 300000, 50000)
        seats = st.selectbox("Seats", [2, 4, 5, 7])
        max_power = st.number_input("Max Power (in bhp)", 50, 500, 100)
        mileage = st.number_input("Mileage (kmpl)", 5.0, 35.0, 20.0)
        engine_cc = st.number_input("Engine Capacity (CC)", 500, 5000, 1200)
        
        # Prepare input for prediction
        input_data = pd.DataFrame({
            'car_age': [car_age],
            'km_driven': [km_driven],
            'Seats': [seats],
            'max_power': [max_power],
            'mileage': [mileage],
            'engine_cc': [engine_cc]
        })

        # One-hot encode selected brand
        for col in X_train.columns:
            if col.startswith("brand_"):
                input_data[col] = 1 if col == f"brand_{brand}" else 0

        # Add missing columns
        missing_cols = set(X_train.columns) - set(input_data.columns)
        for col in missing_cols:
            input_data[col] = 0
        input_data = input_data[X_train.columns]

        # Prediction
        try:
            prediction = model.predict(input_data)
            st.write(f"Predicted Selling Price: ₹ {prediction[0]:,.2f}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

def show_analysis():
    st.header("Detailed Data Analysis")
    df = load_data()

    if df is not None:
        # Brand Distribution
        st.subheader("Car Brand Distribution")
        brand_count = df['brand'].value_counts()
        fig = px.bar(brand_count, x=brand_count.index, y=brand_count.values, labels={'x': 'Brand', 'y': 'Count'}, title="Brand Distribution")
        st.plotly_chart(fig)

        # Accident History and Price Impact
        st.subheader("Price Impact by Accident History")
        fig = px.box(df, x="accident", y="price", title="Price Distribution by Accident History")
        st.plotly_chart(fig)

        # Transmission Breakdown
        st.subheader("Transmission Type Breakdown")
        transmission_count = df['transmission'].value_counts()
        fig = px.pie(transmission_count, names=transmission_count.index, values=transmission_count.values, title="Transmission Type Distribution")
        st.plotly_chart(fig)

        # Price vs Mileage and Car Age
        st.subheader("Price vs Mileage and Car Age")
        fig = px.scatter(df, x="mileage", y="price", trendline="ols", title="Mileage vs. Price")
        st.plotly_chart(fig)
        fig = px.scatter(df, x="car_age", y="price", trendline="ols", title="Car Age vs. Price")
        st.plotly_chart(fig)

def show_model_comparison():
    st.header("Model Comparison")
    st.write("Compare model performance metrics on training and test datasets.")
    
    df = load_data()
    if df is not None:
        X = df.drop(columns=['price'])
        y = df['price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
        }

        metrics = {"Model": [], "RMSE": [], "MAE": [], "R² Score": []}

        for model_name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                metrics["Model"].append(model_name)
                metrics["RMSE"].append(rmse)
                metrics["MAE"].append(mae)
                metrics["R² Score"].append(r2)
            except Exception as e:
                st.error(f"Error with {model_name}: {e}")

        metrics_df = pd.DataFrame(metrics)
        st.table(metrics_df)

def show_contact():
    st.header("Contact Information")
    st.write("For inquiries, please reach out to [your_email@example.com].")

# ---- TRAIN RANDOM FOREST MODEL ----
@st.cache_resource
def train_random_forest_model(df):
    """Trains the Random Forest model."""
    X = df.drop(columns=['price'])
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test

# ---- RENDER PAGES ----
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

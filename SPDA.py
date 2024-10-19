import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title="Car Price Prediction Dashboard", page_icon=":car:", layout="wide")

# ---- LOAD DATA ----
def load_data():
    uploaded_file = st.sidebar.file_uploader("Upload A Dataset (.xlsx)", type=["xlsx"])
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            
            # Data Preprocessing
            df.drop(columns=["Unnamed: 0", "Mileage Unit"], inplace=True, errors='ignore')
            df['Mileage'].fillna(df['Mileage'].mean(), inplace=True)
            df['Engine (CC)'].fillna(df['Engine (CC)'].mean(), inplace=True)
            df['car_age'] = 2024 - df['year']
            df.drop(columns=['year'], inplace=True, errors='ignore')
            df = pd.get_dummies(df, columns=['fuel', 'seller_type', 'transmission', 'owner'], drop_first=True)
            return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    else:
        st.warning("Please upload an Excel file to proceed.")
        return None

# Load the data
df = load_data()

if df is not None:
    # ---- SIDEBAR INPUTS FOR PREDICTION ----
    st.sidebar.header("Input Car Specifications for Prediction:")

    car_age = st.sidebar.slider("Car Age", 0, 20, 10)
    km_driven = st.sidebar.number_input("Kilometers Driven", 0, 300000, 50000)
    seats = st.sidebar.selectbox("Seats", [2, 4, 5, 7])
    max_power = st.sidebar.number_input("Max Power (in bph)", 50, 500, 100)
    mileage = st.sidebar.number_input("Mileage (kmpl)", 5.0, 35.0, 20.0)
    engine_cc = st.sidebar.number_input("Engine Capacity (CC)", 500, 5000, 1200)

    fuel_type = st.sidebar.selectbox("Fuel Type", ['Diesel', 'Petrol', 'LPG'])
    seller_type = st.sidebar.selectbox("Seller Type", ['Individual', 'Dealer', 'Trustmark Dealer'])
    transmission = st.sidebar.selectbox("Transmission", ['Manual', 'Automatic'])
    owner_type = st.sidebar.selectbox("Owner Type", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])

    # ---- MODEL TRAINING ----
    X = df.drop(columns=['selling_price', 'name'], errors='ignore')
    y = df['selling_price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # ---- PREDICTION ----
    try:
        fuel_Diesel = 1 if fuel_type == 'Diesel' else 0
        fuel_LPG = 1 if fuel_type == 'LPG' else 0
        fuel_Petrol = 1 if fuel_type == 'Petrol' else 0

        seller_type_Individual = 1 if seller_type == 'Individual' else 0
        seller_type_Trustmark_Dealer = 1 if seller_type == 'Trustmark Dealer' else 0

        transmission_Manual = 1 if transmission == 'Manual' else 0

        owner_Second_Owner = 1 if owner_type == 'Second Owner' else 0
        owner_Third_Owner = 1 if owner_type == 'Third Owner' else 0
        owner_Fourth_Above = 1 if owner_type == 'Fourth & Above Owner' else 0
        owner_Test_Drive = 1 if owner_type == 'Test Drive Car' else 0

        input_data = pd.DataFrame({
            'car_age': [car_age],
            'km_driven': [km_driven],
            'seats': [seats],
            'max_power (in bph)': [max_power],
            'Mileage': [mileage],
            'Engine (CC)': [engine_cc],
            'fuel_Diesel': [fuel_Diesel],
            'fuel_LPG': [fuel_LPG],
            'fuel_Petrol': [fuel_Petrol],
            'seller_type_Individual': [seller_type_Individual],
            'seller_type_Trustmark_Dealer': [seller_type_Trustmark_Dealer],
            'transmission_Manual': [transmission_Manual],
            'owner_Second Owner': [owner_Second_Owner],
            'owner_Third Owner': [owner_Third_Owner],
            'owner_Fourth & Above Owner': [owner_Fourth_Above],
            'owner_Test Drive Car': [owner_Test_Drive]
        })

        # Prediction
        prediction = model.predict(input_data)
        st.subheader(f"The predicted price of the car is: ₹{int(prediction[0]):,}")
    except ValueError as e:
        st.error(f"An error occurred during prediction: {str(e)}")

    # ---- MODEL EVALUATION ----
    # Making predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate R² score and MAE
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Display model performance
    st.markdown("### Model Performance Metrics")
    st.write(f"*R-squared (R²) Score:* {r2:.2f}")
    st.write(f"*Mean Absolute Error (MAE):* ₹{mae:,.0f}")
else:
    st.stop()

# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

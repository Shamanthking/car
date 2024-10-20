import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---- PAGE CONFIGURATION ----
st.set_page_config(page_title="Car Price Prediction & Analysis Dashboard", page_icon=":car:", layout="wide")

# ---- LOAD DATA FUNCTION ----
def load_data():
    uploaded_file = st.sidebar.file_uploader("Upload A Dataset (.xlsx)", type=["xlsx"])
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            
            # Data Preprocessing
            df.drop(columns=["Unnamed: 0", "Mileage Unit"], inplace=True, errors='ignore')  # Drop unnecessary columns
            df['Mileage'].fillna(df['Mileage'].mean(), inplace=True)  # Fill missing values with mean
            df['Engine (CC)'].fillna(df['Engine (CC)'].mean(), inplace=True)
            df['car_age'] = 2024 - df['year']  # Convert 'year' to 'car_age'
            df.drop(columns=['year'], inplace=True, errors='ignore')  # Drop 'year' after conversion
            df = pd.get_dummies(df, columns=['fuel', 'seller_type', 'transmission', 'owner'], drop_first=True)  # One-hot encoding
            return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    else:
        st.warning("Please upload an Excel file to proceed.")
        return None

# ---- LOAD THE DATA ----
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

    # ---- SIDEBAR FILTERS FOR VISUALIZATION ----
    st.sidebar.header("Please Filter Here:")

    car_brands = st.sidebar.multiselect("Select Car Brand:", options=df["name"].unique(), default=df["name"].unique())
    fuel = st.sidebar.multiselect("Select Fuel Type:", options=df["fuel"].unique(), default=df["fuel"].unique())
    seller_type = st.sidebar.multiselect("Select Seller Type:", options=df["seller_type"].unique(), default=df["seller_type"].unique())

    # Filter DataFrame
    df_selection = df.query("name == @car_brands & fuel == @fuel & seller_type == @seller_type")

    # ---- VISUALIZATION SECTION ----
    st.title(":car: Car Price Analysis Dashboard")

    # 1. Pie Chart for Seller Type Distribution
    st.markdown("### Seller Type Distribution")
    seller_type_distribution = df_selection["seller_type"].value_counts()
    fig_pie_seller_type = px.pie(values=seller_type_distribution.values, names=seller_type_distribution.index,
                                 title="<b>Distribution by Seller Type</b>",
                                 color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig_pie_seller_type, use_container_width=True)

    # 2. Pie Chart for Fuel Type Distribution
    st.markdown("### Fuel Type Distribution")
    fuel_distribution = df_selection["fuel"].value_counts()
    fig_pie_fuel_type = px.pie(values=fuel_distribution.values, names=fuel_distribution.index,
                               title="<b>Distribution by Fuel Type</b>",
                               color_discrete_sequence=px.colors.sequential.Blues)
    st.plotly_chart(fig_pie_fuel_type, use_container_width=True)

    # 3. Box Plot for Selling Price by Fuel Type
    st.markdown("### Box Plot of Selling Price by Fuel Type")
    fig_box_fuel_price = px.box(df_selection, x="fuel", y="selling_price", color="fuel",
                                title="<b>Selling Price Distribution by Fuel Type</b>",
                                labels={"fuel": "Fuel Type", "selling_price": "Selling Price (₹)"},
                                template="plotly_white")
    st.plotly_chart(fig_box_fuel_price, use_container_width=True)

    # 4. Box Plot for Selling Price by Car Brand
    st.markdown("### Box Plot of Selling Price by Car Brand")
    fig_box_brand_price = px.box(df_selection, x="name", y="selling_price", color="name",
                                 title="<b>Selling Price Distribution by Car Brand</b>",
                                 labels={"name": "Car Brand", "selling_price": "Selling Price (₹)"},
                                 template="plotly_white")
    st.plotly_chart(fig_box_brand_price, use_container_width=True)

    # 5. Scatter Plot for Engine Size vs. Selling Price
    st.markdown("### Scatter Plot: Engine Size vs. Selling Price")
    fig_engine_vs_price = px.scatter(df_selection, x="Engine (CC)", y="selling_price", color="fuel",
                                     title="<b>Engine Size vs. Selling Price</b>",
                                     labels={"Engine (CC)": "Engine Size (CC)", "selling_price": "Selling Price (₹)"},
                                     template="plotly_white")
    st.plotly_chart(fig_engine_vs_price, use_container_width=True)

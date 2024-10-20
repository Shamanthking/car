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
            df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')  # Drop unnecessary columns
            
            # Clean any columns that may contain units
            df['Mileage'] = df['Mileage'].str.replace('kmpl', '').astype(float)  # Remove 'kmpl' and convert to float
            df['Engine (CC)'] = df['Engine (CC)'].astype(float)  # Ensure 'Engine (CC)' is float
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
        # Create the input data structure matching the encoded columns
        input_data = pd.DataFrame({
            'car_age': [car_age],
            'km_driven': [km_driven],
            'seats': [seats],
            'max_power': [max_power],
            'Mileage': [mileage],
            'Engine (CC)': [engine_cc],
            'fuel_Diesel': [1 if fuel_type == 'Diesel' else 0],
            'fuel_LPG': [1 if fuel_type == 'LPG' else 0],
            'fuel_Petrol': [1 if fuel_type == 'Petrol' else 0],
            'seller_type_Individual': [1 if seller_type == 'Individual' else 0],
            'seller_type_Trustmark Dealer': [1 if seller_type == 'Trustmark Dealer' else 0],
            'transmission_Manual': [1 if transmission == 'Manual' else 0],
            'owner_Second Owner': [1 if owner_type == 'Second Owner' else 0],
            'owner_Third Owner': [1 if owner_type == 'Third Owner' else 0],
            'owner_Fourth & Above Owner': [1 if owner_type == 'Fourth & Above Owner' else 0],
            'owner_Test Drive Car': [1 if owner_type == 'Test Drive Car' else 0]
        })

        # Ensure input_data has the same columns as the trained model
        missing_cols = set(X_train.columns) - set(input_data.columns)
        for col in missing_cols:
            input_data[col] = 0
        
        input_data = input_data[X_train.columns]  # Align column order

        prediction = model.predict(input_data)
        st.write(f"Predicted Selling Price: {prediction[0]:,.2f}")
    
    except Exception as e:
        st.error(f"Error in prediction: {e}")
    
    # ---- DATA VISUALIZATION ----
    st.title(":bar_chart: Car Price Data Visualization")
    
    # 1. Pie Chart for Seller Type Distribution
    st.markdown("### Seller Type Distribution")
    seller_type_distribution = df['seller_type_Individual'].value_counts()
    fig_pie_seller_type = px.pie(values=seller_type_distribution.values, 
                                 names=seller_type_distribution.index,
                                 title="<b>Distribution by Seller Type</b>",
                                 color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig_pie_seller_type, use_container_width=True)

    # 2. Pie Chart for Fuel Type Distribution
    st.markdown("### Fuel Type Distribution")
    fuel_distribution = df[['fuel_Diesel', 'fuel_Petrol', 'fuel_LPG']].sum()
    fig_pie_fuel_type = px.pie(values=fuel_distribution.values, 
                               names=fuel_distribution.index,
                               title="<b>Distribution by Fuel Type</b>",
                               color_discrete_sequence=px.colors.sequential.Blues)
    st.plotly_chart(fig_pie_fuel_type, use_container_width=True)

    # 3. Box Plot for Selling Price by Fuel Type
    st.markdown("### Box Plot of Selling Price by Fuel Type")
    fuel_types = ['fuel_Diesel', 'fuel_Petrol', 'fuel_LPG']
    df['fuel'] = df[fuel_types].idxmax(axis=1)
    fig_box_fuel_price = px.box(df, x="fuel", y="selling_price", color="fuel",
                                title="<b>Selling Price Distribution by Fuel Type</b>",
                                labels={"fuel": "Fuel Type", "selling_price": "Selling Price (₹)"},
                                template="plotly_white")
    st.plotly_chart(fig_box_fuel_price, use_container_width=True)

    # 4. Scatter Plot for Engine Size vs. Selling Price
    st.markdown("### Scatter Plot: Engine Size vs. Selling Price")
    fig_engine_vs_price = px.scatter(df, x="Engine (CC)", y="selling_price", color="fuel",
                                     title="<b>Engine Size vs. Selling Price</b>",
                                     labels={"Engine (CC)": "Engine Size (CC)", "selling_price": "Selling Price (₹)"},
                                     template="plotly_white")
    st.plotly_chart(fig_engine_vs_price, use_container_width=True)

else:
    st.warning("Please upload a dataset to use the application.")

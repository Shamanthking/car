import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from PIL import Image

# ---- PAGE CONFIGURATION ----
st.set_page_config(page_title="Car Price Prediction & Analysis Dashboard", page_icon=":car:", layout="wide")

# Load the default CSV file
@st.cache
def load_data():
    df = pd.read_csv('data/car_data.csv')  # Path to default CSV
    df['car_age'] = 2024 - df['year']
    df.drop(columns=['year'], inplace=True, errors='ignore')
    df = pd.get_dummies(df, columns=['fuel', 'seller_type', 'transmission', 'owner'], drop_first=True)
    return df

# ---- PAGE NAVIGATION ----
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Predict", "Analysis", "Contact"])

    if page == "Home":
        show_home()
    elif page == "Predict":
        show_predict()
    elif page == "Analysis":
        show_analysis()
    elif page == "Contact":
        show_contact()

# ---- HOME PAGE ----
def show_home():
    st.markdown(
        """
        <style>
        .full-background {
            background-image: url("https://path/to/car_background.jpg");
            background-size: cover;
            background-position: center;
            height: 100vh;
            width: 100vw;
        }
        </style>
        <div class="full-background"></div>
        """,
        unsafe_allow_html=True,
    )
    st.title("Car Price Prediction & Analysis")
    st.subheader("Get accurate predictions on car prices and explore data insights.")
    if st.button("Get Car Price Prediction"):
        st.experimental_set_query_params(page="Predict")

# ---- PREDICT PAGE ----
def show_predict():
    st.title("Car Price Prediction")
    
    df = load_data()
    model, X_train, _, _, _ = train_model(df)

    car_age = st.slider("Car Age", 0, 20, 10)
    km_driven = st.number_input("Kilometers Driven", 0, 300000, 50000)
    seats = st.selectbox("Seats", [2, 4, 5, 7])
    max_power = st.number_input("Max Power (in bph)", 50, 500, 100)
    mileage = st.number_input("Mileage (kmpl)", 5.0, 35.0, 20.0)
    engine_cc = st.number_input("Engine Capacity (CC)", 500, 5000, 1200)
    fuel_type = st.selectbox("Fuel Type", ['Diesel', 'Petrol', 'LPG'])
    seller_type = st.selectbox("Seller Type", ['Individual', 'Dealer', 'Trustmark Dealer'])
    transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
    owner_type = st.selectbox("Owner Type", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])

    input_data = get_prediction_input(car_age, km_driven, seats, max_power, mileage, engine_cc, fuel_type, seller_type, transmission, owner_type)
    display_prediction(model, input_data, X_train)

# ---- ANALYSIS PAGE ----
def show_analysis():
    st.title("Data Analysis")
    df = load_data()

    st.sidebar.header("Filter Data for Visualization")
    selected_year = st.sidebar.selectbox("Select Year", options=["All"] + list(range(2000, 2025)), index=0)

    display_visualizations(df, selected_year)

# ---- CONTACT PAGE ----
def show_contact():
    st.title("Contact Us")
    st.write("For support or inquiries, reach out to us at support@carpriceapp.com")

# ---- HELPER FUNCTIONS ----
def train_model(df):
    X = df.drop(columns=['selling_price', 'name'], errors='ignore')
    y = df['selling_price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test

def get_prediction_input(car_age, km_driven, seats, max_power, mileage, engine_cc, fuel_type, seller_type, transmission, owner_type):
    input_data = pd.DataFrame({
        'car_age': [car_age],
        'km_driven': [km_driven],
        'seats': [seats],
        'max_power (in bph)': [max_power],
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
    return input_data

def display_prediction(model, input_data, X_train):
    try:
        missing_cols = set(X_train.columns) - set(input_data.columns)
        for col in missing_cols:
            input_data[col] = 0
        input_data = input_data[X_train.columns]
        prediction = model.predict(input_data)
        st.write(f"Predicted Selling Price: ₹ {prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")

def display_visualizations(df, selected_year):
    st.markdown("### Seller Type Distribution")
    if selected_year != "All":
        filtered_df = df[df['car_age'] == (2024 - selected_year)]
    else:
        filtered_df = df

    seller_type_distribution = filtered_df['seller_type_Individual'].value_counts()
    fig_pie_seller_type = px.pie(values=seller_type_distribution.values, 
                                 names=seller_type_distribution.index,
                                 title="Distribution by Seller Type",
                                 color_discrete_sequence=px.colors.sequential.Teal)
    st.plotly_chart(fig_pie_seller_type, use_container_width=True)

    st.markdown("### Fuel Type Distribution")
    fuel_distribution = filtered_df[['fuel_Diesel', 'fuel_Petrol', 'fuel_LPG']].sum()
    fig_pie_fuel_type = px.pie(values=fuel_distribution.values, 
                               names=fuel_distribution.index,
                               title="Distribution by Fuel Type",
                               color_discrete_sequence=px.colors.sequential.Plasma)
    st.plotly_chart(fig_pie_fuel_type, use_container_width=True)

    st.markdown("### Selling Price by Fuel Type")
    fuel_types = ['fuel_Diesel', 'fuel_Petrol', 'fuel_LPG']
    filtered_df['fuel'] = filtered_df[fuel_types].idxmax(axis=1)
    fig_box_fuel_price = px.box(filtered_df, x="fuel", y="selling_price", color="fuel",
                                title="Selling Price Distribution by Fuel Type",
                                template="plotly_white",
                                color_discrete_sequence=px.colors.diverging.RdBu)
    st.plotly_chart(fig_box_fuel_price, use_container_width=True)

    st.markdown("### Engine Size vs. Selling Price")
    fig_engine_vs_price = px.scatter(filtered_df, x="Engine (CC)", y="selling_price", color="fuel",
                                     title="Engine Size vs. Selling Price",
                                     template="plotly_white",
                                     color_discrete_sequence=px.colors.sequential.Teal)
    st.plotly_chart(fig_engine_vs_price, use_container_width=True)

if __name__ == "__main__":
    main()

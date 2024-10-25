import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---- PAGE CONFIGURATION ----
st.set_page_config(page_title="Car Price Prediction & Analysis Dashboard", page_icon=":car:", layout="wide")

# ---- LOAD DATA ----
@st.cache
def load_data():
    df = pd.read_csv('full data.csv', on_bad_lines='skip')  # Skips rows with bad lines
    df['car_age'] = 2024 - df['year']
    df.drop(columns=['year'], inplace=True, errors='ignore')
    df = pd.get_dummies(df, columns=['fuel', 'seller_type', 'transmission', 'owner'], drop_first=True)
    return df


# ---- MAIN PAGE NAVIGATION ----
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
    st.title("Car Price Prediction & Analysis")
    st.subheader("Get accurate predictions on car prices and explore data insights.")
    if st.button("Get Car Price Prediction"):
        st.experimental_set_query_params(page="Predict")

# ---- PREDICT PAGE ----
def show_predict():
    st.title("Car Price Prediction")
    
    df = load_data()
    model, X_train, _, _, _ = train_model(df)

    # User input for prediction
    car_age = st.slider("Car Age", 0, 20, 10)
    km_driven = st.number_input("Kilometers Driven", 0, 300000, 50000)
    seats = st.selectbox("Seats", [2, 4, 5, 7])
    max_power = st.number_input("Max Power (in bph)", 50, 500, 100)
    mileage = st.number_input("Mileage (kmpl)", 5.0, 35.0, 20.0)
    engine_cc = st.number_input("Engine Capacity (CC)", 500, 5000, 1200)
    brand = st.selectbox("Brand", df['brand'].unique())
    fuel_type = st.selectbox("Fuel Type", ['Diesel', 'Petrol', 'LPG'])
    seller_type = st.selectbox("Seller Type", ['Individual', 'Dealer', 'Trustmark Dealer'])
    transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
    owner_type = st.selectbox("Owner Type", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])

    # Preparing input data
    input_data = get_prediction_input(car_age, km_driven, seats, max_power, mileage, engine_cc, brand, fuel_type, seller_type, transmission, owner_type)
    display_prediction(model, input_data, X_train)

# ---- ANALYSIS PAGE ----
def show_analysis():
    st.title("Data Analysis")
    df = load_data()

    st.sidebar.header("Filter Data for Visualization")
    selected_brand = st.sidebar.selectbox("Select Brand", options=["All"] + list(df['brand'].unique()), index=0)
    selected_fuel = st.sidebar.selectbox("Select Fuel Type", options=["All", "Diesel", "Petrol", "LPG"])
    selected_seller_type = st.sidebar.selectbox("Select Seller Type", options=["All", "Individual", "Dealer", "Trustmark Dealer"])
    selected_transmission = st.sidebar.selectbox("Select Transmission", options=["All", "Manual", "Automatic"])

    if selected_brand != "All":
        df = df[df['brand'] == selected_brand]
    if selected_fuel != "All":
        df = df[df[f'fuel_{selected_fuel}'] == 1]
    if selected_seller_type != "All":
        df = df[df[f'seller_type_{selected_seller_type}'] == 1]
    if selected_transmission != "All":
        df = df[df[f'transmission_{selected_transmission}'] == 1]

    display_visualizations(df)

# ---- CONTACT PAGE ----
def show_contact():
    st.title("Contact Us")
    st.markdown("""
        - LinkedIn: [Your LinkedIn Profile](https://www.linkedin.com/in/your-profile)
        - Instagram: [Your Instagram Profile](https://www.instagram.com/your-profile)
        - Email: [youremail@example.com](mailto:youremail@example.com)
    """)

# ---- HELPER FUNCTIONS ----
def train_model(df):
    X = df.drop(columns=['selling_price', 'model'], errors='ignore')
    y = df['selling_price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test

def get_prediction_input(car_age, km_driven, seats, max_power, mileage, engine_cc, brand, fuel_type, seller_type, transmission, owner_type):
    input_data = pd.DataFrame({
        'car_age': [car_age],
        'km_driven': [km_driven],
        'seats': [seats],
        'max_power': [max_power],
        'mileage': [mileage],
        'engine': [engine_cc],
        f'brand_{brand}': [1],
        f'fuel_{fuel_type}': [1],
        f'seller_type_{seller_type}': [1],
        f'transmission_{transmission}': [1],
        f'owner_{owner_type}': [1]
    })
    return input_data

def display_prediction(model, input_data, X_train):
    missing_cols = set(X_train.columns) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0
    input_data = input_data[X_train.columns]
    prediction = model.predict(input_data)
    st.write(f"Predicted Selling Price: ₹ {prediction[0]:,.2f}")

def display_visualizations(df):
    st.markdown("## Selling Price Distributions")

    # Histogram of Selling Price
    fig_price_hist = px.histogram(df, x='selling_price', nbins=50, title="Distribution of Selling Prices", color_discrete_sequence=['#636EFA'])
    st.plotly_chart(fig_price_hist, use_container_width=True)

    # Box Plot of Selling Price by Brand
    fig_price_brand = px.box(df, x="brand", y="selling_price", title="Selling Price by Brand", color="brand", template="plotly_white")
    st.plotly_chart(fig_price_brand, use_container_width=True)

    # Scatter Plot of Engine Size vs Selling Price
    fig_engine_vs_price = px.scatter(df, x="engine", y="selling_price", color="fuel", title="Engine Size vs. Selling Price", template="plotly_white")
    st.plotly_chart(fig_engine_vs_price, use_container_width=True)

    # Pie Chart of Fuel Type Distribution
    fuel_distribution = df[['fuel_Diesel', 'fuel_Petrol', 'fuel_LPG']].sum()
    fig_fuel_type = px.pie(values=fuel_distribution.values, names=fuel_distribution.index, title="Fuel Type Distribution", color_discrete_sequence=px.colors.sequential.Teal)
    st.plotly_chart(fig_fuel_type, use_container_width=True)

    st.markdown("## Comparative Analysis")

    # Scatter Plot of Mileage vs Selling Price
    fig_mileage_vs_price = px.scatter(df, x="mileage", y="selling_price", color="fuel", title="Mileage vs. Selling Price", template="plotly_white")
    st.plotly_chart(fig_mileage_vs_price, use_container_width=True)

    # Bar Chart of Average Selling Price by Brand
    avg_price_brand = df.groupby('brand')['selling_price'].mean().reset_index()
    fig_avg_price_brand = px.bar(avg_price_brand, x='brand', y='selling_price', title="Average Selling Price by Brand", color="brand")
    st.plotly_chart(fig_avg_price_brand, use_container_width=True)
    
    # Heatmap of Feature Correlations
    corr_matrix = df.corr()
    fig_corr = px.imshow(corr_matrix, text_auto=True, title="Feature Correlation Heatmap", color_continuous_scale="RdBu")
    st.plotly_chart(fig_corr, use_container_width=True)


if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ---- PAGE CONFIGURATION ----
st.set_page_config(page_title="Car Price Prediction & Analysis Dashboard", page_icon=":car:", layout="wide")

# ---- CSS STYLING ----
def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url({image_url});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ---- LOAD DATA ----
@st.cache_data
def load_data():
    try:
        # Load the dataset
        df = pd.read_csv('data/cleaned_car_data_with_new_price (1).csv', on_bad_lines='skip')
        
        # Confirm the data loaded successfully
        if df is None or df.empty:
            st.error("The dataset is empty or could not be loaded.")
            return None
        
        # Calculate car age if 'Year' column exists
        if 'Year' in df.columns:
            df['car_age'] = 2024 - df['Year']
            df.drop(columns=['Year'], inplace=True)

        # Check for categorical columns before converting to dummy variables
        categorical_columns = ['Fuel_Type', 'Transmission', 'Owner_Type']
        available_columns = [col for col in categorical_columns if col in df.columns]
        
        if available_columns:
            df = pd.get_dummies(df, columns=available_columns, drop_first=True)
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None



# ---- MAIN PAGE NAVIGATION ----
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Analysis", "Contact"])
    
    if page == "Home":
        show_home()
    elif page == "Analysis":
        show_analysis()
    elif page == "Contact":
        show_contact()

# ---- HOME PAGE ----
def show_home():
    set_background("https://c4.wallpaperflare.com/wallpaper/612/507/644/lamborghini-monochrome-car-supercars-wallpaper-preview.jpg")
    st.title("Car Price Prediction & Analysis")
    st.subheader("Get accurate predictions on car prices and explore data insights.")
    
    # Navigation button to the Predict page
    if st.button("Get Car Price Prediction"):
        st.experimental_set_query_params(page="Predict")
        show_predict()

UNSPLASH_ACCESS_KEY = 'bbosDinjZlp3X5JAijT0QaJSHMfnEO860BXL5ltL_2M'

def get_car_image(brand):
    url = f"https://api.unsplash.com/search/photos?query={brand}&client_id={UNSPLASH_ACCESS_KEY}"
    response = requests.get(url)
    data = response.json()
    if data and data['results']:
        return data['results'][0]['urls']['regular']
    return None

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
    brand = st.selectbox("Brand", df['Name'].unique())  # Update this based on the dataset
    fuel_type = st.selectbox("Fuel Type", ['Diesel', 'Petrol', 'LPG'])
    seller_type = st.selectbox("Seller Type", ['Individual', 'Dealer', 'Trustmark Dealer'])
    transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
    owner_type = st.selectbox("Owner Type", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])
    
        # Fetch and display the brand image
    image_url = get_car_image(brand)
    if image_url:
        st.image(image_url, caption=f"{brand} - Representative Image", use_column_width=True)

    # Preparing input data
    input_data = get_prediction_input(car_age, km_driven, seats, max_power, mileage, engine_cc, brand, fuel_type, seller_type, transmission, owner_type)
    display_prediction(model, input_data, X_train)

# ---- ANALYSIS PAGE ----
def show_analysis():
    st.title("Data Analysis")
    df = load_data()

    st.sidebar.header("Filter Data for Visualization")
    selected_brand = st.sidebar.selectbox("Select Brand", options=["All"] + list(df['Name'].unique()), index=0)
    selected_fuel = st.sidebar.selectbox("Select Fuel Type", options=["All", "Diesel", "Petrol", "LPG"])
    selected_seller_type = st.sidebar.selectbox("Select Seller Type", options=["All", "Individual", "Dealer", "Trustmark Dealer"])
    selected_transmission = st.sidebar.selectbox("Select Transmission", options=["All", "Manual", "Automatic"])

    if selected_brand != "All":
        df = df[df['Name'] == selected_brand]
    if selected_fuel != "All" and f'Fuel_Type_{selected_fuel}' in df.columns:
        df = df[df[f'Fuel_Type_{selected_fuel}'] == 1]
    if selected_seller_type != "All" and f'seller_type_{selected_seller_type}' in df.columns:
        df = df[df[f'seller_type_{selected_seller_type}'] == 1]
    if selected_transmission != "All" and f'transmission_{selected_transmission}' in df.columns:
        df = df[df[f'transmission_{selected_transmission}'] == 1]

    display_visualizations(df)

# ---- CONTACT PAGE ----
def show_contact():
    st.title("Contact Us")
    st.markdown("""
        - [🔗 LinkedIn](https://www.linkedin.com/in/shamanth-m-05537b264?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app) 
        - [📸 Instagram](https://www.instagram.com/shamanth_m_/profilecard/?igsh=ZXdpbnppbjl1M3li) 
        - [✉ Email](mailto:shamanth2626@gmail.com)
    """)

# ---- HELPER FUNCTIONS ----
def train_model(df):
    # Check if the 'New_Price' column is present
    if 'New_Price' not in df.columns:
        st.error("The dataset does not contain the 'New_Price' column needed for predictions.")
        return None, None, None, None, None
    
    X = df.drop(columns=['New_Price'])
    y = df['New_Price']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model training
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test


def get_prediction_input(car_age, km_driven, seats, max_power, mileage, engine_cc, brand, fuel_type, seller_type, transmission, owner_type):
    input_data = pd.DataFrame({
        'car_age': [car_age],
        'Kilometers_Driven': [km_driven],
        'Seats': [seats],
        'Power': [max_power],
        'Mileage': [mileage],
        'Engine': [engine_cc],
        f'Name_{brand}': [1],
        f'Fuel_Type_{fuel_type}': [1],
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
    fig_price_hist = px.histogram(df, x='New_Price', nbins=50, title="Distribution of Selling Prices", color_discrete_sequence=['#636EFA'])
    st.plotly_chart(fig_price_hist, use_container_width=True)

    # Box Plot of Selling Price by Brand
    fig_price_brand = px.box(df, x="Name", y="New_Price", title="Selling Price by Brand", color="Name", template="plotly_white")
    st.plotly_chart(fig_price_brand, use_container_width=True)

    # Scatter Plot of Engine Size vs Selling Price
    fig_engine_vs_price = px.scatter(df, x="Engine", y="New_Price", color="Fuel_Type", title="Engine Size vs. Selling Price", template="plotly_white")
    st.plotly_chart(fig_engine_vs_price, use_container_width=True)

    # Pie Chart of Fuel Type Distribution
    fuel_distribution = df[['Fuel_Type_Diesel', 'Fuel_Type_Petrol', 'Fuel_Type_LPG']].sum()
    fig_fuel_type = px.pie(values=fuel_distribution.values, names=fuel_distribution.index, title="Fuel Type Distribution", color_discrete_sequence=px.colors.sequential.Teal)
    st.plotly_chart(fig_fuel_type, use_container_width=True)

    st.markdown("## Comparative Analysis")

    # Scatter Plot of Mileage vs Selling Price
    fig_mileage_vs_price = px.scatter(df, x="Mileage", y="New_Price", color="Fuel_Type", title="Mileage vs. Selling Price", template="plotly_white")
    st.plotly_chart(fig_mileage_vs_price, use_container_width=True)

    # Bar Chart of Average Selling Price by Brand
    avg_price_brand = df.groupby('Name')['New_Price'].mean().reset_index()
    fig_avg_price_brand = px.bar(avg_price_brand, x='Name', y='New_Price', title="Average Selling Price by Brand", color="Name")
    st.plotly_chart(fig_avg_price_brand, use_container_width=True)
    
    # Heatmap of Feature Correlations
    corr_matrix = df.corr()
    fig_corr = px.imshow(corr_matrix, 
                         text_auto=True, 
                         title="Feature Correlation Heatmap", 
                         color_continuous_scale="RdBu")
    st.plotly_chart(fig_corr, use_container_width=True)

if __name__ == "__main__":  # Fixed line for running the main function
    main()

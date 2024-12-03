import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

# ---- PAGE CONFIGURATION ----
st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="wide")

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

# ---- LOAD DATA ----
@st.cache_data
def load_data():
    """Loads and preprocesses the preprocessed car dataset."""
    try:
        file_path = 'preprocessed_car_dekho_data.csv'
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# ---- HOMEPAGE ----
def show_home():
    st.title("Car Price Prediction Web Application")
    st.subheader("Predict the price of used cars based on various features")
    st.write("""
        This Web Application is designed to help users estimate the price of used cars based on features like car age, kilometers driven, fuel type, and more.
        By leveraging machine learning models, we provide predictions to assist users in making informed decisions when buying or selling used cars.
    """)

# ---- TEAM SECTION ----
def show_team():
    st.title("Our Team")
    st.write("Meet the dedicated contributors who developed this application:")
    st.write("""
    - *Deekshith N:* 4AD22CI009
    - *Prashanth Singh H S:* 4AD22CI040
    - *Shamanth M:* 4AD22CI047
    - *Akash A S:* 4AD22CI400
    """)
    st.balloons()

# ---- PREDICTION PAGE ----
def show_prediction():
    st.title("Car Price Prediction")
    df = load_data()
    if df is not None:
        # Input fields for prediction
        car_age = st.slider("Car Age", int(df['car_age'].min()), int(df['car_age'].max()), 5)
        km_driven = st.number_input("Kilometers Driven", int(df['km_driven'].min()), int(df['km_driven'].max()), 50000)
        seats = st.selectbox("Seats", sorted(df['seats'].unique()))
        max_power = st.number_input("Max Power (in bhp)", float(df['max_power'].min()), float(df['max_power'].max()), 100.0)
        mileage = st.number_input("Mileage (kmpl)", float(df['mileage'].min()), float(df['mileage'].max()), 20.0)
        engine_cc = st.number_input("Engine Capacity (CC)", float(df['engine_cc'].min()), float(df['engine_cc'].max()), 1200.0)
        brand = st.selectbox("Brand", df['brand'].unique())
        fuel_type = st.selectbox("Fuel Type", df['fuel_type'].unique())
        seller_type = st.selectbox("Seller Type", df['seller_type'].unique())
        transmission = st.selectbox("Transmission", df['transmission'].unique())

        # Prepare input data for prediction
        input_data = {
            'car_age': car_age,
            'km_driven': km_driven,
            'seats': seats,
            'max_power': max_power,
            'mileage': mileage,
            'engine_cc': engine_cc,
            'brand': brand,
            'fuel_type': fuel_type,
            'seller_type': seller_type,
            'transmission': transmission
        }
        user_data = pd.DataFrame([input_data])
        user_data = pd.get_dummies(user_data).reindex(columns=df.drop(columns=['selling_price']).columns, fill_value=0)

        # Train and predict using Random Forest model
        X = df.drop(columns=['selling_price'])
        y = df['selling_price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predicted_price = model.predict(user_data)
        st.write(f"### Predicted Selling Price: ₹{predicted_price[0]:,.2f}")

# ---- DATA ANALYSIS ----
def show_analysis():
    st.title("Detailed Data Analysis")
    st.write("Explore the data insights to understand car price trends.")
    df = load_data()
    if df is not None:
        st.subheader("Brand Distribution")
        brand_counts = df['brand'].value_counts()
        fig = px.bar(brand_counts, x=brand_counts.index, y=brand_counts.values, labels={'x': 'Brand', 'y': 'Count'})
        st.plotly_chart(fig)

        st.subheader("Fuel Type Distribution")
        fuel_counts = df['fuel_type'].value_counts()
        fig = px.pie(fuel_counts, names=fuel_counts.index, values=fuel_counts.values)
        st.plotly_chart(fig)

# ---- MODEL COMPARISON ----
def show_model_comparison():
    st.title("Model Comparison")
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

# ---- FEEDBACK & CONTACT ----
def show_feedback_contact():
    st.title("We Value Your Feedback!")
    rating = st.selectbox("Rate Us:", ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"], index=4)
    feedback = st.text_area("Questions or suggestions? Let us know.")
    
    if st.button("Submit"):
        feedback_data = {
            "rating": rating,
            "feedback": feedback
        }
        st.write("Thank you for your feedback!")
        st.json(feedback_data)

    st.subheader("Contact Us")
    st.write("If you have further questions or require assistance, reach out at:")
    st.write("Email: support@carpredictionapp.com")
    st.write("Phone: +123-456-7890")

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

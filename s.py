import streamlit as st
import pandas as pd
import numpy as np
import pickle as pk
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from openpyxl import Workbook
from sklearn.preprocessing import LabelEncoder
import os
import statsmodels.api as sm

# ---- HOME PAGE ----
def show_home(df):
    st.title("Welcome to the Car Price Prediction App 🚗")

    # ---- CUSTOM CSS FOR BACKGROUND ----
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("https://img.freepik.com/free-vector/lightened-luxury-sedan-car-against-night-city-with-headlamps-rear-tail-lights-lit_1284-28804.jpg");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        color: white;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.write("""
    This application leverages the power of **machine learning** to analyze car features, uncover insights, 
    and predict car prices with ease. Whether you're a car dealer, buyer, or data enthusiast, this tool 
    is designed to provide you with actionable insights and accurate predictions.
    """)
    st.subheader("📖 How to Use This App:")
    st.markdown("""
    1. **Explore and Analyze Data**  
       - Dive into the dataset with **interactive visualizations** and **metrics**:
       - Understand trends in car features like mileage, engine size, and brand popularity.
       - Identify key factors that influence car prices.
    
    2. **Predict Selling Prices**  
       - Provide the required details such as car age, mileage, and engine specifications.  
       - Instantly predict the expected selling price using powerful machine learning models.
    
    3. **Compare Machine Learning Models**  
       - Evaluate multiple models, including Random Forest, Gradient Boosting, and Linear Regression, 
         to see which performs best on your data.
    
    4. **Leave Feedback**  
       - Share your experience with the app to help us improve!
    """)

    # Display initial insights
    st.subheader("Dataset Overview")
    st.write(df.head())
    st.write(f"Number of records: {df.shape[0]} | Number of features: {df.shape[1]}")

# ---- LOAD DATA FUNCTION ----
def load_data():
    try:
        df = pd.read_csv('data/Processed_Cardetails.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# ---- PREDICTION PAGE FUNCTION ----
def show_prediction(df):
    st.title("Car Price Prediction 🚗")
    
    # Load the pre-trained model
    try:
        model = pk.load(open('sham.pkl', 'rb'))
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return

    # Define car brands and their corresponding models
    car_data = {
        'Ambassador': ['Grand', 'Classic'],
        'Chevrolet': ['Spark', 'Aveo', 'Sail', 'Cruze', 'Optra', 'Beat'],
        'Daewoo': ['Matiz'],
        'Datsun': ['GO', 'RediGO'],
        'Fiat': ['Linea', 'Grande Punto', 'Punto', 'Avventura'],
        'Ford': ['Freestyle', 'Fusion', 'EcoSport', 'Ikon', 'Aspire', 'Classic', 'Figo', 'Fiesta'],
        'Honda': ['Civic', 'Amaze', 'City', 'Jazz', 'WR-V', 'Brio'],
        'Hyundai': ['i20', 'i10', 'Elantra', 'Getz', 'Venue', 'EON', 'Accent', 'Verna', 'Grand i10', 'Xcent', 'Creta', 'Sonata', 'Santro', 'Elite'],
        'Kia': ['Seltos'],
        'Mahindra': ['Logan', 'KUV 100', 'XUV300', 'Verito'],
        'Mahindra Renault': ['Logan'],
        'Maruti': ['A-Star', 'Omni', 'Eeco', '800', 'Ciaz', 'S-Presso', 'Baleno', 'Alto', 'Esteem', 'S-Cross', 'Wagon R', 'Ignis', 'Zen', 'Vitara Brezza', 'Swift', 'SX4', 'Celerio', 'Ritz', 'Dzire'],
        'Mercedes-Benz': ['B Class'],
        'Mitsubishi': ['Lancer'],
        'Nissan': ['Micra', 'Kicks', 'Sunny', 'Terrano'],
        'Opel': ['Astra'],
        'Renault': ['KWID', 'Fluence', 'Koleos', 'Duster', 'Pulse', 'Captur', 'Scala'],
        'Skoda': ['Laura', 'Octavia', 'Rapid', 'Superb', 'Fabia'],
        'Tata': ['Bolt', 'Tiago', 'Nexon', 'Zest', 'Tigor', 'Manza', 'Indigo', 'Indica'],
        'Toyota': ['Glanza', 'Platinum Etios', 'Yaris', 'Etios', 'Corolla'],
        'Volkswagen': ['Jetta', 'Ameo', 'CrossPolo', 'Passat', 'Polo', 'Vento'],
        'Volvo': ['V40'],
        'Other': ['Other']
    }

   # Brand selection
    brand = st.selectbox('Select Car Brand', list(car_data.keys()))

    # Model selection based on the selected brand
    model_options = car_data.get(brand, [])
    selected_model = st.selectbox('Select Car Model', model_options)

    # Additional input fields (same as in your original code)
    Fuel = st.selectbox('Select Fuel Type', ['Diesel', 'Petrol', 'CNG', 'LPG', 'Other'])
    Seller = st.selectbox('Select Type of Seller', ['Individual', 'Dealer', 'Trustmark Dealer', 'Other'])
    Transmission = st.selectbox('Select Car Transmission', ['Manual', 'Automatic', 'Other'])
    Owner = st.selectbox('Select Present Car Owner Type', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Other'])
    Year = st.text_input('Year')
    Km_driven = st.text_input('Km driven')
    engine = st.text_input('Engine')
    max_power = st.text_input('Max Power')
    mileage_kmpl = st.text_input('Mileage (kmpl)')

    if st.button('Predict'):
        # Validate numeric input
        def validate_numeric_input(input_value, field_name):
            try:
                return float(input_value)
            except ValueError:
                st.error(f"Invalid input for {field_name}. Please enter a numeric value.")
                return None

        # Validate all numeric inputs
        Year = validate_numeric_input(Year, 'Year')
        Km_driven = validate_numeric_input(Km_driven, 'Km driven')
        engine = validate_numeric_input(engine, 'Engine')
        max_power = validate_numeric_input(max_power, 'Max Power')
        mileage_kmpl = validate_numeric_input(mileage_kmpl, 'Mileage (kmpl)')

        # Proceed only if all inputs are valid
        if None not in [Year, Km_driven, engine, max_power, mileage_kmpl]:
            try:
                input_data = pd.DataFrame([{
                    'year': Year,
                    'km_driven': Km_driven,
                    'fuel': Fuel,
                    'seller_type': Seller,
                    'transmission': Transmission,
                    'owner': Owner,
                    'engine': engine,
                    'max_power': max_power,
                    'brand': brand,
                    'model': selected_model,
                    'mileage_kmpl': mileage_kmpl
                }])

                # Ensure preprocessing is done (e.g., encoding categorical features, etc.)
                le = LabelEncoder()
                input_data['fuel'] = le.fit_transform(input_data['fuel'])
                input_data['seller_type'] = le.fit_transform(input_data['seller_type'])
                input_data['transmission'] = le.fit_transform(input_data['transmission'])
                input_data['owner'] = le.fit_transform(input_data['owner'])
                input_data['brand'] = le.fit_transform(input_data['brand'])
                input_data['model'] = le.fit_transform(input_data['model'])

                # Predict the price
                prediction = model.predict(input_data)
                output = round(prediction[0] * 19.61, -3)  # Assuming price is predicted in a different unit, so converting
                formatted_output = "{:,.0f}".format(output)
                st.success(f'You can sell your car for {formatted_output} INR')
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")


# ---- DATA ANALYSIS ----
def show_analysis(df):
    st.header("📊 Detailed Data Analysis")
    if df is not None:
        st.write("""
        Explore the dataset through a variety of visualizations and analyses to gain deeper insights into 
        the factors affecting car prices. Each visualization is explained for better interpretation.
        """)

        # 1. Brand Distribution
        st.subheader("🔍 Brand Distribution")
        st.write("This bar chart shows the count of cars available for each brand in the dataset.")
        brand_counts = df['brand'].value_counts()
        fig = px.bar(brand_counts, x=brand_counts.index, y=brand_counts.values, 
                     labels={'x': 'Brand', 'y': 'Count'}, title="Brand Distribution")
        st.plotly_chart(fig)

        # Other visualizations would go here (e.g., Fuel Type, Price Distribution, etc.)

# ---- MODEL COMPARISON ----
def show_model_comparison(df):
    st.header("Model Comparison")
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
        st.dataframe(metrics_df)

# ---- TEAM PAGE ----
def show_team():
    st.title("Meet the Team")
    st.write(""" 
    - *Deekshith N:* 4AD22CI009 
    - *Prashanth Singh H S:* 4AD22CI040 
    - *Shamanth M:* 4AD22CI047 
    - *Akash A S:* 4AD22CI400 
    """)
    st.balloons()

# ---- FEEDBACK & CONTACT PAGE ----
def show_feedback_and_contact():
    st.title("Feedback & Contact")

    # Feedback Form
    st.subheader("We'd love to hear your feedback!")
    rating = st.selectbox("Rate Us:", ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"], index=4)
    feedback = st.text_area("Share your suggestions or comments:")

    if st.button("Submit Feedback"):
        try:
            # Load existing feedback data
            feedback_file = 'feedback.xlsx'
            if os.path.exists(feedback_file):
                feedback_df = pd.read_excel(feedback_file, engine="openpyxl")
            else:
                feedback_df = pd.DataFrame(columns=["rating", "comments"])

            # Append new feedback
            new_feedback = pd.DataFrame([[rating, feedback]], columns=["rating", "comments"])
            with pd.ExcelWriter(feedback_file, engine="openpyxl", mode="a") as writer:
                feedback_df = pd.concat([feedback_df, new_feedback], ignore_index=True)
                feedback_df.to_excel(writer, index=False)

            st.success("Thank you for your feedback!")
        except Exception as e:
            st.error(f"Error saving feedback: {e}")

    # Contact Information
    st.subheader("Contact Information")
    st.write("For inquiries, reach out to us at:")
    st.write("Email: example@example.com")
    st.write("Phone: 123-456-7890")

# ---- MAIN FUNCTION ----
def main():
    df = load_data()
    if df is not None:
        menu = ["Home", "Prediction", "Data Analysis", "Model Comparison", "Team", "Feedback & Contact"]
        choice = st.sidebar.selectbox("Menu", menu)

        if choice == "Home":
            show_home(df)
        elif choice == "Prediction":
            show_prediction(df)
        elif choice == "Data Analysis":
            show_analysis(df)
        elif choice == "Model Comparison":
            show_model_comparison(df)
        elif choice == "Team":
            show_team()
        elif choice == "Feedback & Contact":
            show_feedback_and_contact()

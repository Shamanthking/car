import streamlit as st
import pandas as pd
import numpy as np
import pickle as pk
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
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



# ---- PAGE CONFIGURATION ----
st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="wide")

# ---- FILE SETUP ----
users_file = 'users.xlsx'
feedback_file = 'feedback.xlsx'

# Helper function to create Excel files if they do not exist
def create_empty_excel(file_name, columns):
    wb = Workbook()
    ws = wb.active
    ws.append(columns)
    wb.save(file_name)

# Ensure required files exist
for file, columns in [(users_file, ["username", "email", "password"]),
                      (feedback_file, ["rating", "comments"])]:
    if not os.path.exists(file):
        create_empty_excel(file, columns)
    else:
        try:
            pd.read_excel(file, engine="openpyxl")
        except Exception:
            os.remove(file)
            create_empty_excel(file, columns)

# ---- AUTHENTICATION ----
def add_user(username, email, password):
    try:
        # Check if users.xlsx exists
        if os.path.exists(users_file):
            # Load existing data
            users_df = pd.read_excel(users_file, engine="openpyxl")
        else:
            # Create an empty DataFrame with the required columns
            users_df = pd.DataFrame(columns=["username", "email", "password"])

        # Check if username or email already exists
        if (users_df['username'] == username).any():
            st.sidebar.error("This username is already taken.")
            return
        if (users_df['email'] == email).any():
            st.sidebar.error("This email is already registered.")
            return

        # Append new user data
        new_user = pd.DataFrame([[username, email, password]], columns=["username", "email", "password"])
        updated_users_df = pd.concat([users_df, new_user], ignore_index=True)

        # Save back to the Excel file
        updated_users_df.to_excel(users_file, index=False, engine="openpyxl")
        st.sidebar.success("User registered successfully. Please log in.")

    except Exception as e:
        st.sidebar.error(f"Error while registering user: {e}")


def authenticate_user():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    st.sidebar.title("Authentication")
    auth_option = st.sidebar.radio("Choose Option", ["Login", "Register"], key="auth_option")

    if auth_option == "Login":
        username = st.sidebar.text_input("Username", key="login_username")
        password = st.sidebar.text_input("Password", type="password", key="login_password")
        if st.sidebar.button("Login"):
            try:
                if os.path.exists(users_file):
                    users_df = pd.read_excel(users_file, engine="openpyxl")
                else:
                    st.sidebar.error("No users found. Please register first.")
                    return

                # Validate credentials
                user = users_df[(users_df['username'] == username) & (users_df['password'] == password)]
                if not user.empty:
                    st.sidebar.success(f"Welcome, {username}!")
                    st.session_state.authenticated = True
                    st.session_state.username = username
                else:
                    st.sidebar.error("Invalid username or password.")
            except Exception as e:
                st.sidebar.error(f"Error while authenticating: {e}")

    elif auth_option == "Register":
        new_username = st.sidebar.text_input("Create Username", key="register_username")
        email = st.sidebar.text_input("Email", key="register_email")
        new_password = st.sidebar.text_input("Create Password", type="password", key="register_password")
        confirm_password = st.sidebar.text_input("Confirm Password", type="password", key="confirm_password")
        if st.sidebar.button("Register"):
            if new_password == confirm_password:
                add_user(new_username, email, new_password)
            else:
                st.sidebar.error("Passwords do not match.")

    return st.session_state.authenticated


def save_to_excel(df, file_name):
    try:
        df.to_excel(file_name, index=False, engine="openpyxl")
    except Exception as e:
        st.error(f"Error while saving data: {e}. Retrying...")
        if os.path.exists(file_name):
            os.remove(file_name)  # Delete the file if corrupted
        df.to_excel(file_name, index=False, engine="openpyxl")





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
        df['brand'] = df['name'].apply(get_brand_name)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# ---- PREDICTION PAGE FUNCTION ----
def show_prediction(df):
    st.title("Car Price Prediction 🚗")
    model = pk.load(open('ProcessedCar.pkl', 'rb'))

    # ---- USER INPUT FORM ----
    db['name'] = db['name'].apply(get_brand_name)
    name = st.selectbox("Select Car Brand", df['brand'].unique())
    year = st.slider("Select Manufacture Year", 1994, 2024)
    km_driven = st.slider("Kilometers Driven", 11, 200000)
    fuel = st.selectbox("Fuel Type", df['fuel'].unique())
    seller_type = st.selectbox("Type of Seller", df['seller_type'].unique())
    mileage = st.slider("Car Mileage (km/l)", 10, 40)
    owner = st.selectbox("Type of Owner", df['owner'].unique())
    engine = st.slider("Engine Capacity (CC)", 700, 5000)
    max_power = st.slider("Max Power (BHP)", 0, 200)
    transmission = st.selectbox("Type of Transmission", df['transmission'].unique())
    seats = st.slider("Number of Seats", 2, 10)

    if st.button("Predict"):
  input_data=pd.DataFrame([[name,year,km_driven,fuel,seller_type,transmission,owner,mileage,engine,max_power,seats]],columns=['name','year','km_driven','fuel','seller_type','transmission','owner','mileage','engine','max_power','seats'])
  st.write(input_data)
  input_data['owner'].replace(['First Owner', 'Second Owner', 'Third Owner',
       'Fourth & Above Owner', 'Test Drive Car'],[1,2,3,4,5],inplace=True)
  input_data['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'],[1,2,3,4],inplace=True)
  input_data['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'],[1,2,3],inplace=True)
  input_data['transmission'].replace(['Manual',"Automatic"],[1,2],inplace=True)
  input_data['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai' ,'Toyota' ,'Ford' ,'Renault' ,'Mahindra',
 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz', 'Mitsubishi', 'Audi',
 'Volkswagen', 'BMW', 'Nissan', 'Lexus', 'Jaguar','Land' ,'MG' ,'Volvo', 'Daewoo',
 'Kia', 'Fiat', 'Force', 'Ambassador', 'Ashok', 'Isuzu' ,'Opel'],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],inplace=True)

        # Predict the car price
        car_price = model.predict(input_data)
        st.write(f"### Predicted Car Price: ₹{int(car_price[0]):,} INR")



# ---- DATA ANALYSIS ----
def show_analysis(df):
    st.header("📊 Detailed Data Analysis")
    df = load_data()
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

        # 2. Fuel Type Distribution
        st.subheader("⛽ Fuel Type Distribution")
        st.write("A pie chart illustrating the distribution of cars by fuel type (e.g., Petrol, Diesel, CNG).")
        fuel_counts = df['fuel_type'].value_counts()
        fig = px.pie(fuel_counts, values=fuel_counts.values, names=fuel_counts.index, 
                     title="Fuel Type Distribution", hole=0.4)
        st.plotly_chart(fig)

        # 3. Distribution of Car Prices
        st.subheader("💰 Distribution of Car Prices")
        st.write("This histogram shows the distribution of car prices, helping identify common price ranges.")
        fig = px.histogram(df, x='selling_price', nbins=50, title="Price Distribution", color_discrete_sequence=['#636EFA'])
        st.plotly_chart(fig)

        # 4. Box Plot for Price by Transmission Type
        st.subheader("🚗 Price by Transmission Type")
        st.write("A box plot showing how car prices vary between manual and automatic transmissions.")
        fig = px.box(df, x='transmission_type', y='selling_price', title="Price Distribution by Transmission Type")
        st.plotly_chart(fig)

        # 5. Scatter Plot - Price vs Mileage
        st.subheader("📈 Price vs Mileage")
        st.write("A scatter plot displaying the relationship between mileage and selling price. A trendline is included to identify patterns.")
        fig = px.scatter(df, x='mileage', y='selling_price', trendline="ols", title="Price vs. Mileage")
        st.plotly_chart(fig)

        # 6. Heatmap of Correlation Matrix
        st.subheader("🔗 Correlation Heatmap")
        st.write("This heatmap shows the correlation between numerical features. Strong positive or negative correlations are highlighted.")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # 7. Line Plot - Average Price by Car Age
        st.subheader("📅 Average Price by Car Age")
        st.write("A line chart showing how the average selling price changes with the age of the car.")
        if 'vehicle_age' in df.columns:
            age_price = df.groupby('vehicle_age')['selling_price'].mean().reset_index()
            fig = px.line(age_price, x='vehicle_age', y='selling_price', title="Average Price by Car Age", markers=True)
            st.plotly_chart(fig)

        # 8. Violin Plot for Price by Seller Type
        st.subheader("🛍️ Price by Seller Type")
        st.write("A violin plot illustrating the distribution of car prices based on seller type, with box plot overlays.")
        fig = px.violin(df, x='seller_type', y='selling_price', box=True, title="Price Distribution by Seller Type")
        st.plotly_chart(fig)

        # 9. Average Mileage by Fuel Type
        st.subheader("⚡ Average Mileage by Fuel Type")
        st.write("A bar chart showing the average mileage for each fuel type. Useful for identifying efficiency trends.")
        if 'mileage' in df.columns and 'fuel_type' in df.columns:
            mileage_fuel = df.groupby('fuel_type')['mileage'].mean().reset_index()
            fig = px.bar(mileage_fuel, x='fuel_type', y='mileage', color='fuel_type', 
                         title="Average Mileage by Fuel Type", labels={'mileage': 'Average Mileage (kmpl)', 'fuel_type': 'Fuel Type'})
            st.plotly_chart(fig)

        # 10. Distribution of Engine Size
        st.subheader("🏎️ Distribution of Engine Size")
        st.write("A histogram showing the distribution of engine capacities across cars in the dataset.")
        fig = px.histogram(df, x='engine', nbins=50, title="Engine Size Distribution", color_discrete_sequence=['#FFA15A'])
        st.plotly_chart(fig)

        # 11. Price vs Engine Size
        st.subheader("⚙️ Price vs Engine Size")
        st.write("A scatter plot highlighting the relationship between engine capacity and selling price. Trendline included for clarity.")
        fig = px.scatter(df, x='engine', y='selling_price', trendline="ols", 
                         title="Price vs. Engine Size", labels={'engine': 'Engine Size (CC)', 'selling_price': 'Selling Price'})
        st.plotly_chart(fig)


# ---- MODEL COMPARISON ----
def show_model_comparison():
    st.header("Model Comparison")
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
            if os.path.exists(feedback_file):
                feedback_df = pd.read_excel(feedback_file, engine="openpyxl")
            else:
                feedback_df = pd.DataFrame(columns=["rating", "comments"])

            # Append new feedback
            new_feedback = pd.DataFrame([[rating, feedback]], columns=["rating", "comments"])
            with pd.ExcelWriter(feedback_file, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
                new_feedback.to_excel(writer, index=False, header=False, startrow=len(feedback_df) + 1)
            st.success("Thank you for your feedback!")

        except Exception as e:
            st.error(f"Error while saving feedback: {e}")

    # Contact Information
    st.subheader("Contact Us")
    st.write("""
    If you have any questions or need support, feel free to reach out to us:

    - 📧 **Email**: shamanth2626@gmail.com  
    - 📞 **Phone**: +8660847706  
    - 🌐 **Website**: [www.carpriceprediction.com](https://q8pptv2nhseudi6hdkzzc3.streamlit.app)
    """)

    # Social Media Links
    st.write("Follow us on:")
    st.markdown("""
    - [LinkedIn](https://www.linkedin.com/in/shamanth-m-05537b264)🖇  
    - [Instagram](https://www.instagram.com/shamanth_m_) 📸
    """)



# ---- MAIN APP ----
if "df" not in st.session_state:
    st.session_state.df = load_data()

if authenticate_user():
    st.sidebar.title("Menu")
    menu = st.sidebar.radio("Select a page:", ["Home", "Prediction", "Model Comparission", "Analysis", "Team", "Feedback"])

    if st.session_state.df is not None:
        if menu == "Home":
            show_home(st.session_state.df)
        elif menu == "Prediction":
            show_prediction(st.session_state.df)
        elif menu == "Analysis":
            show_analysis(st.session_state.df)
        elif menu == "Model Comparission":
            show_model_comparison()
        elif menu == "Team":
            show_team()
        elif menu == "Feedback":
            show_feedback_and_contact()
    

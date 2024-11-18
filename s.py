import streamlit as st
import pandas as pd
import numpy as np
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
import os

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



# ---- DATA LOADING ----
@st.cache_data
def load_data():
    """Loads and preprocesses the car dataset from a fixed path."""
    try:
        file_path = 'data/carr.csv'
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

        # Encode categorical features
        cat_cols = df.select_dtypes(include=['object']).columns
        df[cat_cols] = df[cat_cols].apply(LabelEncoder().fit_transform)

        # Impute missing values
        imputer = SimpleImputer(strategy="mean")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


# ---- HOME PAGE ----
def show_home(df):
    st.title("Welcome to the Car Price Prediction App 🚗")

    # ---- CUSTOM CSS FOR BACKGROUND ----
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("https://w0.peakpx.com/wallpaper/440/206/HD-wallpaper-black-background-car-cars-vehicles.jpg");
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

# ---- PREDICTION PAGE ----
def show_prediction(df):
    st.header("Car Price Prediction")
    if df is not None:
        car_age = st.slider("Car Age", 0, 20, 10)
        km_driven = st.number_input("Kilometers Driven", 0, 300000, 50000)
        seats = st.selectbox("Seats", [2, 4, 5, 7])
        max_power = st.number_input("Max Power (in bhp)", 50, 500, 100)
        mileage = st.number_input("Mileage (kmpl)", 5.0, 35.0, 20.0)
        engine_cc = st.number_input("Engine Capacity (CC)", 500, 5000, 1200)
        brand = st.selectbox("Brand", df['brand'].unique())
        fuel_type = st.selectbox("Fuel Type", ['Diesel', 'Petrol', 'LPG'])
        seller_type = st.selectbox("Seller Type", ['Individual', 'Dealer', 'Trustmark Dealer'])
        transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])

        X = df.drop(columns=['selling_price'])
        y = df['selling_price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        user_data = pd.DataFrame({
            'car_age': [car_age],
            'km_driven': [km_driven],
            'seats': [seats],
            'max_power': [max_power],
            'mileage': [mileage],
            'engine_cc': [engine_cc],
        })

        # One-hot encoding for the categorical features
        categorical_features = pd.DataFrame({'brand': [brand], 'fuel_type': [fuel_type], 'seller_type': [seller_type], 'transmission': [transmission]})
        categorical_encoded = pd.get_dummies(categorical_features, drop_first=True)
        user_data = pd.concat([user_data, categorical_encoded], axis=1)
        user_data = user_data.reindex(columns=X.columns, fill_value=0)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predicted_price = model.predict(user_data)
        st.write(f"### Predicted Selling Price: ₹{predicted_price[0]:,.2f}")

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
if authenticate_user():
    st.sidebar.title("Menu")
    menu = st.sidebar.radio("Select a page:", ["Home", "Prediction", "Analysis", "Team", "Feedback"])

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
    else:
        st.error("Data could not be loaded. Please check the dataset.")


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

# ---- PAGE CONFIGURATION ----
st.set_page_config(page_title="Car Price Prediction & Analysis Dashboard", page_icon="🚗", layout="wide")

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
    """Loads and preprocesses the car dataset from a fixed path."""
    try:
        # Specify the fixed path to the dataset file
        file_path = '/data/carr.csv'
        
        # Load data from the dataset file
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        
        # Display initial columns and data types
        st.write("Initial Columns:", df.columns)
        st.write("Data Types:", df.dtypes)
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

        # Impute missing values for numeric columns
        imputer = SimpleImputer(strategy="mean")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

        # Display preprocessed dataset
        st.write("Preprocessed Data Sample:", df.head())
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# ---- PAGE SECTIONS ----
def show_home():
    st.title("Car Price Prediction & Analysis")
    st.subheader("Get predictions and insights into car price data with multiple model comparisons.")

def show_prediction():
    st.header("Car Price Prediction")

    df = load_data()
    if df is not None:
        # Define feature columns and target variable
        X = df.drop(columns=['selling_price'])
        y = df['selling_price']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define models
        models = {
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "Linear Regression": LinearRegression(),
            "K-Neighbors Regressor": KNeighborsRegressor(n_neighbors=5),
            "Decision Tree": DecisionTreeRegressor(random_state=42)
        }

        # Model training and evaluation
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            st.write(f"{model_name}:")
            st.write(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

def show_analysis():
    st.header("Detailed Data Analysis")

    df = load_data()
    if df is not None:
        # 1. Bar Plot for Brand Distribution
        st.subheader("Brand Distribution")
        brand_counts = df['brand'].value_counts()
        fig = px.bar(brand_counts, x=brand_counts.index, y=brand_counts.values, labels={'x': 'Brand', 'y': 'Count'})
        st.plotly_chart(fig)

        # 2. Pie Chart for Fuel Type Distribution
        st.subheader("Fuel Type Distribution")
        fuel_counts = df['fuel_type'].value_counts()
        fig = px.pie(fuel_counts, values=fuel_counts.values, names=fuel_counts.index, title="Fuel Type Distribution")
        st.plotly_chart(fig)

        # 3. Histogram of Car Prices
        st.subheader("Distribution of Car Prices")
        fig = px.histogram(df, x='selling_price', nbins=50, title="Price Distribution")
        st.plotly_chart(fig)

        # 4. Box Plot for Price by Transmission Type
        st.subheader("Price by Transmission Type")
        fig = px.box(df, x='transmission_type', y='selling_price', title="Price Distribution by Transmission Type")
        st.plotly_chart(fig)

        # 5. Scatter Plot - Price vs Mileage
        st.subheader("Price vs Mileage")
        fig = px.scatter(df, x='mileage', y='selling_price', trendline="ols", title="Price vs. Mileage")
        st.plotly_chart(fig)

        # 6. Heatmap of Correlation Matrix
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # 7. Line Plot - Average Price by Car Age
        st.subheader("Average Price by Car Age")
        if 'vehicle_age' in df.columns:
            age_price = df.groupby('vehicle_age')['selling_price'].mean().reset_index()
            fig = px.line(age_price, x='vehicle_age', y='selling_price', title="Average Price by Car Age")
            st.plotly_chart(fig)

        # 8. Violin Plot for Price by Seller Type
        st.subheader("Price by Seller Type")
        fig = px.violin(df, x='seller_type', y='selling_price', box=True, title="Price Distribution by Seller Type")
        st.plotly_chart(fig)

def show_model_comparison():
    st.header("Model Comparison")
    st.write("Compare model performance metrics on test dataset.")

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

def show_contact():
    st.header("Contact Us")
    st.markdown("""
        - [LinkedIn](https://www.linkedin.com/in/shamanth-m-05537b264) 🖇️
        - [Instagram](https://www.instagram.com/shamanth_m_) 📸
        - [Email](mailto:shamanth2626@gmail.com) 📧
    """)

# ---- DISPLAY SELECTED PAGE ----
if 'page' not in st.session_state:
    st.session_state.page = 'home'

def switch_page(page_name):
    st.session_state.page = page_name

st.sidebar.title("Navigation")
st.sidebar.button("Home", on_click=switch_page, args=('home',))
st.sidebar.button("Prediction", on_click=switch_page, args=('prediction',))
st.sidebar.button("Data Analysis", on_click=switch_page, args=('analysis',))
st.sidebar.button("Model Comparison", on_click=switch_page, args=('model_comparison',))
st.sidebar.button("Contact", on_click=switch_page, args=('contact',))

# Display the appropriate page
if st.session_state.page == 'home':
    show_home()
elif st.session_state.page == 'prediction':
    show_prediction()
elif st.session_state.page == 'analysis':
    show_analysis()
elif st.session_state.page == 'model_comparison':
    show_model_comparison()
elif st.session_state.page == 'contact':
    show_contact()

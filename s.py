import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OrdinalEncoder
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ---- PAGE CONFIGURATION ----
st.set_page_config(page_title="Car Price Prediction & Analysis Dashboard", page_icon=":car:", layout="wide")

# ---- CUSTOM CSS FOR BACKGROUND ----
page_bg_img = '''
<style>
.stApp {
    background-image: url("https://in.pinterest.com/pin/604749056203375514/");
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
    """Loads and preprocesses the car dataset."""
    try:
        df = pd.read_csv('data/CAR DETAILS FROM deekshith.csv', on_bad_lines='skip')
        df['car_age'] = 2024 - df['year']
        df.drop(columns=['year'], inplace=True)

        # Encoding categorical features
        encoder = OrdinalEncoder()
        categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner']
        df[categorical_cols] = encoder.fit_transform(df[categorical_cols])
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None
# ---- HOME PAGE ----
def show_home():
    st.title("Car Price Prediction")
    st.subheader("Get accurate predictions on car prices and explore data insights.")

    # Load data and train model
    df = load_data()
    if df is not None:
        model, X_train, X_test, y_train, y_test = train_model(df)

        # Prediction input fields
        car_age = st.slider("Car Age", 0, 20, 10)
        km_driven = st.number_input("Kilometers Driven", 0, 300000, 50000)
        seats = st.selectbox("Seats", [2, 4, 5, 7])
        max_power = st.number_input("Max Power (in bhp)", 50, 500, 100)
        mileage = st.number_input("Mileage (kmpl)", 5.0, 35.0, 20.0)
        engine_cc = st.number_input("Engine Capacity (CC)", 500, 5000, 1200)
        
        # Prepare input for prediction
        input_data = pd.DataFrame({
            'car_age': [car_age],
            'Kilometers_Driven': [km_driven],
            'Seats': [seats],
            'Power': [max_power],
            'Mileage': [mileage],
            'Engine': [engine_cc]
        })

        # Add any missing columns for prediction
        missing_cols = set(X_train.columns) - set(input_data.columns)
        for col in missing_cols:
            input_data[col] = 0
        input_data = input_data[X_train.columns]

        # Prediction
        prediction = model.predict(input_data)
        st.write(f"Predicted Selling Price: ₹ {prediction[0]:,.2f}")
# ---- MAIN PAGE NAVIGATION ----
def main():
    """Navigation between different sections of the app."""
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Analysis", "Prediction", "Model Comparison", "Contact"])
    
    if page == "Home":
        show_home()
    elif page == "Analysis":
        show_analysis()
    elif page == "Prediction":
        show_predict()
    elif page == "Model Comparison":
        show_model_comparison()
    elif page == "Contact":
        show_contact()

# ---- HOME PAGE ----
def show_home():
    st.title("Car Price Prediction")
    st.subheader("Get accurate predictions on car prices and explore data insights.")

    # Load data and train model
    df = load_data()
    if df is not None:
        model, X_train, X_test, y_train, y_test = train_model(df)

        # Prediction input fields
        car_age = st.slider("Car Age", 0, 20, 10)
        km_driven = st.number_input("Kilometers Driven", 0, 300000, 50000)
        seats = st.selectbox("Seats", [2, 4, 5, 7])
        max_power = st.number_input("Max Power (in bhp)", 50, 500, 100)
        mileage = st.number_input("Mileage (kmpl)", 5.0, 35.0, 20.0)
        engine_cc = st.number_input("Engine Capacity (CC)", 500, 5000, 1200)
        
        # Prepare input for prediction
        input_data = pd.DataFrame({
            'car_age': [car_age],
            'Kilometers_Driven': [km_driven],
            'Seats': [seats],
            'Power': [max_power],
            'Mileage': [mileage],
            'Engine': [engine_cc]
        })

        # Add any missing columns for prediction
        missing_cols = set(X_train.columns) - set(input_data.columns)
        for col in missing_cols:
            input_data[col] = 0
        input_data = input_data[X_train.columns]

        # Prediction
        prediction = model.predict(input_data)
        st.write(f"Predicted Selling Price: ₹ {prediction[0]:,.2f}")

# ---- ANALYSIS PAGE ----
def show_analysis():
    """Displays various data visualizations for exploratory analysis."""
    st.title("Data Analysis")
    df = load_data()

    if df is not None:
        # Bar charts for categorical variables
        st.subheader("Bar Charts for Categorical Variables")
        plot_bar_chart(df, 'name', 'Brand Distribution')
        plot_bar_chart(df, 'fuel', 'Fuel Type Distribution')
        plot_bar_chart(df, 'seller_type', 'Seller Type Distribution')
        plot_bar_chart(df, 'owner', 'Owner Type Distribution')
        plot_bar_chart(df, 'Seats', 'Seats Distribution')

        # Histograms for numerical variables
        st.subheader("Histograms for Numerical Variables")
        plot_histogram(df, 'selling_price', 'Distribution of Selling Price')
        plot_histogram(df, 'km_driven', 'Distribution of Kilometers Driven')

        # Correlation heatmap
        st.subheader("Feature Correlation Heatmap")
        plot_correlation_heatmap(df)

# ---- PREDICTION PAGE ----
def show_predict():
    """Predicts the selling price based on user input."""
    st.title("Car Price Prediction")
    df = load_data()
    if df is not None:
        model, X_train, _, _, _ = train_random_forest_model(df)
        input_data = get_prediction_input(X_train)
        display_prediction(model, input_data, X_train)

# ---- MODEL COMPARISON PAGE ----
def show_model_comparison():
    """Compares RMSE, MAE, and R² Score across different models."""
    st.title("Model Comparison")
    df = load_data()
    if df is not None:
        # Splitting Data
        X = df.drop(columns=['selling_price', 'name'])
        y = df['selling_price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model metrics calculation
        model_metrics = calculate_model_metrics(X_train, X_test, y_train, y_test)
        st.dataframe(model_metrics)

        # Scatter plot for Random Forest predictions vs. actual
        plot_rf_scatter(X_train, X_test, y_train, y_test)

        # Feature importance plot for Random Forest
        plot_feature_importance(RandomForestRegressor(n_estimators=100, random_state=42), X_train, y_train)

        # Gradient Boosting loss function plot
        plot_gbm_loss(GradientBoostingRegressor(n_estimators=100, random_state=42), X_train, y_train)

# ---- CONTACT PAGE ----
def show_contact():
    """Contact details section."""
    st.title("Contact Us")
    st.markdown("""
        - [LinkedIn](https://www.linkedin.com/in/shamanth-m-05537b264)
        - [Instagram](https://www.instagram.com/shamanth_m_)
        - [Email](mailto:shamanth2626@gmail.com)
    """)

# ---- HELPER FUNCTIONS ----

def train_random_forest_model(df):
    """Trains a Random Forest model on the data."""
    X = df.drop(columns=['selling_price', 'name'])
    y = df['selling_price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

def get_prediction_input(X_train):
    """Gathers user input for prediction."""
    car_age = st.slider("Car Age", 0, 20, 10)
    km_driven = st.number_input("Kilometers Driven", 0, 300000, 50000)
    seats = st.selectbox("Seats", [2, 4, 5, 7])
    max_power = st.number_input("Max Power (in bph)", 50, 500, 100)
    mileage = st.number_input("Mileage (kmpl)", 5.0, 35.0, 20.0)
    engine_cc = st.number_input("Engine Capacity (CC)", 500, 5000, 1200)
    
    input_data = pd.DataFrame({
        'car_age': [car_age],
        'km_driven': [km_driven],
        'Seats': [seats],
        'max_power': [max_power],
        'mileage': [mileage],
        'engine_cc': [engine_cc]
    })
    
    # Align columns with training data
    for col in set(X_train.columns) - set(input_data.columns):
        input_data[col] = 0
    input_data = input_data[X_train.columns]
    
    return input_data

def display_prediction(model, input_data, X_train):
    """Displays the predicted car price based on the model."""
    prediction = model.predict(input_data)
    st.write(f"Predicted Selling Price: ₹ {prediction[0]:,.2f}")

def plot_bar_chart(df, column, title):
    """Plots a bar chart for a specified column."""
    counts = df[column].value_counts()
    fig = px.bar(counts, x=counts.index, y=counts.values, labels={'x': column, 'y': 'Count'}, title=title)
    st.plotly_chart(fig)

def plot_histogram(df, column, title):
    """Plots a histogram for a specified column."""
    fig = px.histogram(df, x=column, title=title, color_discrete_sequence=['#636EFA'])
    st.plotly_chart(fig)

def plot_correlation_heatmap(df):
    """Plots a correlation heatmap."""
    corr_matrix = df.corr()
    fig = px.imshow(corr_matrix, text_auto=True, title="Feature Correlation Heatmap", color_continuous_scale="RdBu")
    st.plotly_chart(fig)

def plot_rf_scatter(X_train, X_test, y_train, y_test):
    """Scatter plot of predicted vs. actual values for Random Forest."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=predictions, mode='markers', name='Predicted vs. Actual'))
    fig.add_trace(go.Scatter(x=y_test, y=y_test, mode='lines', name='Ideal Line'))
    fig.update_layout(title="RF: Predicted vs. Actual Prices", xaxis_title="Actual", yaxis_title="Predicted")
    st.plotly_chart(fig)

def plot_feature_importance(model, X_train, y_train):
    """Feature importance plot for Random Forest."""
    model.fit(X_train, y_train)
    importance_df = pd.DataFrame({"Feature": X_train.columns, "Importance": model.feature_importances_}).sort_values(by="Importance", ascending=False)
    fig = px.bar(importance_df, x="Importance", y="Feature", title="Feature Importances", orientation="h", color="Importance")
    st.plotly_chart(fig)

def plot_gbm_loss(model, X_train, y_train):
    """Plots loss function for Gradient Boosting."""
    model.fit(X_train, y_train)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=model.train_score_, mode='lines', name='Training Loss'))
    fig.update_layout(title="GBM Loss Function vs Number of Trees", xaxis_title="Number of Trees", yaxis_title="Loss")
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()

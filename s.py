import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

# ---- PAGE CONFIGURATION ----
st.set_page_config(page_title="Car Price Prediction & Analysis Dashboard", page_icon=":car:", layout="wide")

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
    """Loads and preprocesses the car dataset."""
    try:
        df = pd.read_csv('data/used_cars.csv', on_bad_lines='skip')
        df['car_age'] = 2024 - df['model_year']
        df.drop(columns=['model_year'], inplace=True)

        # Encoding categorical features
        encoder = OrdinalEncoder()
        categorical_cols = ['fuel_type', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title']
        df[categorical_cols] = encoder.fit_transform(df[categorical_cols])
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# ---- INITIALIZE SESSION STATE FOR PAGE TRACKING ----
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Function to switch page
def switch_page(page_name):
    st.session_state.page = page_name

# Sidebar Navigation buttons
st.sidebar.title("Navigation")
st.sidebar.button("Home", on_click=switch_page, args=('home',))
st.sidebar.button("Prediction", on_click=switch_page, args=('prediction',))
st.sidebar.button("Data Analysis", on_click=switch_page, args=('analysis',))
st.sidebar.button("Model Comparison", on_click=switch_page, args=('model_comparison',))
st.sidebar.button("Contact", on_click=switch_page, args=('contact',))

# ---- PAGE SECTIONS ----
def show_home():
    st.title("Car Price Prediction")
    st.subheader("Get accurate predictions on car prices and explore data insights.")

def show_prediction():
    st.header("Car Price Prediction")

    # Load data and train model
    df = load_data()
    if df is not None:
        model, X_train, X_test, y_train, y_test = train_random_forest_model(df)

        # Prediction input fields
        brand = st.selectbox("Brand", sorted(df['brand'].unique()))
        car_age = st.slider("Car Age", 0, 20, 10)
        km_driven = st.number_input("Kilometers Driven", 0, 300000, 50000)
        seats = st.selectbox("Seats", [2, 4, 5, 7])
        max_power = st.number_input("Max Power (in bhp)", 50, 500, 100)
        mileage = st.number_input("Mileage (kmpl)", 5.0, 35.0, 20.0)
        engine_cc = st.number_input("Engine Capacity (CC)", 500, 5000, 1200)
        
        # Prepare input for prediction
        input_data = pd.DataFrame({
            'brand': [brand],
            'car_age': [car_age],
            'km_driven': [km_driven],
            'Seats': [seats],
            'max_power': [max_power],
            'mileage': [mileage],
            'engine_cc': [engine_cc]
        })

        # Add missing columns for prediction
        missing_cols = set(X_train.columns) - set(input_data.columns)
        for col in missing_cols:
            input_data[col] = 0
        input_data = input_data[X_train.columns]

        # Prediction
        prediction = model.predict(input_data)
        st.write(f"Predicted Selling Price: ₹ {prediction[0]:,.2f}")

def show_analysis():
    st.header("Detailed Data Analysis")
    df = load_data()

    if df is not None:
        # Brand Distribution
        st.subheader("Car Brand Distribution")
        brand_count = df['brand'].value_counts()
        fig = px.bar(brand_count, x=brand_count.index, y=brand_count.values, labels={'x': 'Brand', 'y': 'Count'}, title="Brand Distribution")
        st.plotly_chart(fig)

        # Accident History and Price Impact
        st.subheader("Price Impact by Accident History")
        fig = px.box(df, x="accident", y="price", title="Price Distribution by Accident History")
        st.plotly_chart(fig)

        # Exterior and Interior Color Distribution
        st.subheader("Top 10 Exterior and Interior Colors")
        ext_color_counts = df['ext_col'].value_counts().nlargest(10)
        fig = px.bar(ext_color_counts, x=ext_color_counts.index, y=ext_color_counts.values, title="Top 10 Exterior Colors")
        st.plotly_chart(fig)
        
        int_color_counts = df['int_col'].value_counts().nlargest(10)
        fig = px.bar(int_color_counts, x=int_color_counts.index, y=int_color_counts.values, title="Top 10 Interior Colors")
        st.plotly_chart(fig)

        # Transmission Breakdown
        st.subheader("Transmission Type Breakdown")
        transmission_count = df['transmission'].value_counts()
        fig = px.pie(transmission_count, names=transmission_count.index, values=transmission_count.values, title="Transmission Type Distribution")
        st.plotly_chart(fig)

        # Price vs. Mileage and Model Year
        st.subheader("Price vs Mileage and Model Year")
        fig = px.scatter(df, x="milage", y="price", trendline="ols", title="Mileage vs. Price")
        st.plotly_chart(fig)
        fig = px.scatter(df, x="car_age", y="price", trendline="ols", title="Car Age vs. Price")
        st.plotly_chart(fig)

        # Brand Price Variations
        st.subheader("Price Variations by Brand")
        fig = px.violin(df, x="brand", y="price", box=True, title="Price Distribution by Brand")
        st.plotly_chart(fig)

        # Correlation Heatmap
        st.subheader("Feature Correlation Heatmap")
        correlation_matrix = df.corr()
        fig = px.imshow(correlation_matrix, text_auto=True, color_continuous_scale="RdBu", title="Feature Correlation Heatmap")
        st.plotly_chart(fig)

        # Histograms for Numerical Columns
        st.subheader("Distribution of Numerical Features")
        for column in ['price', 'milage', 'car_age', 'engine_cc']:
            fig = px.histogram(df, x=column, title=f"Distribution of {column.capitalize()}")
            st.plotly_chart(fig)

def show_model_comparison():
    st.header("Model Comparison")
    st.write("Compare model performance metrics on training and test datasets.")
    
    # Load data and split
    df = load_data()
    if df is not None:
        X = df.drop(columns=['price'])
        y = df['price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Models to compare
        models = {
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
        }

        # Dictionary to store metrics
        metrics = {"Model": [], "RMSE": [], "MAE": [], "R² Score": []}

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Store metrics
            metrics["Model"].append(model_name)
            metrics["RMSE"].append(f"{rmse:,.2f}")
            metrics["MAE"].append(f"{mae:,.2f}")
            metrics["R² Score"].append(f"{r2:.2f}")

        # Display metrics in a table
        metrics_df = pd.DataFrame(metrics)
        st.table(metrics_df)

        # Plot Random Forest Scatter and Feature Importance
        rf_model = models["Random Forest"]
        plot_rf_scatter(X_train, X_test, y_train, y_test, rf_model)
        plot_feature_importance(rf_model, X_train, y_train)

def show_contact():
    st.header("Contact Us")
    st.markdown("""
        - [LinkedIn](https://www.linkedin.com/in/shamanth-m-05537b264)
        - [Instagram](https://www.instagram.com/shamanth_m_)
        - [Email](mailto:shamanth2626@gmail.com)
    """)

# ---- HELPER FUNCTIONS ----
def train_random_forest_model(df):
    """Trains a Random Forest model on the data."""
    X = df.drop(columns=['price', 'brand'])
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

def plot_rf_scatter(X_train, X_test, y_train, y_test, model):
    """Scatter plot of predicted vs. actual values for Random Forest."""
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

# ---- MAIN SECTION ----
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

# Car Price Prediction & Analysis Dashboard 🚗



#streamlit link********"https://q8pptv2nhseudi6hdkzzc3.streamlit.app/?page=Predict"*********



This **Car Price Prediction & Analysis Dashboard** is a Streamlit application designed for analyzing and predicting car prices using machine learning models. It provides various functionalities such as data visualization, model training, and prediction, along with an intuitive navigation system.

## Features

- **Home Page**: A brief introduction to the app’s purpose and capabilities.
- **Prediction**: Allows users to input various car features and get a predicted selling price.
- **Data Analysis**: Visualizes data through bar charts, histograms, and a correlation heatmap for better insights into the features affecting car price.
- **Model Comparison**: Shows a performance comparison of different models, including metrics like RMSE, MAE, and R² score.
- **Contact**: Links for contacting the creator via LinkedIn, Instagram, and email.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [App Pages](#app-pages)
4. [Helper Functions](#helper-functions)
5. [Dependencies](#dependencies)
6. [Contact](#contact)

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repository-url.git
   cd your-repository-folder
   ```

2. **Install the Required Packages**:
   Ensure Python is installed and run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the App**:
   ```bash
   streamlit run app.py
   ```

---

## Usage

To run the dashboard, execute `streamlit run app.py` in your terminal, and open the app in the provided local server URL. You can interact with the sidebar to navigate between different sections of the dashboard, explore data insights, and make predictions.

---

## App Pages

### Home Page
The **Home** page provides an overview of the application’s purpose, focusing on car price prediction and data analysis.

### Prediction
The **Prediction** page enables users to predict car prices by inputting various car attributes such as:

- Car age
- Kilometers driven
- Number of seats
- Maximum power
- Mileage
- Engine capacity

It uses a pre-trained Random Forest Regressor model to predict the car price based on the input values.

### Data Analysis
The **Data Analysis** page allows users to visualize the dataset. It includes:

- **Bar Charts**: For categorical variables like car brand, fuel type, seller type, etc.
- **Histograms**: For continuous variables like selling price and kilometers driven.
- **Correlation Heatmap**: For understanding relationships between different numeric features.

### Model Comparison
The **Model Comparison** page presents a comparison table of various models (Linear Regression, Random Forest, and Gradient Boosting) with their performance metrics (RMSE, MAE, and R² score). 

This section also includes:
- A **scatter plot** of predicted vs. actual values for Random Forest.
- A **feature importance plot** for the Random Forest model.
- A **loss function plot** for Gradient Boosting.

### Contact
The **Contact** page provides links to the creator's social profiles and email for feedback or inquiries.

---

## Helper Functions

### `load_data()`
Loads and preprocesses the car dataset, including encoding categorical features and computing car age.

### `train_random_forest_model()`
Trains a Random Forest model on the preprocessed data.

### `plot_bar_chart()`
Plots bar charts for categorical columns to visualize their distribution.

### `plot_histogram()`
Plots histograms for numerical columns to show data distribution.

### `plot_correlation_heatmap()`
Displays a heatmap of feature correlations within the dataset.

### `plot_rf_scatter()`
Plots predicted vs. actual values for the Random Forest model on the test data.

### `plot_feature_importance()`
Plots feature importance values for the Random Forest model.

### `plot_gbm_loss()`
Shows a loss function plot for the Gradient Boosting model to illustrate model training over the number of trees.

---

## Dependencies

This application uses the following libraries:

- **Streamlit**: For building the web application interface
- **pandas**: For data loading and preprocessing
- **scikit-learn**: For machine learning models and data splitting
- **plotly**: For interactive data visualizations

Install dependencies from the `requirements.txt` file.

---

## Contact

- **LinkedIn**: [LinkedIn Profile](https://www.linkedin.com/in/shamanth-m-05537b264)
- **Instagram**: [Instagram Profile](https://www.instagram.com/shamanth_m_)
- **Email**: [Email Me](mailto:shamanth2626@gmail.com)

---

## License

This project is open source and available under the [MIT License](LICENSE).



Thank you for using the Car Price Prediction & Analysis Dashboard! Happy predicting!

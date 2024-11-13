# Car Price Prediction & Analysis Dashboard 🚗

**Streamlit App Link:** [Car Price Prediction & Analysis Dashboard](https://q8pptv2nhseudi6hdkzzc3.streamlit.app/?page=Predict)

This **Car Price Prediction & Analysis Dashboard** is a Streamlit-based application designed to help users analyze and predict car prices using machine learning models. The dashboard offers an intuitive navigation system and interactive features such as data visualization, model training, and car price prediction.

---

## Features

- **Home Page**: Provides an introduction to the app’s purpose and its functionalities.
- **Prediction**: Allows users to input car details and get a predicted selling price.
- **Data Analysis**: Visualizes the data through charts, histograms, and a heatmap to gain insights into the factors that influence car prices.
- **Model Comparison**: Shows the performance of various models (e.g., Random Forest, Gradient Boosting) using metrics like RMSE, MAE, and R².
- **Contact**: Provides links for contacting the creator via LinkedIn, Instagram, and email.

---

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

To launch the dashboard, run `streamlit run app.py` in your terminal, then open the provided local server URL. Use the sidebar to navigate through the different sections of the app, explore data insights, and make predictions.

---

## App Pages

### Home Page
The **Home** page introduces the purpose of the application, focusing on car price prediction and data analysis.

### Prediction
The **Prediction** page allows users to predict car prices by entering car attributes such as:

- Car age
- Kilometers driven
- Number of seats
- Maximum power
- Mileage
- Engine capacity

It leverages a pre-trained Random Forest Regressor model to provide an estimated selling price based on the user’s inputs.

### Data Analysis
The **Data Analysis** page lets users visualize the dataset through various charts:

- **Bar Charts** for categorical features like car brand, fuel type, seller type, etc.
- **Histograms** for continuous features like selling price and kilometers driven.
- **Correlation Heatmap** to explore relationships between numeric features.

### Model Comparison
The **Model Comparison** page provides a performance comparison of different models (Linear Regression, Random Forest, Gradient Boosting) using key metrics:

- **RMSE**, **MAE**, and **R² score**
- **Scatter plot** of predicted vs. actual values for the Random Forest model
- **Feature importance plot** for Random Forest
- **Loss function plot** for Gradient Boosting

### Contact
The **Contact** page includes links to the creator's social profiles and an email option for feedback or inquiries.

---

## Helper Functions

### `load_data()`
Loads and preprocesses the car dataset, including encoding categorical features and calculating car age.

### `train_random_forest_model()`
Trains a Random Forest model on the preprocessed data.

### `plot_bar_chart()`
Creates bar charts for categorical columns to visualize distributions.

### `plot_histogram()`
Creates histograms for numeric columns to show data distributions.

### `plot_correlation_heatmap()`
Displays a heatmap of feature correlations within the dataset.

### `plot_rf_scatter()`
Plots predicted vs. actual values for the Random Forest model on test data.

### `plot_feature_importance()`
Displays feature importance for the Random Forest model.

### `plot_gbm_loss()`
Shows the loss function for Gradient Boosting to illustrate model performance over iterations.

---

## Dependencies

This application requires the following libraries:

- **Streamlit**: For building the web application interface
- **pandas**: For data loading and preprocessing
- **scikit-learn**: For machine learning models and data processing
- **plotly**: For creating interactive visualizations

To install all dependencies, use the `requirements.txt` file provided.

---

## Contact

For questions, feedback, or collaboration opportunities, feel free to reach out:

- **LinkedIn**: [Shamanth M's LinkedIn Profile](https://www.linkedin.com/in/shamanth-m-05537b264)
- **Instagram**: [Shamanth M's Instagram](https://www.instagram.com/shamanth_m_)
- **Email**: [Email Me](mailto:shamanth2626@gmail.com)

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

Thank you for using the **Car Price Prediction & Analysis Dashboard**! Happy predicting! 🚗

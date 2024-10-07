pip install plotly
import pandas as pd
import streamlit as st
import plotly.express as px

# ---- PAGE CONFIGURATION ----
st.set_page_config(page_title="Car Price Analysis Dashboard", page_icon=":car:", layout="wide")

# ---- LOAD DATA ----
def load_data():
    uploaded_file = st.sidebar.file_uploader("Upload A Dataset (.xlsx)", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        return df
    else:
        st.warning("Please upload an Excel file to proceed.")
        st.stop()

# Load the data
df = load_data()

# ---- SIDEBAR FILTERS ----
st.sidebar.header("Please Filter Here:")

car_brands = st.sidebar.multiselect("Select Car Brand:", options=df["name"].unique(), default=df["name"].unique())
year = st.sidebar.slider("Select Year Range:", min_value=int(df["year"].min()), max_value=int(df["year"].max()), value=(int(df["year"].min()), int(df["year"].max())))
fuel = st.sidebar.multiselect("Select Fuel Type:", options=df["fuel"].unique(), default=df["fuel"].unique())
seller_type = st.sidebar.multiselect("Select Seller Type:", options=df["seller_type"].unique(), default=df["seller_type"].unique())

# Filter DataFrame
df_selection = df.query("name == @car_brands & year >= @year[0] & year <= @year[1] & fuel == @fuel & seller_type == @seller_type")

# Display warning if no data is available
if df_selection.empty:
    st.warning("No data available based on the current filter settings!")
    st.stop()

# ---- MAIN DASHBOARD ----
st.title(":car: Car Price Analysis Dashboard")

# ---- ADDITIONAL VISUALIZATIONS ----

# 1. Pie Chart for Seller Type Distribution
st.markdown("### Seller Type Distribution")
seller_type_distribution = df_selection["seller_type"].value_counts()
fig_pie_seller_type = px.pie(values=seller_type_distribution.values, names=seller_type_distribution.index,
                             title="<b>Distribution by Seller Type</b>",
                             color_discrete_sequence=px.colors.sequential.RdBu)
st.plotly_chart(fig_pie_seller_type, use_container_width=True)

# 2. Pie Chart for Fuel Type Distribution
st.markdown("### Fuel Type Distribution")
fuel_distribution = df_selection["fuel"].value_counts()
fig_pie_fuel_type = px.pie(values=fuel_distribution.values, names=fuel_distribution.index,
                           title="<b>Distribution by Fuel Type</b>",
                           color_discrete_sequence=px.colors.sequential.Blues)
st.plotly_chart(fig_pie_fuel_type, use_container_width=True)

# 3. Box Plot for Selling Price by Fuel Type
st.markdown("### Box Plot of Selling Price by Fuel Type")
fig_box_fuel_price = px.box(df_selection, x="fuel", y="selling_price", color="fuel",
                            title="<b>Selling Price Distribution by Fuel Type</b>",
                            labels={"fuel": "Fuel Type", "selling_price": "Selling Price (₹)"},
                            template="plotly_white")
st.plotly_chart(fig_box_fuel_price, use_container_width=True)

# 4. Box Plot for Selling Price by Car Brand
st.markdown("### Box Plot of Selling Price by Car Brand")
fig_box_brand_price = px.box(df_selection, x="name", y="selling_price", color="name",
                             title="<b>Selling Price Distribution by Car Brand</b>",
                             labels={"name": "Car Brand", "selling_price": "Selling Price (₹)"},
                             template="plotly_white")
st.plotly_chart(fig_box_brand_price, use_container_width=True)

# 5. Scatter Plot for Engine Size vs. Selling Price
st.markdown("### Scatter Plot: Engine Size vs. Selling Price")
fig_engine_vs_price = px.scatter(df_selection, x="Engine (CC)", y="selling_price", color="fuel",
                                 title="<b>Engine Size vs. Selling Price</b>",
                                 labels={"Engine (CC)": "Engine Size (CC)", "selling_price": "Selling Price (₹)"},
                                 template="plotly_white")
st.plotly_chart(fig_engine_vs_price, use_container_width=True)

# 6. Bar Chart for Average Selling Price by Owner Type
st.markdown("### Average Selling Price by Owner Type")
price_by_owner = df_selection.groupby("owner")[["selling_price"]].mean().reset_index()
fig_avg_price_owner = px.bar(price_by_owner, x="owner", y="selling_price", color="owner",
                             title="<b>Average Selling Price by Owner Type</b>",
                             labels={"owner": "Owner Type", "selling_price": "Selling Price (₹)"},
                             template="plotly_white")
st.plotly_chart(fig_avg_price_owner, use_container_width=True)

# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

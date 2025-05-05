import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Supermarket Sales Optimization Dashboard", layout="wide")

# Load and preprocess data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("supermarket_sales.csv")
        # Parse date and time together
        df['Date'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'], format='%m/%d/%Y %H:%M')
        
        # Feature engineering
        df['Day_of_Week'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['Is_Weekend'] = df['Day_of_Week'].isin([5, 6]).astype(int)
        df['Is_Member'] = (df['Customer type'] == 'Member').astype(int)
        df['Payment_Method'] = df['Payment'].map({'Cash': 0, 'Credit card': 1, 'Ewallet': 2})
        
        # Handle missing values
        df.fillna({
            'Unit price': df['Unit price'].mean(),
            'Quantity': df['Quantity'].median(),
            'Rating': df['Rating'].mean(),
            'gross income': df['gross income'].mean()
        }, inplace=True)
        
        # Outlier removal (IQR method for Total)
        Q1 = df['Total'].quantile(0.25)
        Q3 = df['Total'].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df['Total'] >= Q1 - 1.5 * IQR) & (df['Total'] <= Q3 + 1.5 * IQR)]
        
        return df
    except FileNotFoundError:
        st.error("Dataset 'supermarket_sales.csv' not found. Please upload the file.")
        return None

df = load_data()
if df is None:
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
selected_city = st.sidebar.multiselect("Select City", options=df['City'].unique() if df is not None else [], default=df['City'].unique() if df is not None else [])
selected_gender = st.sidebar.multiselect("Select Gender", options=df['Gender'].unique() if df is not None else [], default=df['Gender'].unique() if df is not None else [])
selected_customer_type = st.sidebar.multiselect("Select Customer Type", options=df['Customer type'].unique() if df is not None else [], default=df['Customer type'].unique() if df is not None else [])
selected_payment = st.sidebar.multiselect("Select Payment Method", options=df['Payment'].unique() if df is not None else [], default=df['Payment'].unique() if df is not None else [])
date_range = st.sidebar.date_input("Select Date Range", value=[df['Date'].min().date(), df['Date'].max().date()] if df is not None else [pd.to_datetime('2019-01-01').date(), pd.to_datetime('2019-12-31').date()])

if df is not None:
    filtered_df = df[
        (df['City'].isin(selected_city)) &
        (df['Gender'].isin(selected_gender)) &
        (df['Customer type'].isin(selected_customer_type)) &
        (df['Payment'].isin(selected_payment)) &
        (df['Date'].dt.date.between(pd.to_datetime(date_range[0]).date(), pd.to_datetime(date_range[1]).date()))
    ]
else:
    filtered_df = pd.DataFrame()

# Dashboard title
st.title("📊 Simba Supermarket Sales Dashboard")

# Show data
st.subheader("Sample of Filtered Data")
st.dataframe(filtered_df.head())

# Sales Trend Visualization (Plotly)
st.subheader("📈 Daily Sales Trend")
if not filtered_df.empty:
    daily_sales = filtered_df.groupby(filtered_df['Date'].dt.date)["Total"].sum().reset_index()
    daily_sales['Date'] = pd.to_datetime(daily_sales['Date'])
    fig1 = px.line(daily_sales, x='Date', y='Total', title="Total Sales Over Time", labels={'Total': 'Sales'})
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.write("No data available for the selected filters.")

# Seasonal Patterns (Matplotlib/Seaborn)
st.subheader("📅 Monthly Sales Seasonality")
if not filtered_df.empty:
    monthly_sales = filtered_df.groupby("Month")["Total"].mean().reset_index()
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.barplot(data=monthly_sales, x='Month', y='Total', ax=ax2)
    ax2.set_title("Average Sales by Month")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Average Sales")
    st.pyplot(fig2)
else:
    st.write("No data available for the selected filters.")

# Product Line Sales Comparison (Plotly)
st.subheader("🛍️ Sales by Product Line")
if not filtered_df.empty:
    product_sales = filtered_df.groupby("Product line")["Total"].sum().sort_values().reset_index()
    fig3 = px.bar(product_sales, x='Total', y='Product line', orientation='h', title="Total Sales by Product Line", labels={'Total': 'Sales'})
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.write("No data available for the selected filters.")

# Gross Income by Customer Type (Matplotlib/Seaborn)
st.subheader("💰 Gross Income by Customer Type")
if not filtered_df.empty:
    customer_income = filtered_df.groupby("Customer type")["gross income"].sum().reset_index()
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    sns.barplot(data=customer_income, x='Customer type', y='gross income', ax=ax4)
    ax4.set_title("Gross Income by Customer Type")
    ax4.set_xlabel("Customer Type")
    ax4.set_ylabel("Gross Income")
    st.pyplot(fig4)
else:
    st.write("No data available for the selected filters.")

# Sales by Payment Method (Matplotlib/Seaborn)
st.subheader("💳 Sales by Payment Method")
if not filtered_df.empty:
    payment_sales = filtered_df.groupby("Payment")["Total"].sum().reset_index()
    fig5, ax5 = plt.subplots(figsize=(10, 4))
    sns.barplot(data=payment_sales, x='Payment', y='Total', ax=ax5)
    ax5.set_title("Total Sales by Payment Method")
    ax5.set_xlabel("Payment Method")
    ax5.set_ylabel("Sales")
    st.pyplot(fig5)
else:
    st.write("No data available for the selected filters.")

# Model 1: Linear Regression
st.subheader("📈 Sales Prediction Model (Linear Regression)")
if not filtered_df.empty:
    X = filtered_df[['Unit price', 'Quantity', 'Day_of_Week', 'Month', 'Is_Weekend', 'Is_Member', 'Payment_Method', 'Rating']]
    y = filtered_df['Total']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    st.markdown(f"**Linear Regression Metrics**")
    st.markdown(f"- R² Score: {r2_score(y_test, y_pred_lr):.2f}")
    st.markdown(f"- MSE: {mean_squared_error(y_test, y_pred_lr):.2f}")
    st.markdown(f"- MAE: {mean_absolute_error(y_test, y_pred_lr):.2f}")
else:
    st.write("No data available for model training.")

# Model 2: Random Forest
if not filtered_df.empty:
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    st.markdown(f"**Random Forest Metrics**")
    st.markdown(f"- R² Score: {r2_score(y_test, y_pred_rf):.2f}")
    st.markdown(f"- MSE: {mean_squared_error(y_test, y_pred_rf):.2f}")
    st.markdown(f"- MAE: {mean_absolute_error(y_test, y_pred_rf):.2f}")
else:
    st.write("No data available for model training.")

# Inventory Recommendations
st.subheader("📦 Inventory Recommendations")
if not filtered_df.empty:
    st.markdown("Based on Random Forest model and gross income, prioritize inventory for high-performing product lines:")
    product_income = filtered_df.groupby("Product line")["gross income"].sum().sort_values(ascending=True).reset_index()
    fig7 = px.bar(product_income, x='gross income', y='Product line', orientation='h', title="Gross Income by Product Line")
    st.plotly_chart(fig7, use_container_width=True)
    feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.write("Feature Importance for Sales Prediction:")
    st.write(feature_importance)
else:
    st.write("No data available for inventory recommendations.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Your Group | AUCA | Big Data Project")

if __name__ == "__main__":
    st.write("Thank you for using our system.")
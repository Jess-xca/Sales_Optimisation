import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import io
import uuid
import plotly.express as px
import calendar
from datetime import timedelta

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Supermarket Sales Optimization Dashboard", layout="wide")

# Expected columns in the dataset
EXPECTED_COLUMNS = [
    'Date', 'Time', 'City', 'Gender', 'Customer type', 'Payment', 'Unit price', 
    'Quantity', 'Total', 'Rating', 'gross income', 'Product line'
]

# Load and preprocess data
@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(file)
        
        # Validate columns
        missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns in dataset: {', '.join(missing_cols)}")
            return None
            
        # Validate data types
        if not np.issubdtype(df['Unit price'].dtype, np.number):
            st.error("'Unit price' column must be numeric.")
            return None
        if not np.issubdtype(df['Quantity'].dtype, np.number):
            st.error("'Quantity' column must be numeric.")
            return None
            
        # Parse date and time together
        try:
            df['Date'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'], format='%m/%d/%Y %H:%M')
        except Exception as e:
            st.error(f"Error parsing dates: {str(e)}")
            return None
        
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
            'gross income': df['gross income'].mean(),
            'City': df['City'].mode()[0],
            'Gender': df['Gender'].mode()[0],
            'Customer type': df['Customer type'].mode()[0],
            'Payment': df['Payment'].mode()[0]
        }, inplace=True)
        
        # Outlier removal (IQR method for Total)
        Q1 = df['Total'].quantile(0.25)
        Q3 = df['Total'].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df['Total'] >= Q1 - 1.5 * IQR) & (df['Total'] <= Q3 + 1.5 * IQR)]
        
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# File uploader
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload supermarket_sales.csv", type=["csv"])
if uploaded_file is None:
    # Dashboard title
    st.title("📊 Simba Supermarket Sales Dashboard")
    st.error("Please upload the dataset to continue.")
    st.stop()

df = load_data(uploaded_file)
if df is None:
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
selected_city = st.sidebar.multiselect("Select City", options=df['City'].unique(), default=df['City'].unique())
selected_gender = st.sidebar.multiselect("Select Gender", options=df['Gender'].unique(), default=df['Gender'].unique())
selected_customer_type = st.sidebar.multiselect("Select Customer Type", options=df['Customer type'].unique(), default=df['Customer type'].unique())
selected_payment = st.sidebar.multiselect("Select Payment Method", options=df['Payment'].unique(), default=df['Payment'].unique())
date_range = st.sidebar.date_input("Select Date Range", value=[df['Date'].min().date(), df['Date'].max().date()])
unit_price_range = st.sidebar.slider("Unit Price Range", float(df['Unit price'].min()), float(df['Unit price'].max()), (float(df['Unit price'].min()), float(df['Unit price'].max())))
quantity_range = st.sidebar.slider("Quantity Range", int(df['Quantity'].min()), int(df['Quantity'].max()), (int(df['Quantity'].min()), int(df['Quantity'].max())))

# Apply filters
filtered_df = df[
    (df['City'].isin(selected_city)) &
    (df['Gender'].isin(selected_gender)) &
    (df['Customer type'].isin(selected_customer_type)) &
    (df['Payment'].isin(selected_payment)) &
    (df['Date'].dt.date.between(pd.to_datetime(date_range[0]).date(), pd.to_datetime(date_range[1]).date())) &
    (df['Unit price'].between(unit_price_range[0], unit_price_range[1])) &
    (df['Quantity'].between(quantity_range[0], quantity_range[1]))
]

# Check if filtered data is empty
if filtered_df.empty:
    st.warning("No data matches the selected filters. Please adjust the filters.")
    st.stop()

# Dashboard title
st.title("📊 Simba Supermarket Sales Dashboard")

# Show data
st.subheader("Sample of Filtered Data")
st.dataframe(filtered_df.head())

# Download filtered data
csv = filtered_df.to_csv(index=False)
st.download_button(
    label="Download Filtered Data",
    data=csv,
    file_name="filtered_supermarket_sales.csv",
    mime="text/csv"
)

# Correlation Heatmap
st.subheader("🔗 Correlation Heatmap")
numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
corr_matrix = filtered_df[numeric_cols].corr()
fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax_corr)
ax_corr.set_title("Correlation Heatmap of Numeric Features")
st.pyplot(fig_corr)

# Box Plot for Outlier Analysis
st.subheader("📉 Outlier Analysis (Total Sales)")
fig_box, ax_box = plt.subplots(figsize=(10, 4))
sns.boxplot(x=filtered_df['Total'], ax=ax_box)
ax_box.set_title("Box Plot of Total Sales")
ax_box.set_xlabel("Total Sales")
st.pyplot(fig_box)

# Sales Trend Visualization (Plotly)
st.subheader("📈 Daily Sales Trend")
daily_sales = filtered_df.groupby(filtered_df['Date'].dt.date)["Total"].sum().reset_index()
daily_sales['Date'] = pd.to_datetime(daily_sales['Date'])
fig1 = px.line(daily_sales, x='Date', y='Total', title="Total Sales Over Time", labels={'Total': 'Sales'})
st.plotly_chart(fig1, use_container_width=True)

# Seasonal Patterns (Plotly with Month Names)
st.subheader("📅 Monthly Sales Seasonality")
monthly_sales = filtered_df.groupby("Month")["Total"].mean().reset_index()
monthly_sales['Month_Name'] = monthly_sales['Month'].apply(lambda x: calendar.month_name[int(x)])
fig2 = px.bar(monthly_sales, x='Month_Name', y='Total', title="Average Sales by Month",
              labels={'Month_Name': 'Month', 'Total': 'Average Sales'})
st.plotly_chart(fig2, use_container_width=True)

# Product Line Sales Comparison (Plotly)
st.subheader("🛍️ Sales by Product Line")
product_sales = filtered_df.groupby("Product line")["Total"].sum().sort_values().reset_index()
fig3 = px.bar(product_sales, x='Total', y='Product line', orientation='h', title="Total Sales by Product Line", labels={'Total': 'Sales'})
st.plotly_chart(fig3, use_container_width=True)

# Gross Income by Customer Type (Matplotlib/Seaborn)
st.subheader("💰 Gross Income by Customer Type")
customer_income = filtered_df.groupby("Customer type")["gross income"].sum().reset_index()
fig4, ax4 = plt.subplots(figsize=(10, 4))
sns.barplot(data=customer_income, x='Customer type', y='gross income', ax=ax4)
ax4.set_title("Gross Income by Customer Type")
ax4.set_xlabel("Customer Type")
ax4.set_ylabel("Gross Income")
st.pyplot(fig4)

# Sales by Payment Method (Matplotlib/Seaborn)
st.subheader("💳 Sales by Payment Method")
payment_sales = filtered_df.groupby("Payment")["Total"].sum().reset_index()
fig5, ax5 = plt.subplots(figsize=(10, 4))
sns.barplot(data=payment_sales, x='Payment', y='Total', ax=ax5)
ax5.set_title("Total Sales by Payment Method")
ax5.set_xlabel("Payment Method")
ax5.set_ylabel("Sales")
st.pyplot(fig5)

# Model Training and Comparison
st.subheader("📈 Sales Prediction Models")
X = filtered_df[['Unit price', 'Quantity', 'Day_of_Week', 'Month', 'Is_Weekend', 'Is_Member', 'Payment_Method', 'Rating']]
y = filtered_df['Total']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
st.markdown(f"**Linear Regression Metrics**")
st.markdown(f"- R² Score: {r2_score(y_test, y_pred_lr):.2f}")
st.markdown(f"- MSE: {mean_squared_error(y_test, y_pred_lr):.2f}")
st.markdown(f"- MAE: {mean_absolute_error(y_test, y_pred_lr):.2f}")

# Random Forest with GridSearchCV
rf_param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
    'min_samples_split': [2]
}
rf_model = RandomForestRegressor(random_state=42)
rf_grid = GridSearchCV(rf_model, rf_param_grid, cv=3, scoring='neg_mean_squared_error')
rf_grid.fit(X_train, y_train)
y_pred_rf = rf_grid.predict(X_test)
st.markdown(f"**Random Forest Metrics (Best Parameters: {rf_grid.best_params_})**")
st.markdown(f"- R² Score: {r2_score(y_test, y_pred_rf):.2f}")
st.markdown(f"- MSE: {mean_squared_error(y_test, y_pred_rf):.2f}")
st.markdown(f"- MAE: {mean_absolute_error(y_test, y_pred_rf):.2f}")

# Model Comparison Plot
fig_comparison, ax_comparison = plt.subplots(figsize=(10, 6))
ax_comparison.scatter(y_test, y_pred_lr, label='Linear Regression', alpha=0.5)
ax_comparison.scatter(y_test, y_pred_rf, label='Random Forest', alpha=0.5)
ax_comparison.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax_comparison.set_xlabel("Actual Total Sales")
ax_comparison.set_ylabel("Predicted Total Sales")
ax_comparison.set_title("Actual vs Predicted Sales")
ax_comparison.legend()
st.pyplot(fig_comparison)

# Future Sales Prediction
st.subheader("🔮 Future Sales Prediction (Next 3 Months)")
# Create future dates for 90 days (3 months)
latest_date = filtered_df['Date'].max().date()
future_dates = [latest_date + timedelta(days=x) for x in range(1, 91)]
future_df = pd.DataFrame({'Date': future_dates})
future_df['Date'] = pd.to_datetime(future_df['Date'])
future_df['Day_of_Week'] = future_df['Date'].dt.dayofweek
future_df['Month'] = future_df['Date'].dt.month
future_df['Is_Weekend'] = future_df['Day_of_Week'].isin([5, 6]).astype(int)

# Simulate other features based on historical averages
avg_features = filtered_df.groupby('Product line').agg({
    'Unit price': 'mean',
    'Quantity': 'mean',
    'Is_Member': 'mean',
    'Payment_Method': 'mean',
    'Rating': 'mean'
}).reset_index()

future_predictions = []
for _, row in avg_features.iterrows():
    temp_df = future_df.copy()
    temp_df['Product line'] = row['Product line']
    temp_df['Unit price'] = row['Unit price']
    temp_df['Quantity'] = row['Quantity']
    temp_df['Is_Member'] = row['Is_Member']
    temp_df['Payment_Method'] = row['Payment_Method']
    temp_df['Rating'] = row['Rating']
    X_future = temp_df[['Unit price', 'Quantity', 'Day_of_Week', 'Month', 'Is_Weekend', 'Is_Member', 'Payment_Method', 'Rating']]
    temp_df['Predicted_Sales'] = rf_grid.predict(X_future)
    future_predictions.append(temp_df)

future_predictions_df = pd.concat(future_predictions, ignore_index=True)

# Visualize future sales by product line
fig_future = px.line(future_predictions_df, x='Date', y='Predicted_Sales', color='Product line',
                     title="Predicted Sales for Next 3 Months by Product Line",
                     labels={'Predicted_Sales': 'Predicted Sales'})
st.plotly_chart(fig_future, use_container_width=True)

# Inventory Recommendations
st.subheader("📦 Inventory Recommendations")
st.markdown("Based on predicted sales, historical trends, and gross income, here are inventory recommendations to avoid overstock and stockouts over the next 3 months:")
lead_time_days = 7  # Assumed lead time
safety_stock_factor = 0.1  # 10% safety stock

# Calculate predicted demand and recommended stock levels
inventory_recommendations = future_predictions_df.groupby('Product line').agg({
    'Predicted_Sales': 'sum',
    'Quantity': 'mean'
}).reset_index()
inventory_recommendations['Predicted_Units'] = (inventory_recommendations['Predicted_Sales'] / inventory_recommendations['Quantity']).round()
inventory_recommendations['Safety_Stock'] = (inventory_recommendations['Predicted_Units'] * safety_stock_factor).round()
inventory_recommendations['Recommended_Stock'] = inventory_recommendations['Predicted_Units'] + inventory_recommendations['Safety_Stock']
inventory_recommendations['Historical_Sales'] = filtered_df.groupby('Product line')['Total'].sum().values
inventory_recommendations = inventory_recommendations.sort_values('Predicted_Sales', ascending=False)

# Create inventory tips table
tips_data = []
for _, row in inventory_recommendations.iterrows():
    product = row['Product line']
    predicted_units = int(row['Predicted_Units'])
    safety_stock = int(row['Safety_Stock'])
    recommended_stock = int(row['Recommended_Stock'])
    reorder_point = int(predicted_units / 90 * lead_time_days + safety_stock)
    note = "High-performing product. Prioritize restocking and monitor for promotions." if row['Historical_Sales'] > inventory_recommendations['Historical_Sales'].quantile(0.75) else ""
    tips_data.append({
        'Product Line': product,
        'Predicted Demand (Units)': predicted_units,
        'Recommended Stock (Units)': recommended_stock,
        'Safety Stock (Units)': safety_stock,
        'Reorder Point (Units)': reorder_point,
        'Note': note
    })

tips_df = pd.DataFrame(tips_data)
st.markdown("### Inventory Tips")
st.dataframe(tips_df)

# Visualize recommended stock levels
fig_inventory = px.bar(inventory_recommendations, x='Recommended_Stock', y='Product line', orientation='h',
                       title="Recommended Stock Levels by Product Line (Next 3 Months)",
                       labels={'Recommended_Stock': 'Recommended Stock (Units)'})
st.plotly_chart(fig_inventory, use_container_width=True)

# Analysis Summary
st.subheader("📝 Analysis Summary")
try:
    with open("summary.md", "r") as file:
        st.markdown(file.read())
except FileNotFoundError:
    st.error("Summary file (summary.md) not found. Please ensure it exists in the same directory as this script.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Your Group | AUCA | Big Data Project")

if __name__ == "__main__":
    st.write("Thank you for using our system.")
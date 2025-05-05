# Supermarket Sales Optimization Dashboard

Welcome to the Supermarket Sales Optimization Dashboard! This project, developed for Simba Supermarket, analyzes supermarket sales data to optimize inventory management and supply chains. It uses a dataset of over 800 transactions from multiple branches to identify sales trends, seasonal variations, and customer behavior, predicting future sales to reduce overstock and stockouts.
This README is for guiding you through setting up and running the dashboard, explaining how it works, and showing how it meets the project requirements outlined by the instructor.

# Project Overview

The project investigates retail sales data to enhance inventory and supply chain efficiency. It includes:

- A Python-based Streamlit dashboard (supermarket_sales_dashboard.py) for interactive data exploration.
- A dataset (supermarket_sales.csv) with sales transactions.
- A requirements.txt file listing Python libraries.

# The dashboard lets you:

- Filter data by city, gender, customer type, payment method, and date range.
- Visualize sales trends, seasonality, and product performance.
- View predictions from machine learning models (Linear Regression, Random Forest).
- Get inventory recommendations based on sales and profit.

# Prerequisites

You’ll need:

- Python 3.8 or higher (download from python.org).
- A code editor like Visual Studio Code (optional; download from code.visualstudio.com).
- A terminal (built into VS Code or your OS).
- The project files (supermarket_sales_dashboard.py, supermarket_sales.csv, requirements.txt) in one folder.

# Setup Instructions

Get the Project Files:

- Place supermarket_sales_dashboard.py, supermarket_sales.csv, requirements.txt, and project_report.tex in a folder (e.g., supermarket_sales_project).
- Ensure supermarket_sales.csv is in the same folder as the script.

# Install Python Libraries:

- Open a terminal in your project folder:
  In VS Code: View > Terminal.
  Or use Command Prompt (Windows), Terminal (macOS/Linux) and run cd /path/to/supermarket_sales_project.
- Install dependencies:pip install -r requirements.txt
- This installs streamlit, pandas, matplotlib, seaborn, plotly, openpyxl, scikit-learn, and numpy.

# Check the Dataset:

Verify supermarket_sales.csv is in the folder. It should have columns: Invoice ID, Branch, City, Customer type, Gender, Product line, Unit price, Quantity, Tax 18%, Total, Date, Time, Payment, cogs, gross margin percentage, gross income, Rating.

# Running the Dashboard

In the terminal, navigate to your project folder (if not already there).
Run the Streamlit app:streamlit run supermarket_sales_dashboard.py

Your browser should open to http://localhost:8501. If not, visit that URL manually.

The dashboard will display charts, tables, and model results. If you see errors (e.g., "Dataset not found"), ensure supermarket_sales.csv is in the correct folder.

# How the Dashboard Works

The dashboard is interactive and easy to use. Here’s what you can do:

- View Data:

The "Sample of Filtered Data" table shows transactions, with Date including times (e.g., 2019-03-08 13:08:00).

- Filter Data:

Use the sidebar to filter by:
City (e.g., Yangon,...).
Gender (Male, Female).
Customer Type (Member, Normal).
Payment Method (Cash, Credit card, Ewallet).
Date Range (select start/end dates).

Filters update charts and tables instantly.

- Explore Charts:

Daily Sales Trend: Plotly line chart of total sales over time.
Monthly Sales Seasonality: Seaborn bar chart of average sales by month.
Sales by Product Line: Plotly horizontal bar chart ranking product lines.
Gross Income by Customer Type: Seaborn bar chart comparing Member vs. Normal income.
Sales by Payment Method: Seaborn bar chart of sales by payment type.

- Predictive Models:

Linear Regression: Predicts sales with metrics (R², MSE, MAE).
Random Forest: More accurate predictions with similar metrics.

- Inventory Recommendations:

Plotly chart of gross income by product line.
Table of feature importance (e.g., unit price, quantity) to guide inventory stocking.

# Fulfilling Project Requirements

The project meets all requirements specified by the instructor:

- Dataset Collection:

Uses a dataset of 800+ transactions, with attributes like Invoice ID, City, Product line, Total, Date, Time, and gross income.
Supports inventory and supply chain optimization by analyzing sales patterns and customer behavior.

- Importing Necessary Libraries:

Imports streamlit, pandas, numpy, matplotlib, seaborn, plotly, and scikit-learn for data processing, visualization, and modeling.
Dependencies are listed in requirements.txt for easy installation.

- Dataset Pre-processing:

Combines Date and Time into a single Date column (e.g., 2019-03-08 13:08:00).
Creates features: Day_of_Week, Month, Is_Weekend, Is_Member, Payment_Method.
Fills missing values (means for Unit price, Rating, gross income; median for Quantity).
Removes outliers in Total using the IQR method.

- Creating the Models:

Linear Regression: Predicts sales using features like unit price and customer type.
Random Forest: Captures complex patterns for better accuracy.
Models use 80% training and 20% testing data.

- Evaluating the Model:

Evaluates models with R², MSE, and MAE, displayed in the dashboard.
Random Forest outperforms Linear Regression (R² ≈ 0.87 vs. 0.73).

- Result Visualization:

Visualizes trends (daily sales), seasonality (monthly sales), and comparisons (product lines, customer types, payment methods).
Uses Plotly for interactive charts and Seaborn/Matplotlib for clear visuals.
Note: The dataset lacks promotion data, so promotion impact analysis is a limitation (noted in the report).

- Developing an Interface/Dashboard for Interactivity:

Streamlit dashboard with filters, data table, visualizations, model metrics, and inventory recommendations.
Accessible via streamlit run supermarket_sales_dashboard.py.

# Troubleshooting

"Dataset not found":

Ensure supermarket_sales.csv is in the same folder as supermarket_sales_dashboard.py.
Check the file name is exact (case-sensitive).

Library Installation Fails:

Run pip install streamlit pandas matplotlib seaborn plotly openpyxl scikit-learn numpy.
Use Python 3.8–3.10.

Dashboard Doesn’t Load:

Use streamlit run supermarket_sales_dashboard.py (not python ...).
Visit http://localhost:8501 or check the terminal for a different port (e.g., 8502).

# Additional Notes

Scalability: For larger datasets, use PySpark.
Excel Support: openpyxl is included but unused. To load Excel files, modify load_data() to use pd.read_excel().
Customization: Edit supermarket_sales_dashboard.py in VS Code to add features or change visuals.

- Install dependencies: pip install -r requirements.txt.
- Run the dashboard: streamlit run supermarket_sales_dashboard.py.

# Supermarket Sales Dashboard

The Supermarket Sales Dashboard is an interactive web application built with Streamlit to analyze historical supermarket sales data, predict future sales, and optimize inventory levels. The dashboard helps identify sales trends, seasonal patterns, and potential promotion impacts, enabling data-driven decisions to reduce overstock and stockouts.

---

## Table of Contents
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Code Structure and Importance](#code-structure-and-importance)
- [Key Concepts and Tools](#key-concepts-and-tools)
- [Dataset Requirements](#dataset-requirements)
- [Future Improvements](#future-improvements)

---

## Installation

To run the dashboard, you need Python 3.8+ and the required packages. Follow these steps:

1. **Clone or Download the Project**:
   - Download the project files (`supermarket_sales_dashboard.py`, `summary.md`) or clone the repository if hosted.
   ```bash
   git clone https://github.com/Jess-xca/Sales_Optimisation.git
   ```


2. **Set Up a Virtual Environment** (recommended, optional):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   - Install the required Python packages using `pip`:
     ```bash
     pip install streamlit pandas numpy matplotlib seaborn plotly scikit-learn
     ```
   - Or, run:
     ```bash
     pip install -r requirements.txt
     ```

4. **Verify Files**:
   - Ensure `supermarket_sales_dashboard.py` and `summary.md` are in the same directory.

---

## Running the Project

1. **Navigate to the Project Directory**:
   ```bash
   cd path/to/project
   ```

2. **Run the Streamlit App**:
   ```bash
   streamlit run supermarket_sales_dashboard.py
   ```

3. **Access the Dashboard**:
   - Open a web browser and go to `http://localhost:8501` (or the URL shown in the terminal).
   - Upload a compatible CSV file (see [Dataset Requirements](#dataset-requirements)) or download the sample CSV template from the dashboard to start analyzing data.

---

## Code Structure and Importance

The dashboard is implemented in `supermarket_sales_dashboard.py`, with a static summary in `summary.md`. Below is an overview of each section and its importance:

1. **Sample Dataset Download**:
   - **Functionality**: Allows users to download a sample CSV file with the expected headers and one example row, illustrating the required dataset format.
   - **Importance**: Helps users understand the dataset structure before uploading their data, reducing errors and ensuring compatibility.

2. **Data Loading and Preprocessing** (`load_data` function):
   - **Functionality**: Loads a CSV file, validates columns and data types, parses dates, engineers features (e.g., `Day_of_Week`, `Is_Weekend`), handles missing values, and removes outliers.
   - **Importance**: Ensures data quality and consistency, enabling reliable analysis and predictions. Feature engineering captures temporal and behavioral patterns critical for modeling.

3. **Sidebar Filters**:
   - **Functionality**: Allows users to filter data by city, gender, customer type, payment method, date range, unit price, and quantity.
   - **Importance**: Enables targeted analysis by focusing on specific segments, making the dashboard flexible for different business questions.

4. **Data Visualizations**:
   - **Correlation Heatmap**: Shows relationships between numeric features (e.g., `Unit price`, `Total`).
   - **Outlier Analysis**: Visualizes outliers in total sales via a box plot.
   - **Daily Sales Trend**: Plots sales over time to identify trends.
   - **Monthly Sales Seasonality**: Displays average sales by month (with month names) to highlight seasonal patterns.
   - **Sales by Product Line**: Compares total sales across product lines.
   - **Gross Income by Customer Type**: Analyzes profitability by customer type.
   - **Sales by Payment Method**: Shows sales distribution by payment method.
   - **Importance**: Visualizations provide intuitive insights into trends, seasonality, and customer behavior, guiding strategic decisions.

5. **Sales Prediction Models**:
   - **Functionality**: Trains Linear Regression and Random Forest models to predict total sales, with metrics (R², MSE, MAE) and a comparison plot.
   - **Importance**: The Random Forest model, tuned via GridSearchCV, provides accurate predictions by capturing complex patterns, serving as the basis for future sales forecasts.

6. **Future Sales Prediction (Next 3 Months)**:
   - **Functionality**: Predicts sales for the next 90 days by product line using the Random Forest model, visualized as a line chart.
   - **Importance**: Enables proactive planning by forecasting demand, critical for inventory and promotion strategies.

7. **Inventory Recommendations**:
   - **Functionality**: Calculates predicted units, safety stock, recommended stock, and reorder points for each product line, displayed in a table with notes for high-performing products.
   - **Importance**: Optimizes inventory to prevent overstock (reducing costs) and stockouts (avoiding lost sales), with actionable reorder points.

8. **Analysis Summary** (`summary.md`):
   - **Functionality**: Provides a static, non-technical summary of trends, seasonality, promotion impacts, predictions, and inventory recommendations.
   - **Importance**: Makes findings accessible to all stakeholders, ensuring business decisions are informed by clear insights.

9. **Footer**:
   - **Functionality**: Displays project credits.
   - **Importance**: Acknowledges the team and adds a professional touch.

---

## Key Concepts and Tools

### **Concepts**
- **Sales Analysis**: The dashboard identifies trends (e.g., weekend peaks), seasonal patterns (e.g., monthly variations), and inferred promotion impacts (e.g., high-sales periods). These insights help understand customer behavior and market dynamics.
- **Predictive Modeling**: Uses machine learning (Random Forest) to forecast sales based on features like unit price, quantity, and temporal factors. Predictions guide inventory and strategic planning.
- **Inventory Optimization**: Calculates stock levels to balance supply and demand, using safety stock and reorder points to minimize overstock (excess inventory) and stockouts (lost sales).
- **Data Preprocessing**: Ensures data quality through validation, missing value imputation, outlier removal, and feature engineering, critical for accurate analysis and modeling.
- **Interactive Visualization**: Provides dynamic, user-friendly plots and filters to explore data intuitively.

### **Tools and Packages**
- **Streamlit**: A Python library for building interactive web apps. Used to create the dashboard’s user interface, including filters, visualizations, and data displays.
  - Version: Compatible with Streamlit 1.0+.
  - Importance: Enables rapid development of a web-based, interactive dashboard without frontend expertise.
- **Pandas**: Handles data manipulation, filtering, and preprocessing (e.g., grouping, merging).
  - Importance: Core library for efficient data processing and analysis.
- **NumPy**: Supports numerical computations, such as outlier detection and feature calculations.
  - Importance: Provides fast array operations for data preprocessing.
- **Matplotlib/Seaborn**: Generates static visualizations (e.g., heatmaps, box plots, bar plots).
  - Importance: Produces publication-quality plots for exploratory analysis.
- **Plotly**: Creates interactive visualizations (e.g., line charts, bar charts).
  - Importance: Enhances user experience with zoomable, hoverable plots.
- **Scikit-learn**: Implements machine learning models (Linear Regression, Random Forest) and metrics (R², MSE, MAE).
  - Importance: Enables robust predictive modeling with hyperparameter tuning.
- **Python Standard Library**:
  - `calendar`: Converts month numbers to names (e.g., 1 → January).
  - `datetime`: Handles date parsing and future date generation.
  - Importance: Supports temporal feature engineering and user-friendly displays.

---

## Dataset Requirements

The dashboard expects a CSV file (`supermarket_sales.csv`) with the following columns:
- `Invoice ID`: Unique transaction identifier (string).
- `Branch`: Store branch name (string).
- `City`: City of the store (string).
- `Customer type`: Member or Normal (string).
- `Gender`: Customer gender (string).
- `Product line`: Product category (string).
- `Unit price`: Price per unit (numeric).
- `Quantity`: Number of units sold (numeric).
- `Tax 18%`: Tax amount (numeric).
- `Total`: Total sale amount (numeric).
- `Date`: Transaction date (format: MM/DD/YYYY).
- `Time`: Transaction time (format: HH:MM).
- `Payment`: Payment method (Cash, Credit card, Ewallet) (string).
- `cogs`: Cost of goods sold (numeric).
- `gross margin percentage`: Profit margin percentage (numeric).
- `gross income`: Profit from the sale (numeric).
- `Rating`: Customer rating (numeric).

**Notes**:
- Missing values are imputed (numeric: mean/median, categorical: mode or 'Unknown' for `Invoice ID`).
- Outliers in `Total` are removed using the IQR method.
- Ensure consistent date/time formats to avoid parsing errors.
- Download the sample CSV template from the dashboard to see the expected format.

---

## Future Improvements

- **Promotion Data**: Add a `Promotion` column to the dataset to directly analyze promotional impacts.
- **Configurable Parameters**: Allow users to adjust lead time and safety stock via sidebar inputs.
- **Enhanced Modeling**: Include categorical features (e.g., `Product line`) in the Random Forest model using one-hot encoding.
- **Export Options**: Enable downloading visualizations as images or inventory recommendations as PDFs.
- **Real-Time Validation**: Compare predictions against actual sales to refine the model.
- **Scalability**: Optimize for large datasets by caching expensive computations or adding pagination for tables.

---

## License
This project is for educational purposes and developed by US at AUCA for a Big Data Project (Prof Sunday IDOWU). Contact the team for usage permissions.

---

**Made with ❤️ by US | AUCA | Big Data Project**
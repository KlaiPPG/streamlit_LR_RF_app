# 🧪 Chemical Data Analysis Streamlit App

A simplified, user-friendly web interface for automated univariate regression and random forest analysis of chemical data.

## 🚀 Quick Start

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Run the App
```bash
streamlit run streamlit_app.py
```

### 3. Open in Browser
The app will automatically open in your browser at `http://localhost:8501`

## 📊 How to Use

1. **Upload Data**: Use the sidebar to upload your CSV or Excel file
2. **Select Sheet**: If Excel has multiple sheets, choose the one you want
3. **Choose Target**: Select the column you want to predict
4. **Configure**: Adjust preprocessing settings if needed
5. **Run Analysis**: Click the "Run Analysis" button
6. **View Results**: See univariate regression results, random forest performance, and visualizations
7. **Download**: Get CSV files of your results

## 🎯 Features

- **Automated Preprocessing**: Handles missing values, removes sparse columns, converts column names
- **Univariate Analysis**: Tests each feature individually with statistical significance
- **Random Forest**: Full predictive modeling with cross-validation
- **Interactive Visualizations**: Correlation heatmaps, predicted vs actual plots, feature importance
- **Excel Support**: Handles multiple sheets and various formats
- **Downloadable Results**: Export all analysis results as CSV files

## 📁 Supported Data Formats

- CSV files (`.csv`)
- Excel files (`.xlsx`, `.xls`)

## 🔧 Configuration Options

- **Missing Value Threshold**: Controls which columns to remove (default: 95% missing)
- **Significance Level**: Filter univariate results by p-value (0.001, 0.01, 0.05, 0.1)
- **Sheet Selection**: Choose specific Excel sheets to analyze

## 🧪 Designed for Chemical Data

This tool is specifically optimized for:
- HPLC/LCMS data with retention time columns
- Process data with many zero/missing measurements
- Impurity analysis and feature selection
- Chemical process optimization

## 💡 Tips for Best Results

1. **Clean Column Names**: The app handles RRT values and numeric column names automatically
2. **Target Selection**: Choose continuous numeric variables for regression
3. **Sample Size**: Works best with 20+ samples
4. **Feature Count**: Can handle hundreds of features efficiently

## 🔍 Understanding the Results

### Univariate Regression Table
- **Feature**: Column name from your data
- **Coefficient**: Slope of the relationship with target
- **p_value**: Statistical significance (lower = more significant)
- **R_squared**: How much variance this feature explains alone
- **Non_Zero_Count**: Number of non-zero measurements

### Random Forest Metrics
- **Train/Test R²**: Model performance on training and test data
- **CV R² Mean/Std**: Cross-validation performance and stability
- **Feature Importance**: Which features the model finds most predictive

## 🐛 Troubleshooting

**"No numeric columns found"**: Make sure your data contains numbers, not just text
**"Target column removed"**: Your target has too many missing values
**"File not found"**: Check file path and format
**App won't start**: Make sure all requirements are installed

## 🔄 Next Steps

This is a simplified version focusing on core functionality. You can extend it by:
- Adding more model types (Ridge, Lasso, SVM)
- Including correlation-based feature grouping
- Adding more visualization options
- Implementing batch processing for multiple files

---

*Built for easy chemical data analysis and modeling*

# Sales-forecasting
This project forecasts sales using Machine Learning models (Random Forest, XGBoost, ARIMA) with time-series data.
## 📌 Overview
This project aims to predict sales using various Machine Learning models, including Random Forest, XGBoost, and ARIMA, based on historical sales data.

## 📂 Files in the Repository
- **sales_forecasting.ipynb** - Jupyter Notebook with code and explanations.
- **data/** - Contains datasets used for training and evaluation.
- **submission.csv** - Final predictions for the test set.

## 🛠 Installation & Usage
1. Install dependencies:
   ```sh
  pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow statsmodels
Open the Jupyter Notebook:


jupyter notebook sales_forecasting.ipynb
Run all cells to process data, train models, and generate predictions.

📊 Model Comparison
Model	RMSE	MAPE
RandomForest: RMSE = 224.64, MAPE = 484290623118327872.00%
XGBoost: RMSE = 226.92, MAPE = 810310199853106432.00%
ARIMA: RMSE = 798.92, MAPE = 28059652193113329664.00%

Best Model: Random Forest performed best with the lowest RMSE and MAPE.

Key Insights:

Sales peak around paydays and holidays.

Promotions and oil prices slightly impact sales trends.

Time-based lag features improve forecasting accuracy.


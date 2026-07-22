# Telco Customer Churn Analysis & Prediction

This is a Data Analysis project about Customer Churn in a Telecom company.

The main goal is to find why customers leave (churn) and build basic models to predict churn risk. This helps the business team make better retention strategies.

## Project Structure

* `data/`: Contains original data and cleaned data.
* `src/`: Python scripts for data processing and analysis.
  * `data_loader.py`: Load raw data and clean missing values.
  * `eda.py`: Perform Exploratory Data Analysis (EDA) and create visual charts.
  * `preprocessing.py`: Prepare data for modeling (Encoding, Scaling, Split).
  * `models.py`: Train Logistic Regression and Random Forest models.
  * `evaluation.py`: Evaluate models and save result charts.
  * `main.py`: Run the whole pipeline.
* `outputs/`: Saved charts, tables, and insights.
  * `figures/`: EDA graphs, Confusion Matrices, ROC curves, Feature Importance.
  * `model_results.csv`: Comparison of model performance metrics.
* `models/`: Saved trained model files (`.pkl`).

## Key Analysis & Insights

Through Exploratory Data Analysis (EDA), here are some main findings:
* **Contract Type:** Customers with month-to-month contracts have a much higher churn rate compared to long-term contracts.
* **Payment Method:** Customers using Electronic Check leave more often.
* **Tenure:** New customers (short tenure) are more likely to churn than loyal customers.
* **Important Features:** Random Forest Feature Importance shows that tenure, monthly charges, and contract type are the top factors for churn.

## How to Run

1. Clone this repository:

```bash

git clone <your-repository-url>
cd Customer-Churn-Prediction
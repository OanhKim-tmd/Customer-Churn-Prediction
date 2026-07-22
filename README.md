![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Power BI](https://img.shields.io/badge/PowerBI-F2C811?style=for-the-badge&logo=Power%20BI&logoColor=white)

# Telco Customer Churn Analysis & Prediction

This is a Data Analysis project about Customer Churn in a Telecom company.

The main goal is to find why customers leave (churn) and build basic models to predict churn risk. This helps the business team make better retention strategies.

## Project Structure

* `dashboard/`: Contains the Power BI dashboard (`BI_prj.pbix`) and screenshots.
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
  * `models/`: Contains evaluation metrics like `cv_results.csv`, `feature_importance.csv`, and `model_results.csv`.

## 💡 Business Questions & Power BI Dashboard

In addition to the Machine Learning models, this project includes an interactive Power BI Dashboard to help business stakeholders monitor customer churn.

*(Upload an image of your dashboard to the `dashboard/screenshots/` folder and name it `overview.png`, then uncomment the line below to show it here)*
<!-- ![Overview Dashboard](dashboard/screenshots/overview.png) -->

This project answers key business questions regarding customer churn:
* **What is the overall churn rate?** The overall churn rate in this dataset is 26.54%.
* **How does churn vary by contract type?** Month-to-month contracts have the highest churn rate (42.71%), while Two-year contracts show strong retention with only 2.83% churn.
* **What is the average tenure of churned vs non-churned customers?** Churned customers have a shorter average tenure (17.98 months) compared to retained customers (37.57 months).
* **Does monthly charges influence churn?** Yes. Higher monthly charges (bands 68.50 to 98.65) correlate with higher churn rates (35-40%), whereas lower charges (18.25) see less than 10% churn.
* **What is the churn rate among customers with tech support?** Customers without tech support churn at a much higher rate (41.64%) compared to those with it (15.17%).

## 📊 Key Analysis & Insights (EDA)

Through Exploratory Data Analysis (EDA), here are some main findings:
* **Contract Type:** Customers with month-to-month contracts have a much higher churn rate compared to long-term contracts.
* **Payment Method:** Customers using Electronic Check leave more often.
* **Tenure:** New customers (short tenure) are more likely to churn than loyal customers.
* **Important Features:** Random Forest Feature Importance shows that tenure, monthly charges, and contract type are the top factors for churn.

## 🤖 Model Evaluation

We trained and evaluated Machine Learning models (Logistic Regression & Random Forest) to predict customer churn. Below are the visual results of our models:

### 1. Random Forest Performance
**Confusion Matrix & ROC Curve:**
<p align="center">
  <img src="outputs/figures/random_forest_confusion_matrix.png" width="45%" alt="RF Confusion Matrix"/>
  <img src="outputs/figures/random_forest_roc_curve.png" width="45%" alt="RF ROC Curve"/>
</p>

**Feature Importance:**
*(Tenure and Total/Monthly Charges are the most critical factors deciding if a customer will churn)*
<p align="center">
  <img src="outputs/figures/random_forest_feature_importance.png" width="70%" alt="Feature Importance"/>
</p>

### 2. Logistic Regression Performance
<p align="center">
  <img src="outputs/figures/logistic_regression_confusion_matrix.png" width="45%" alt="LR Confusion Matrix"/>
  <img src="outputs/figures/logistic_regression_roc_curve.png" width="45%" alt="LR ROC Curve"/>
</p>

## How to Run

1. Clone this repository:

```bash
git clone <your-repository-url>
cd Customer-Churn-Prediction
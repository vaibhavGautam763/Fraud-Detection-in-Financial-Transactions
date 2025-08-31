# Fraud-Detection-in-Financial-Transactions
Data Science Project

# Fraud Detection in Financial Transactions

**Author:** Vaibhav Gautam  
**Project Type:** Machine Learning / Data Science Internship Project  
**Dataset:** Financial transactions (~6.36M rows, 10 columns)

---

## Project Overview

This project focuses on developing a machine learning model to **detect fraudulent transactions** in a financial dataset. Fraud is rare but costly, and proactive detection helps minimize financial losses.

Key highlights:
- Cleaned and preprocessed large dataset
- Engineered features capturing unusual transaction behavior
- Built a **LightGBM model** for fraud prediction
- Interpreted model using **SHAP values**
- Provided actionable business insights

---

## Dataset Description

The dataset contains the following columns:

|       Column     |          Description         |
|------------------|------------------------------|
| `step`           | Time step of the transaction |
| `type`           | Transaction type (CASH_IN, CASH_OUT, etc.) |
| `amount`         | Transaction amount |
| `nameOrig`       | Origin account ID |
| `oldbalanceOrg`  | Original balance of origin account |
| `newbalanceOrig` | New balance of origin account |
| `nameDest`       | Destination account ID |
| `oldbalanceDest` | Original balance of destination account |
| `newbalanceDest` | New balance of destination account |
| `isFraud`        | Target label: 1 if fraud, 0 otherwise |
| `isFlaggedFraud` | Flagged as fraud by rules (binary) |

---

## Project Steps

### 1. Data Cleaning
- Handled missing values and outliers
- Created balance consistency features: `orig_balance_inconsistent`, `dest_balance_inconsistent`
- Flagged transactions where origin = destination

### 2. Feature Engineering
- Log transformation of `amount`
- Ratios: `amount / oldbalanceOrg`, `amount / oldbalanceDest`
- Encoded categorical `type`
- Added boolean flags: zero balances, inconsistent balances

### 3. Model Development
- **Model Used:** LightGBM (gradient boosting)
- **Imbalanced Data Handling:** `scale_pos_weight` and optional SMOTE
- **Train/Test Split:** 80/20 stratified

### 4. Model Evaluation
- Metrics: **ROC-AUC**, **PR-AUC**, **Precision@K**
- Confusion matrix to visualize performance
- SHAP analysis to understand feature importance

### 5. Key Findings
- Most important fraud predictors: `amount_over_oldOrg`, `log_amount`, balance inconsistency
- Model can flag high-risk transactions for human review

---

## Repository Structure

```
fraud-detection/
│
├── data/                           # Dataset CSV file
├── models/                         # Saved LightGBM model
├── fraud_detection_notebook.ipynb  # Complete notebook
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
```

---

## Business Recommendations

- Implement **real-time scoring** for high-risk transactions
- Use **human-in-the-loop** for flagged cases
- Monitor feature distribution changes to detect model drift
- Retrain periodically to adapt to new fraud patterns

---

## Contact

**Vaibhav Gautam**  
Email: gautamvaibhav020@gmail.com 
LinkedIn: https://www.linkedin.com/in/vaibhav-gautam-8b3ab71a9/


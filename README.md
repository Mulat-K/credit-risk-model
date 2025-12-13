# Credit Risk Probability Model Using Alternative Data

An end-to-end implementation for building, deploying, and automating a credit risk model using alternative (behavioral) data.

---

## Overview

This project develops a **Credit Scoring and Risk Probability Model** for a Buy-Now-Pay-Later (BNPL) service at **Bati Bank**, in partnership with an eCommerce platform. The solution transforms transactional behavioral data into predictive risk signals, enabling the bank to assess customer creditworthiness, assign credit scores, and optimize loan amount and duration decisions.

The project follows strong **engineering discipline**, **regulatory awareness (Basel II)**, and **MLOps best practices**, from data exploration to production deployment.

---

## Business Need

Bati Bank aims to launch a BNPL product that allows qualified customers to purchase goods on credit. Traditional credit bureau data is unavailable for many customers, so **alternative data** derived from transaction behavior must be used.

The core challenge is to:
- Infer customer credit risk **without a direct default label**
- Build a reliable, explainable, and scalable credit scoring system
- Ensure compliance with regulatory expectations and internal risk governance

---

## Credit Scoring Business Understanding

### Basel II Accord and the Need for Interpretable Models

The Basel II Capital Accord emphasizes **risk-sensitive capital requirements**, **model governance**, and **transparency** in credit risk measurement. Financial institutions must demonstrate that their models are not only predictive but also **interpretable, auditable, and well-documented**.

Because this model directly influences lending decisions and capital exposure, the approach must:
- Clearly document assumptions and feature transformations  
- Provide traceability from raw data to final risk scores  
- Enable validation by internal risk teams and regulators  

As a result, purely opaque “black-box” models are insufficient on their own. Interpretability is a first-class requirement.

---

### Proxy Default Variable and Business Risks

The dataset does not contain an explicit **loan default indicator**, which is typically required for supervised credit risk modeling. Since the BNPL product is new, historical repayment outcomes do not yet exist.

To address this, a **proxy target variable** is created using **Recency, Frequency, and Monetary (RFM)** behavioral metrics. Customers exhibiting disengaged behavior (low frequency, low monetary value, high recency) are labeled as **high-risk proxies**.

This approach enables model development but introduces business risks:
- **Label noise**: Not all disengaged customers will default
- **Bias risk**: Behavioral patterns may disadvantage specific customer groups
- **Decision risk**: Proxy-based predictions may diverge from true repayment behavior

These risks are mitigated through conservative thresholds, continuous monitoring, and future retraining once real default data becomes available.

---

### Model Complexity vs Interpretability Trade-offs

In regulated financial environments, there is a critical trade-off between **model performance** and **explainability**.

**Interpretable models (e.g., Logistic Regression with WoE)**:
- Easier to validate and explain to regulators
- Provide clear feature impact and direction
- Support stable scorecard development
- May underperform on complex, non-linear patterns

**Complex models (e.g., Gradient Boosting, Random Forests)**:
- Higher predictive accuracy and ROC-AUC
- Capture non-linear interactions
- Harder to explain and govern
- Increase operational and compliance complexity

This project evaluates multiple models and balances performance with regulatory suitability in line with Basel II expectations.

---

## Data Description

**Source:**  
Xente eCommerce Transaction Dataset (Kaggle)

### Key Fields

| Feature | Description |
|------|------------|
| TransactionId | Unique transaction identifier |
| AccountId | Customer account identifier |
| CustomerId | Unique customer identifier |
| ProductId | Item purchased |
| ProductCategory | Product category |
| ChannelId | Platform used (Web, Android, iOS, PayLater) |
| Amount | Transaction amount (debit/credit) |
| Value | Absolute transaction value |
| TransactionStartTime | Timestamp of transaction |
| PricingStrategy | Merchant pricing structure |
| FraudResult | Fraud indicator (1 = Fraud, 0 = Non-fraud) |

---

## Project Structure

```text
credit-risk-model/
│
├── .github/workflows/ci.yml
├── data/                     # Ignored in Git
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── data_processing.py
│   ├── train.py
│   ├── predict.py
│   └── api/
│       ├── main.py
│       └── pydantic_models.py
│
├── tests/
│   └── test_data_processing.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```
## Exploratory Data Analysis (EDA)

EDA is performed in `notebooks/eda.ipynb` and focuses on:

- Dataset structure and data types  
- Summary statistics  
- Distribution of numerical and categorical features  
- Correlation analysis  
- Missing value detection  
- Outlier identification using box plots  

### Key Insights

- Transaction values are highly right-skewed  
- Significant variance exists in customer engagement  
- Certain channels and product categories dominate transaction volume  
- Behavioral metrics strongly differentiate customer segments  

---

## Feature Engineering

Feature engineering is implemented using `sklearn.pipeline.Pipeline` to ensure reproducibility and consistency across training and inference.

### Aggregate Features

- Total transaction amount per customer  
- Average transaction amount  
- Transaction count  
- Standard deviation of transaction amounts  

### Time-Based Features

- Transaction hour  
- Day of month  
- Month  
- Year  

### Encoding and Scaling

- One-Hot Encoding for categorical variables  
- Standardization for numerical features  
- Missing value imputation (mean / median / mode)  

### WoE and Information Value

- Weight of Evidence (WoE) transformation  
- Information Value (IV) for feature selection  
- Improves interpretability and stability of logistic regression models  

---

## Proxy Target Variable Engineering

### RFM Metrics

For each customer:

- **Recency:** Days since last transaction  
- **Frequency:** Number of transactions  
- **Monetary:** Total transaction value  

### Clustering

- K-Means clustering with `k = 3`  
- Features scaled prior to clustering  
- Fixed `random_state` for reproducibility  

### High-Risk Label Definition

- Least engaged cluster labeled as `is_high_risk = 1`  
- Other clusters labeled as `0`  
- Target variable merged back into the training dataset  

---

## Model Training and Experiment Tracking

### Models Trained

- Logistic Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosting  

### Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC  

### MLflow

- Tracks model parameters, metrics, and artifacts  
- Enables comparison across multiple model runs  
- Registers the best-performing model in the Model Registry  

---

## Testing

Unit tests are implemented using `pytest`, covering:

- Feature engineering output validation  
- Schema consistency checks  
- Pipeline reproducibility tests  

---

## Deployment

### FastAPI Service

- `/predict` endpoint returns customer risk probability  
- Loads the best-performing model from the MLflow Model Registry  
- Input and output validation handled via Pydantic  

### Containerization

- `Dockerfile` for API service  
- `docker-compose.yml` for local orchestration  

---

## CI/CD Pipeline

GitHub Actions workflow with the following steps:

- Triggered on push to `main`  
- Runs code linting (`flake8` or `black`)  
- Executes unit tests using `pytest`  
- Fails the build on linting or test errors  

---

## Skills and Tools

### Skills

- Feature Engineering  
- Credit Risk Modeling  
- Model Governance  
- MLOps and CI/CD  

### Tools

- Python, scikit-learn  
- MLflow  
- FastAPI  
- Docker  
- GitHub Actions  
- Pytest  

---

## Author

**Analytics Engineer – Bati Bank**

This project demonstrates the full lifecycle of a production-grade credit risk model using alternative data, aligned with regulatory requirements and modern engineering best practices.

------------------------------------------------------------------------

## ⚙️ Installation & Setup

``` bash
git clone https://github.com/Mulat-K/credit-risk-model.git
cd credit-risk-model
```
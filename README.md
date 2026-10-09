# 🚗 VahanBima - Customer Lifetime Value (CLTV) Prediction

A high-performance machine learning predictive modeling project built for **VahanBima** to forecast Customer Lifetime Value (CLTV) and segment motor insurance policyholders.

---

## 📖 The Story & Problem Statement

Imagine sitting in the headquarters of **VahanBima**, one of India's leading motor vehicle insurance companies. The company provides motor vehicle insurance with 24/7 claim settlement across diverse regions in India. 

To enhance customer retention and satisfaction, VahanBima wants to launch exclusive VIP doorstep services, dedicated claims management resources, and personalized experience programs. However, to roll out these targeted perks, the company needs to answer one critical question: **Who are our most valuable customers?**

To solve this, the objective is to build an interpretable and high-performance machine learning model to predict the **Customer Lifetime Value (CLTV)**—the total financial worth a customer brings to the company over their lifetime—based on user demographics, policy attributes, and historical engagement data.

---

## 📊 Dataset Description

The dataset includes customer and policy information across three core splits:
* **`train.csv`**: ~90K records containing customer attributes, policy details, and the target variable (`cltv`).
* **`test.csv`**: ~60K records containing customer and policy attributes (target variable hidden for out-of-sample prediction).
* **`sample_submission.csv`**: Format template containing `id` and `cltv`.

### Data Dictionary
* **`id`**: Unique identifier of a customer
* **`gender`**: Gender of the customer
* **`area`**: Residential area type (Urban/Rural)
* **`qualification`**: Highest educational qualification of the user
* **`income`**: Annual income bracket earned by the customer (in Rupees)
* **`marital_status`**: Marital status (`0`: Single, `1`: Married)
* **`vintage`**: Number of years since the first policy date
* **`claim_amount`**: Total amount claimed by the customer (in Rupees)
* **`num_policies`**: Total number of policies issued by the customer
* **`policy`**: Active policy category of the customer
* **`type_of_policy`**: Tier type of active policy (e.g., Gold, Platinum)
* **`cltv`**: Customer Lifetime Value (**Target Variable**)

---

## ⚙️ Methodology & Approach

1. **Data Preprocessing & Encoding**: 
   * Handled mixed text categorical features using robust Label Encoding to convert human-readable categories into numerical representations.
   * Addressed missing values and ensured uniform alignment between training and testing datasets.
2. **Feature Engineering**: 
   * Constructed domain-specific behavioral ratios (such as claim-to-income and vintage-to-policy metrics) to provide richer signals for tree-based algorithms.
3. **Model Selection**: 
   * Leveraged **LightGBM** and **XGBoost** as primary gradient boosting regressors, chosen for their dominance in structured tabular datasets, fast execution speed, and robust handling of heterogeneous feature types.
4. **Validation Strategy**: 
   * Implemented rigorous **5-Fold Cross-Validation (K-Fold)** with early stopping to prevent overfitting, ensure stable generalization, and accurately evaluate model performance against unseen validation folds.

---

## 📈 Evaluation Metric & Results

* **Evaluation Metric**: $R^2$ (R-Squared) score, measuring the proportion of variance in customer lifetime values explained by the model.
* **Achieved Score**: The fine-tuned LightGBM and blended ensemble achieved a robust **Out-Of-Fold $R^2$ Score of ~0.1594**, successfully capturing core financial variance and satisfying production baseline requirements.

---

## 🚀 Repository Structure

```text
├── analysis.ipynb          # Main Jupyter Notebook containing EDA, preprocessing, and model training
├── final_submission.csv    # Generated test predictions formatted for deployment
└── README.md               # Project documentation

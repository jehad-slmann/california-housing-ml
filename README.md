# California Housing Price Prediction

## Overview

This project builds **Regression models** to predict housing prices using the **California Housing dataset** from `sklearn`.

It covers the full workflow:

* Data loading
* Exploration
* Preprocessing
* Model training
* Evaluation

---

## Dataset

The dataset is loaded using:

```python
from sklearn.datasets import fetch_california_housing
```

It contains features such as:

* Median income
* House age
* Average rooms
* Population
* Location (latitude & longitude)

**Target variable:**

* `MedHouseVal` → Median house value

---

## Workflow

### 1. Import Libraries

* numpy
* sklearn (datasets, preprocessing, model)

### 2. Load Data

```python
df = fetch_california_housing(as_frame=True).frame
```

---

### 3. Data Exploration

* Check missing values:

```python
df.isna().sum()
```

* Summary statistics:

```python
df.describe()
```

---

### 4. Feature & Target Split

```python
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]
```

---

### 5. Train-Test Split

```python
train_test_split(X, y, test_size=0.2, random_state=42)
```

---

### 6. Feature Scaling

Standardization is applied using:

```python
StandardScaler()
```

---

### 7. Model Training

A **Linear Regression model** is trained:

```python
LinearRegression().fit(x_train_scaled, y_train)
```

---

### 8. Model Evaluation

Model performance is evaluated using:

```python
model.score(x_test_scaled, y_test)
```
Result of: 60%
---

### 9. RF Model Training

A **Random Forest Regressor** is trained:

```python
RandomForestRegressor().fit(x_train_scaled, y_train)
```
---

### 10. RF Model Evaluation

Model performance is evaluated using:

```python
model.score(x_test_scaled, y_test)
```
Result of: 80%
---

## Results

* The Random Forest model outputs an **R² score** indicating how well it explains variance in housing prices.
* Higher values → better performance.
 --> Random Forest model perform better that Linear Regression model
---

## Key Insights

* Median income is the strongest predictor of house prices
* Location (latitude & longitude) also plays a significant role
* Linear models struggle because relationships are non-linear
---

## Model Comparison

* Linear Regression (R² ≈ 0.60)
 --> Limited by assumption of linear relationships
* Random Forest (R² ≈ 0.80)
 --> Captures non-linear interactions and feature dependencies
---

## How to Run

1. Install dependencies:

```bash
pip install  numpy scikit-learn
```

2. Open the notebook in VS Code or Jupyter:

```bash
jupyter notebook
```

3. Run all cells.

---


## Author

Jehad Slman

---

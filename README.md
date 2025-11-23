# 🤖 Model_Hub – Machine Learning Model Trainer

Model_Hub is a Streamlit-based interactive application that allows users to upload datasets, preprocess data, train multiple ML models, and view performance metrics — all without writing code.

---

## 🚀 Features

### 📂 Data Handling
- Upload CSV files
- Preview dataset
- Label Encoding for categorical columns
- Scaling options:
  - StandardScaler
  - MinMaxScaler

---

### ⚙ Machine Learning Models Supported
| Model Type | Algorithm |
|------------|------------|
| Regression | Linear Regression, Decision Tree Regressor |
| Classification | Logistic Regression, Decision Tree Classifier |

---

### 📊 Metrics Output

#### **Regression**
- R² Score
- Mean Squared Error (MSE)

#### **Classification**
- Accuracy
- Precision
- Recall
- F1 Score

---

## 🛠 Tech Stack

| Library | Usage |
|---------|--------|
| Streamlit | UI & App |
| Pandas / NumPy | Data handling |
| Scikit-learn | ML Models & Metrics |

---

## 🔧 Installation

```bash
pip install -r requirements.txt

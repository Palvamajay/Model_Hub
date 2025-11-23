# creating the data science app for the Algorithms 
# loading the libraries 
import pandas as pd
import numpy as np 
import streamlit as st 
import warnings as wt 
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import accuracy_score,mean_squared_error,precision_score,recall_score,f1_score, r2_score


# giving the title to the project 

st.markdown(
    "<h1 style='text-align: center; color: #4682B4; font-size: 45px;'>🤖Model_Hub</h1>",
    unsafe_allow_html=True
)

# Uploading the data 
st.warning("uploade the cleaned data")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# Load only once
if uploaded_file is not None and st.session_state.get("df") is None:
    st.session_state.df = pd.read_csv(uploaded_file)
df = st.session_state.get("df")

# Stop page until CSV uploaded
if df is None:
    st.stop()

# Reviewing the data         
st.subheader("🔍 Data Preview")
if st.button("Review"):
  st.write(df.head())

st.markdown("---")
# ===== LABEL ENCODING =====
# Get object-type columns

col1, col2, col3 = st.columns(3)

# ------------------ ENCODING ------------------

with col1:
    st.subheader("Encoding")
    object_cols = df.select_dtypes(include="object").columns
    cols_label = st.multiselect("Select columns to encode", object_cols, key="encode_cols")
    if st.button("Apply Encoding"):
        le = LabelEncoder()
        for col in cols_label:
            df[col] = le.fit_transform(df[col])
        st.success("Encoding Applied!")
       


# ---------- SCALING ----------
with col2:
    st.subheader("Scaling")
    # IMPORTANT: Recalculate after encoding
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    scale_cols = st.multiselect("Select columns to scale", num_cols, key="scale_cols_new")
    if st.button("Apply Scaling"):
        if len(scale_cols) > 0:
            scaler = StandardScaler()
            df[scale_cols] = scaler.fit_transform(df[scale_cols])
            st.success("Scaling applied successfully!")
        else:
            st.warning("Select columns first!")

# Min Max scalar
with col3:
    st.subheader("Min_Max_scalar")
    select_mn=st.multiselect("Select columns to scale", num_cols, key="select_mn_cols_new")
    if st.button("Appily MinMax"):
      if len(scale_cols) > 0:
            scaler_mn = MinMaxScaler()
            df[scale_cols] = scaler_mn.fit_transform(df[scale_cols])
            st.success("Scaling applied successfully!")
      else:
            st.warning("Select columns first!")

st.markdown("---")
# ===== TRAIN/TEST SPLIT =====
st.subheader("Data_Split")
y = st.selectbox("Select target variable", df.columns, key="target")
x = st.multiselect("Select feature variable", df.columns, key="features")

if st.button("Split Data"):
    X = df[x]
    Y = df[y]
    st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = \
        train_test_split(X, Y, test_size=0.2, random_state=50)
    st.success("Data split successfully!")
    
st.markdown("---")

st.subheader("Select Model")

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Linear Regression"):
        st.session_state.selected_model = "linear"
        st.success("Linear Regression Selected")

with col2:
    if st.button("Logistic  Regression"):
        st.session_state.selected_model = "logistic"
        st.success("Logistic Regression Selected")

with col3:
    if st.button("DecisionTree Regressor"):
        st.session_state.selected_model = "dt_reg"
        st.success("Decision Tree Regressor Selected")

with col4:
    if st.button("DecisionTree Classifier"):
        st.session_state.selected_model = "dt_clf"
        st.success("Decision Tree Classifier Selected")



# Now train model based on selection
if st.session_state.selected_model == "linear":
    st.markdown("---")
    st.subheader("Training Linear Regression Model")
    if st.button("Train Regression Model"):
        model = LinearRegression()
        model.fit(st.session_state.X_train, st.session_state.y_train)
        st.session_state.y_pred = model.predict(st.session_state.X_test)
        st.success("Model trained successfully!")

elif st.session_state.selected_model == "logistic":
    st.markdown("---")
    st.subheader("Training Logistic Regression Model")
    if st.button("Train Logistic Regression"):
        model = LogisticRegression()
        model.fit(st.session_state.X_train, st.session_state.y_train)
        st.session_state.y_pred = model.predict(st.session_state.X_test)
        st.success("Model trained successfully!")

elif st.session_state.selected_model == "dt_reg":
    st.markdown("---")
    st.subheader("Training DecisionTree Regressor")
    if st.button("Train DecisionTree Regressor"):
        model = DecisionTreeRegressor()
        model.fit(st.session_state.X_train, st.session_state.y_train)
        st.session_state.y_pred = model.predict(st.session_state.X_test)
        st.success("Model trained successfully!")

elif st.session_state.selected_model == "dt_clf":
    st.markdown("---")
    st.subheader("Training DecisionTree Classifier")
    if st.button("Train DecisionTree Classifier"):
        model = DecisionTreeClassifier()
        model.fit(st.session_state.X_train, st.session_state.y_train)
        st.session_state.y_pred = model.predict(st.session_state.X_test)
        st.success("Model trained successfully!")

#===========================================================================================================================================================
st.markdown("---")
# ===== METRICS =====
st.header("Predict the score")

if st.button("Click here"):
    try:
        model_type = st.session_state.selected_model
        y_test = st.session_state.y_test
        y_pred = st.session_state.y_pred

        # ----- REGRESSION METRICS -----
        if model_type in ["linear", "dt_reg"]:
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)

            st.subheader("📊 Regression Metrics")
            st.write("**R² Score:**", r2)
            st.write("**MSE Score:**", mse)

        # ----- CLASSIFICATION METRICS -----
        elif model_type in ["logistic", "dt_clf"]:
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted')
            rec = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            st.subheader("📈 Classification Metrics")
            st.write("**Accuracy:**", acc)
            st.write("**Precision:**", prec)
            st.write("**Recall:**", rec)
            st.write("**F1 Score:**", f1)
            
        else:
            st.warning("⚠ Model type unknown. Please train a model first.")

    except KeyError:
        st.error("❌ Train the model before calculating metrics.")


#streamlit
#pandas
#numpy
#scikit-learn
        

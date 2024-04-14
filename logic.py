import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import streamlit as st

def load_data(file):
    return pd.read_csv(file)

def display_dataset_info(data):
    st.subheader("Thông tin dataset:")
    st.write(data)

def dropna_data(data):
    data.dropna(axis=0)

def preprocess_data(data):
    label_encoder = LabelEncoder()
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        data[col] = label_encoder.fit_transform(data[col])

    # Feature scaling sử dụng StandardScaler
    scaler = StandardScaler()
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    
    return data

def train_model(model_type, X_train, y_train):
    if model_type == 'Linear Regression':
        model = LinearRegression()
    elif model_type == 'Logistic Regression':
        model = LogisticRegression()
    else:
        model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, model_type, X_train, y_train, X_test, y_test):
    if model_type != 'KNN':
        y_pred = model.predict(X_test)
        if model_type == 'Linear Regression':
            mse = mean_squared_error(y_test, y_pred)
            return mse
        elif model_type == 'Logistic Regression':
            accuracy = accuracy_score(y_test, y_pred)
            return accuracy
    else:
        return None

def predict(model, X_test):
    # Dự đoán kết quả từ mô hình và trả về kết quả dự đoán
    return model.predict(X_test)


def main_logic(uploaded_file):
    if uploaded_file is not None:
        # Đọc dữ liệu từ tập tin CSV
        data = load_data(uploaded_file)

        # Tiền xử lý dữ liệu
        data = preprocess_data(data)

        return data

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from logic import load_data, preprocess_data, train_model, evaluate_model, predict, display_dataset_info, dropna_data
from sklearn.model_selection import train_test_split

def main():
    st.title('Phân tích Dataset với Shiny')
    
    # Chọn tập tin dữ liệu để import
    uploaded_file = st.file_uploader("Chọn tập tin CSV", type="csv")
    
    if uploaded_file is not None:
        # Đọc dữ liệu từ tập tin CSV
        data = load_data(uploaded_file)

        # Tiền xử lý dữ liệu
        data = preprocess_data(data)

        # Phần UI
        st.sidebar.title('Phân tích Dataset với Shiny')
        st.sidebar.subheader(' ')

         # Button to display data
        display_data_button = st.sidebar.button("Hiển thị dữ liệu", key="display_data")


         # Button to display data
        display_data_button_dropna = st.sidebar.button("Dữ liệu sau khi xử lý", key="display_data_dropna")
        
        y_column = st.sidebar.selectbox('Chọn biến Y:', options=data.columns.tolist())
        
        # Loại bỏ biến Y đã chọn từ danh sách tùy chọn của biến X
        x_options = [col for col in data.columns.tolist() if col != y_column]
        
        # Dropdown cho biến X
        x_column = st.sidebar.selectbox('Chọn biến X:', options=x_options)
        
        # Dropdown cho mô hình
        model_type = st.sidebar.selectbox('Chọn loại mô hình:', options=['Linear Regression', 'Logistic Regression', 'KNN'])
        

        # Phần UI bên phải
        st.subheader('Bên phải')

        if display_data_button:
            # Hiển thị dữ liệu
            st.write(data)
            st.dataframe(data)

        if display_data_button_dropna:
            st.write(data)
            dropna_data(data)
            st.write(data)
        
        # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
        X = data.drop(columns=[y_column])
        y = data[y_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Đào tạo mô hình
        model = train_model(model_type, X_train, y_train)

        # Training the model
        model.fit(X_train, y_train)

        # Making predictions
        y_pred = model.predict(X_test)

        # Đánh giá mô hình
        evaluation_result = evaluate_model(model, model_type, X_train, y_train, X_test, y_test)

        # Kết quả dự đoán
        prediction = predict(model, X_test)

        # Hiển thị kết quả
        st.write('Result:', evaluation_result)
        st.write('Prediction:', prediction)

        # Hiển thị biểu đồ
        st.subheader('Biểu đồ')

        # Scatter plot of actual data
        plt.scatter(X_test[x_column], y_test)
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        st.pyplot()

        # Line plot of predicted values
        plt.plot(X_test[x_column], y_pred, color='red')
        plt.xlabel(x_column)
        plt.ylabel(y_column + ' (predicted)')
        st.pyplot()

if __name__ == "__main__":
    main()

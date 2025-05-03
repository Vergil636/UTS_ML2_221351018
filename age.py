import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

# Load scaler dan label encoder
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="Age_group.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul Aplikasi
st.title("Age Group Prediction System")
st.write("Masukkan data kesehatan untuk memprediksi kelompok usia.")

# Form input pengguna
age = st.number_input("Usia", min_value=1, max_value=120, value=30)
gender = st.selectbox("Jenis Kelamin", options=[1, 2], format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan")
bmi = st.number_input("Body Mass Index (BMI)", min_value=1.0, max_value=100.0, value=25.0)
glucose = st.number_input("Glukosa darah setelah puasa", min_value=10.0, max_value=200.0, value=100.0)
diabetic = st.selectbox("Diabetes atau Tidak", options=[1, 2], format_func=lambda x: "Ya" if x == 1 else "Tidak")
oral = st.number_input("Respon Oral Glukosa", min_value=0.0, max_value=300.0, value=100.0)
insulin = st.number_input("Kadar Insulin Darah", min_value=0.0, max_value=500.0, value=15.0)

# Tombol prediksi
if st.button("Prediksi Kelompok Usia"):
    # Susun data input
    input_data = np.array([[age, gender, bmi, glucose, diabetic, oral, insulin]])
    input_scaled = scaler.transform(input_data).astype(np.float32)

    # Inference dengan model TFLite
    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    predicted_label = np.argmax(prediction)
    age_group = label_encoder.inverse_transform([predicted_label])[0]

    st.success(f"Kelompok usia yang diprediksi: **{age_group.upper()}**")

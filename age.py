import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# Load scaler dan model TFLite
scaler = joblib.load("scaler.pkl")
interpreter = tf.lite.Interpreter(model_path="Age_group.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Mapping dari indeks label ke kategori usia
label_mapping = {
    0: "Remaja",
    1: "Dewasa",
    2: "Lansia"
}

# Judul Aplikasi
st.title("Prediksi Golongan Usia Berdasarkan Karakteristik Fisik")
st.write("Masukkan data untuk memprediksi golongan usia.")

# Form input pengguna
gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
age = st.number_input("Umur (Tahun)", min_value=0, max_value=100, value=30)
bmi = st.number_input("Indeks Massa Tubuh (BMI)", min_value=10.0, max_value=60.0, value=22.0)
bgaf = st.number_input("Glukosa Darah Setelah Puasa", min_value=50.0, max_value=200.0, value=90.0)
diabetes = st.selectbox("Apakah Mengidap Diabetes?", ["Ya", "Tidak"])
oral = st.number_input("Respon Oral", min_value=0.0, max_value=100.0, value=50.0)
bloodinsulan = st.number_input("Tingkat Insulin Darah", min_value=0.0, max_value=300.0, value=80.0)

# Encoding
gender_encoded = 1 if gender == "Laki-laki" else 0
diabetes_encoded = 1 if diabetes == "Ya" else 0

# Prediksi
if st.button("Prediksi Kategori Usia"):
    input_data = np.array([[gender_encoded, age, bmi, bgaf, diabetes_encoded, oral, bloodinsulan]])
    input_scaled = scaler.transform(input_data).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    # Tangani output sebagai klasifikasi atau regresi
    if prediction.shape[1] > 1:
        predicted_label_index = int(np.argmax(prediction))
    else:
        predicted_label_index = int(round(prediction[0][0]))

    predicted_label = label_mapping.get(predicted_label_index, "Tidak diketahui")
    st.success(f"Golongan usia Anda adalah: **{predicted_label}**")

import streamlit as st
import numpy as np
import pickle
import os

# === Definisi class Perceptron (wajib sebelum pickle.load) ===
# Catatan:
# Meskipun model Perceptron sudah pernah dibuat dan dilatih di notebook, 
# di file Python (app.py) ini kita tetap harus menuliskan ulang definisi kelasnya. 
# Alasannya, pickle hanya menyimpan parameter dan bobot model, bukan kode kelasnya. 
# Saat model di-load, Python perlu tahu kembali bagaimana struktur kelas tersebut. 
# Jika definisi kelas tidak tersedia, akan muncul error AttributeError karena Python 
# tidak mengenali "Perceptron" yang tersimpan di file pickle.
class Perceptron:
    """Klasifikasi Perceptron sederhana."""

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)


# === 1) Load model Perceptron ===
model_path = os.path.join(os.path.dirname(__file__), "perceptron_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# === 2) Judul Aplikasi ===
st.title("🌸 Prediksi Bunga Iris — Perceptron")
st.title("Denis — 230712245")  # isikan dengan nama dan npm praktikan
st.write("Masukkan panjang sepal dan petal untuk memprediksi jenis bunga (Setosa=0, Versicolor=1).")

# === 3) Input User ===
sepal_length = st.number_input("Sepal Length [cm]", min_value=0.0, max_value=10.0, value=5.5, step=0.1)
petal_length = st.number_input("Petal Length [cm]", min_value=0.0, max_value=10.0, value=4.0, step=0.1)

# === 4) Buat array data baru ===
X_new = np.array([[sepal_length, petal_length]])

# === 5) Prediksi ===
if st.button("Prediksi"):
    y_pred = model.predict(X_new)
    label = "Setosa (0)" if y_pred[0] == 0 else "Versicolor (1)"
    st.success(f"Hasil Prediksi: **{label}**")


# ==========================================================
# Cara Menjalankan Aplikasi Streamlit ini:
#
# 1. Buka terminal / command prompt:
#    - Di VS Code: tekan Ctrl + Shift + ~
#    - Atau buka Command Prompt / Anaconda Prompt secara manual
#
# 2. Arahkan terminal ke folder tempat file app.py ini disimpan:
#    cd "D:\Kuliah\Semester 7\Asdos Mesin\Praktikum 1"
#
# 3. Pastikan library streamlit sudah terinstal:
#    pip install streamlit
#
# 4. Jalankan perintah berikut untuk menjalankan aplikasi:
#    streamlit run app.py
#
# 5. Setelah itu aplikasi akan otomatis terbuka di browser
#    (default: http://localhost:8501)
#
# Catatan: pastikan file 'perceptron_model.pkl' ada di folder
# yang sama dengan app.py, agar model bisa di-load dengan benar.
# ==========================================================

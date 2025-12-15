import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Konfigurasi Halaman
st.set_page_config(page_title="Analisis Data Siswa", layout="wide")

st.title("Dashboard Analisis: Clustering & Logistic Regression")
st.markdown("""
Aplikasi ini menampilkan alur pemrosesan data setahap demi setahap:
1. **Raw Data**: Melihat data asli.
2. **Preprocessing**: Pembersihan dan transformasi data.
3. **Clustering**: Pengelompokan siswa berdasarkan pola jawaban.
4. **Logistic Regression**: Prediksi tingkat stres siswa.
""")

# --- BAGIAN 1: LOAD RAW DATA ---
st.header("1. Raw Data (Data Mentah)")
uploaded_file = 'responses.csv' 

# Fungsi cache agar data tidak di-load berulang kali setiap interaksi
@st.cache_data
def load_data():
    try:
        return pd.read_csv(uploaded_file)
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error("File 'responses.csv' tidak ditemukan! Pastikan file ada di folder yang sama dengan app.py")
    st.stop()

st.write("Menampilkan 5 baris pertama dari data mentah:")
st.dataframe(df.head())

col1, col2 = st.columns(2)
with col1:
    st.write("**Info Dataset:**")
    st.write(f"Jumlah Baris: {df.shape[0]}")
    st.write(f"Jumlah Kolom: {df.shape[1]}")
with col2:
    st.write("**Cek Missing Values (Jumlah Kosong):**")
    st.dataframe(df.isnull().sum())

# --- BAGIAN 2: DATA CLEANING & PREPROCESSING ---
st.header("2. Data Cleaning & Preprocessing")
st.info("Proses mengubah data mentah menjadi siap untuk Machine Learning.")

# Step 2.1: Handling Missing Values
st.subheader("2.1 Penanganan Missing Values")
st.write("Masalah: Kolom `State` memiliki banyak data kosong, dan `Gender` memiliki beberapa kosong.")

# Proses Cleaning
df_clean = df.copy()
# Drop State
df_clean = df_clean.drop(columns=['State'])
# Impute Gender
gender_mode = df_clean['Gender'].mode()[0]
df_clean['Gender'] = df_clean['Gender'].fillna(gender_mode)

st.write("Solusi: Kolom `State` dihapus. Kolom `Gender` diisi dengan modus (nilai terbanyak).")
st.write("**Hasil (Cek ulang missing values):**")
st.dataframe(df_clean.isnull().sum().to_frame(name='Jumlah Kosong').T)

# Step 2.2: Encoding
st.subheader("2.2 Encoding (Mengubah Teks ke Angka)")
st.write("Masalah: Algoritma tidak bisa membaca teks seperti 'Male', 'Female', atau 'US'.")

# Encoding
le = LabelEncoder()
categorical_cols = ['Category', 'Country', 'Gender', 'Before-Environment', 'Now-Environment']
df_encoded = df_clean.copy()

for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])

st.write("Solusi: Menggunakan Label Encoding.")
st.write("**Hasil Data Setelah Encoding:**")
st.dataframe(df_encoded.head())

# Step 2.3: Scaling
st.subheader("2.3 Scaling (Normalisasi Data)")
st.write("Masalah: Rentang angka antar kolom berbeda jauh (misal: Umur 15-20, tapi Stress 1-10). Ini buruk untuk Clustering.")

scaler = StandardScaler()
numerical_cols = ['Age', 'Before-ClassworkStress', 'Before-HomeworkStress', 'Before-HomeworkHours',
                  'Now-ClassworkStress', 'Now-HomeworkStress', 'Now-HomeworkHours',
                  'FamilyRelationships', 'FriendRelationships']

df_scaled = df_encoded.copy()
df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])

st.write("Solusi: Menggunakan StandardScaler.")
st.write("**Hasil Data Setelah Scaling (Siap untuk Clustering):**")
st.dataframe(df_scaled.head())

# --- BAGIAN 3: CLUSTERING ---
st.header("3. Clustering (K-Means)")
st.markdown("Mengelompokkan siswa ke dalam beberapa cluster berdasarkan kemiripan data.")

# Input User untuk Jumlah Cluster
k = st.slider("Pilih Jumlah Cluster (K):", min_value=2, max_value=5, value=3)

# Proses Clustering
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(df_scaled)

# Gabungkan hasil ke dataframe asli agar mudah dibaca
df_result_cluster = df.copy()
df_result_cluster['Cluster'] = clusters

# Visualisasi dengan PCA (Reduksi Dimensi ke 2D)
pca = PCA(n_components=2)
pca_comp = pca.fit_transform(df_scaled)
df_result_cluster['PCA1'] = pca_comp[:, 0]
df_result_cluster['PCA2'] = pca_comp[:, 1]

c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("Visualisasi Sebaran Cluster")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df_result_cluster, x='PCA1', y='PCA2', hue='Cluster', palette='viridis', s=100, ax=ax)
    plt.title(f"K-Means Clustering dengan {k} Cluster")
    st.pyplot(fig)

with c2:
    st.subheader("Statistik Cluster")
    st.write("Rata-rata fitur per cluster:")
    # Menampilkan rata-rata beberapa kolom penting
    cols_to_show = ['Age', 'Now-ClassworkStress', 'Now-HomeworkHours']
    cluster_stats = df_result_cluster.groupby('Cluster')[cols_to_show].mean()
    st.dataframe(cluster_stats)

# --- BAGIAN 4: LOGISTIC REGRESSION ---
st.header("4. Logistic Regression")
st.markdown("Membuat model untuk memprediksi: **Apakah siswa mengalami Stress Tinggi pada Tugas Kelas (Classwork Stress)?**")

# Target Definisi
st.write("Target: Jika `Now-ClassworkStress` >= 4, maka **High Stress (1)**. Jika tidak, **Low Stress (0)**.")

# Persiapan Data
# Gunakan data yang sudah di-encode (df_encoded)
X = df_encoded.drop(columns=['Now-ClassworkStress'])
y = df_encoded['Now-ClassworkStress'].apply(lambda x: 1 if x >= 4 else 0)

# Split Data
test_size = st.slider("Tentukan ukuran data testing (%):", 10, 50, 20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

if st.button("Latih Model Logistic Regression"):
    # Training
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluasi
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    st.success(f"Model berhasil dilatih! Akurasi: {acc:.2%}")
    
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        st.write("**Confusion Matrix:**")
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        ax_cm.set_xlabel('Prediksi')
        ax_cm.set_ylabel('Aktual')
        st.pyplot(fig_cm)
        st.caption("0: Low Stress, 1: High Stress")
        
    with col_res2:
        st.write("**Classification Report:**")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

st.divider()
st.caption("Dibuat dengan Streamlit • Analisis Data Siswa")
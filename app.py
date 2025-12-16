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

st.title("Dashboard Analisis & Prediksi Stres Siswa")
st.markdown("""
Aplikasi ini menampilkan alur pemrosesan data setahap demi setahap:
1. **Raw Data**: Melihat data asli.
2. **Preprocessing**: Pembersihan dan transformasi data.
3. **Clustering**: Pengelompokan siswa berdasarkan pola jawaban.
4. **Logistic Regression**: Evaluasi model.
5. **Simulasi Prediksi**: Coba prediksi data baru!
""")

# --- BAGIAN 1: LOAD RAW DATA ---
st.header("1. Raw Data (Data Mentah)")
uploaded_file = 'responses.csv' 

@st.cache_data
def load_data():
    try:
        return pd.read_csv(uploaded_file)
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error("File 'responses.csv' tidak ditemukan! Pastikan file ada di folder yang sama.")
    st.stop()

st.write("Menampilkan 5 baris pertama dari data mentah:")
st.dataframe(df.head())

# --- BAGIAN 2: DATA CLEANING & PREPROCESSING ---
st.header("2. Data Cleaning & Preprocessing")

# 2.1 Handling Missing Values
df_clean = df.copy()
df_clean = df_clean.drop(columns=['State'])
gender_mode = df_clean['Gender'].mode()[0]
df_clean['Gender'] = df_clean['Gender'].fillna(gender_mode)

# 2.2 Encoding (DIPERBARUI UNTUK PREDIKSI)
st.subheader("2.2 Encoding & Scaling")
st.write("Mengubah data teks menjadi angka dan menormalisasi skala data.")

# Kita simpan encoder dalam dictionary agar bisa dipakai ulang saat Prediksi nanti
encoders = {} 
categorical_cols = ['Category', 'Country', 'Gender', 'Before-Environment', 'Now-Environment']
df_encoded = df_clean.copy()

for col in categorical_cols:
    le = LabelEncoder()
    # Fit pada data asli agar mapping tersimpan (misal: Male=1, Female=0)
    df_encoded[col] = le.fit_transform(df_encoded[col])
    encoders[col] = le # Simpan encoder ini

st.write("Data telah di-encode.")

# 2.3 Scaling
scaler = StandardScaler()
numerical_cols = ['Age', 'Before-ClassworkStress', 'Before-HomeworkStress', 'Before-HomeworkHours',
                  'Now-ClassworkStress', 'Now-HomeworkStress', 'Now-HomeworkHours',
                  'FamilyRelationships', 'FriendRelationships']

df_scaled = df_encoded.copy()
df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])

st.dataframe(df_scaled.head(3))

# --- BAGIAN 3: CLUSTERING ---
st.header("3. Clustering (K-Means)")
k = st.slider("Pilih Jumlah Cluster (K):", 2, 5, 3)
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(df_scaled)

df_result_cluster = df.copy()
df_result_cluster['Cluster'] = clusters

pca = PCA(n_components=2)
pca_comp = pca.fit_transform(df_scaled)
df_result_cluster['PCA1'] = pca_comp[:, 0]
df_result_cluster['PCA2'] = pca_comp[:, 1]

c1, c2 = st.columns([2, 1])
with c1:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=df_result_cluster, x='PCA1', y='PCA2', hue='Cluster', palette='viridis', s=100, ax=ax)
    st.pyplot(fig)
with c2:
    st.write("Rata-rata fitur per cluster:")
    st.dataframe(df_result_cluster.groupby('Cluster')[['Age', 'Now-ClassworkStress']].mean())

# --- BAGIAN 4: LOGISTIC REGRESSION (TRAINING) ---
st.header("4. Logistic Regression Evaluation")

# Persiapan Data
X = df_encoded.drop(columns=['Now-ClassworkStress'])
y = df_encoded['Now-ClassworkStress'].apply(lambda x: 1 if x >= 4 else 0)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training Model (Otomatis dijalankan agar siap untuk prediksi)
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.write(f"Model telah dilatih dengan **Akurasi: {acc:.2%}**")
st.write("Classification Report:")
st.text(classification_report(y_test, y_pred))

# --- BAGIAN 5: PREDIKSI (SIMULASI) ---
st.markdown("---")
st.header("5. Simulasi Prediksi Stres Siswa")
st.info("Masukkan data siswa di bawah ini (Sidebar/Form) untuk memprediksi apakah mereka mengalami stres tinggi atau tidak.")

# Layout Input dengan Kolom
col_input1, col_input2, col_input3 = st.columns(3)

# Dictionary untuk menampung input user
user_input = {}

# --- Kolom 1: Data Profil ---
with col_input1:
    st.subheader("Profil Siswa")
    user_input['Category'] = st.selectbox("Category", options=encoders['Category'].classes_)
    user_input['Country'] = st.selectbox("Country", options=encoders['Country'].classes_)
    user_input['Gender'] = st.selectbox("Gender", options=encoders['Gender'].classes_)
    user_input['Age'] = st.number_input("Age", min_value=10, max_value=30, value=16)

# --- Kolom 2: Kondisi Sebelum (Before) ---
with col_input2:
    st.subheader("Kondisi Sebelumnya")
    user_input['Before-Environment'] = st.selectbox("Before Environment", options=encoders['Before-Environment'].classes_)
    user_input['Before-ClassworkStress'] = st.slider("Before Classwork Stress (1-6)", 1, 6, 3)
    user_input['Before-HomeworkStress'] = st.slider("Before Homework Stress (1-6)", 1, 6, 3)
    user_input['Before-HomeworkHours'] = st.number_input("Before Homework Hours", min_value=0.0, value=2.0)

# --- Kolom 3: Kondisi Sekarang & Relasi ---
with col_input3:
    st.subheader("Kondisi Sekarang & Relasi")
    user_input['Now-Environment'] = st.selectbox("Now Environment", options=encoders['Now-Environment'].classes_)
    user_input['Now-HomeworkStress'] = st.slider("Now Homework Stress (1-6)", 1, 6, 3)
    user_input['Now-HomeworkHours'] = st.number_input("Now Homework Hours", min_value=0.0, value=4.0)
    user_input['FamilyRelationships'] = st.slider("Family Relationships (-3 s/d 3)", -3, 3, 0)
    user_input['FriendRelationships'] = st.slider("Friend Relationships (-3 s/d 3)", -3, 3, 0)

# Tombol Prediksi
if st.button("🔍 Prediksi Tingkat Stres"):
    # 1. Buat DataFrame dari input
    input_df = pd.DataFrame([user_input])

    # 2. Lakukan Encoding pada data input (menggunakan encoder yang sudah disimpan tadi)
    try:
        for col in categorical_cols:
            input_df[col] = encoders[col].transform(input_df[col])
        
        # 3. Urutkan kolom agar sesuai dengan X_train
        input_df = input_df[X.columns]
        
        # 4. Prediksi
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        # 5. Tampilkan Hasil
        st.markdown("---")
        st.subheader("Hasil Prediksi:")
        
        if prediction == 1:
            st.error("⚠️ **HIGH STRESS DETECTED**")
            st.write("Model memprediksi siswa ini mengalami tingkat stres yang tinggi pada tugas kelas.")
            st.write(f"Confidence (Keyakinan Model): {probability[1]:.2%}")
        else:
            st.success("✅ **LOW STRESS**")
            st.write("Model memprediksi siswa ini dalam kondisi stres yang wajar/rendah.")
            st.write(f"Confidence (Keyakinan Model): {probability[0]:.2%}")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data: {e}")

st.caption("Prediksi berdasarkan pola data historis siswa.")
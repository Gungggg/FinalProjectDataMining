import streamlit as st
import pandas as pd
import numpy as np #noqa
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Analisis Data Siswa", layout="wide")

st.title("Dashboard Analisis & Prediksi Stres Siswa")
st.markdown("""
Aplikasi ini menampilkan alur pemrosesan data setahap demi setahap:
1. **Raw Data**: Melihat data asli.
2. **Preprocessing**: Pembersihan dan transformasi data.
3. **Clustering**: Pengelompokan siswa berdasarkan pola jawaban.
4. **Logistic Regression**: Evaluasi model prediksi.
5. **Simulasi Prediksi**: Coba prediksi data baru (Real-time).
""")

# ==========================================
# 1. LOAD RAW DATA
# ==========================================
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

with st.expander("Tampilkan Data Mentah"):
    st.dataframe(df.head())

# ==========================================
# 2. DATA CLEANING & PREPROCESSING
# ==========================================
st.header("2. Data Cleaning & Preprocessing")

# 2.1 Handling Missing Values
df_clean = df.copy()
# Drop kolom State sesuai rencana
if 'State' in df_clean.columns:
    df_clean = df_clean.drop(columns=['State'])

# Isi missing value Gender
gender_mode = df_clean['Gender'].mode()[0]
df_clean['Gender'] = df_clean['Gender'].fillna(gender_mode)

st.write("✅ Missing Values handled & Column 'State' dropped.")

# 2.2 Encoding
st.subheader("2.2 Encoding (Text to Number)")

# Dictionary untuk menyimpan encoder agar bisa dipakai di prediksi nanti
encoders = {} 
categorical_cols = ['Category', 'Country', 'Gender', 'Before-Environment', 'Now-Environment']

# Kita buat dataframe baru khusus angka (df_encoded)
df_encoded = df_clean.copy()

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    encoders[col] = le # Simpan encoder

st.write("Contoh data setelah Encoding:")
st.dataframe(df_encoded.head(3))

# [TAMBAHAN] Tombol untuk menyimpan/download data bersih
csv_download = df_encoded.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Data Hasil Cleaning (CSV)",
    data=csv_download,
    file_name="data_cleaned_encoded.csv",
    mime="text/csv",
)

# 2.3 Scaling (Khusus untuk Clustering)
scaler = StandardScaler()
numerical_cols = ['Age', 'Before-ClassworkStress', 'Before-HomeworkStress', 'Before-HomeworkHours',
                  'Now-ClassworkStress', 'Now-HomeworkStress', 'Now-HomeworkHours',
                  'FamilyRelationships', 'FriendRelationships']

# Kita scale hanya kolom numerik
df_scaled = df_encoded.copy()
df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])

# ==========================================
# 3. CLUSTERING (K-MEANS)
# ==========================================
st.header("3. Clustering (K-Means)")
st.caption("Eksperimen dengan jumlah cluster untuk melihat pola pengelompokan siswa.")

col_k, col_viz = st.columns([1, 3])

with col_k:
    k = st.slider("Pilih Jumlah Cluster (K):", 2, 5, 3)
    
    # Menjalankan K-Means
    # Kita drop target prediksi dari data clustering agar fair (opsional, tapi disarankan)
    X_cluster = df_scaled.drop(columns=['Now-ClassworkStress']) if 'Now-ClassworkStress' in df_scaled.columns else df_scaled
    
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_cluster)
    
    # Tambahkan label ke dataframe hasil
    df_result_cluster = df.copy()
    df_result_cluster['Cluster'] = clusters

with col_viz:
    # PCA untuk visualisasi 2D
    pca = PCA(n_components=2)
    pca_comp = pca.fit_transform(X_cluster)
    df_result_cluster['PCA1'] = pca_comp[:, 0]
    df_result_cluster['PCA2'] = pca_comp[:, 1]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=df_result_cluster, x='PCA1', y='PCA2', hue='Cluster', palette='viridis', s=100, ax=ax)
    plt.title(f"Visualisasi {k} Cluster Siswa")
    st.pyplot(fig)

st.write("**Statistik Rata-rata per Cluster:**")
st.dataframe(df_result_cluster.groupby('Cluster')[['Age', 'Now-ClassworkStress', 'FamilyRelationships']].mean())

# ==========================================
# 4. LOGISTIC REGRESSION (TRAINING)
# ==========================================
st.header("4. Logistic Regression Evaluation")

# 4.1 Definisi Target & Fitur
# Target: Apakah Now-ClassworkStress >= 4? (1 = Ya, 0 = Tidak)
y = df_encoded['Now-ClassworkStress'].apply(lambda x: 1 if x >= 4 else 0)

# Fitur: Semua kolom kecuali target
X = df_encoded.drop(columns=['Now-ClassworkStress'])

# 4.2 Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4.3 Training
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# 4.4 Evaluasi
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.success(f"Model berhasil dilatih! **Akurasi: {acc:.2%}**")

with st.expander("Lihat Classification Report"):
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

# ==========================================
# 5. SIMULASI PREDIKSI
# ==========================================
st.markdown("---")
st.header("5. Simulasi Prediksi Stres Siswa")
st.info("Masukkan profil siswa di bawah ini untuk memprediksi potensi Stres Tinggi pada tugas kelas.")

# Layout Input Form
col1, col2, col3 = st.columns(3)
user_input = {}

with col1:
    st.subheader("Data Diri")
    # Mengambil opsi dari encoder classes
    cat_opt = encoders['Category'].classes_
    count_opt = encoders['Country'].classes_
    gend_opt = encoders['Gender'].classes_
    
    user_input['Category'] = st.selectbox("Category", cat_opt)
    user_input['Country'] = st.selectbox("Country", count_opt)
    user_input['Gender'] = st.selectbox("Gender", gend_opt)
    user_input['Age'] = st.number_input("Age", 10, 30, 20)

with col2:
    st.subheader("Lingkungan Awal")
    env_bef_opt = encoders['Before-Environment'].classes_
    user_input['Before-Environment'] = st.selectbox("Before Env.", env_bef_opt)
    user_input['Before-ClassworkStress'] = st.slider("Before Class Stress", 1, 6, 3)
    user_input['Before-HomeworkStress'] = st.slider("Before HW Stress", 1, 6, 3)
    user_input['Before-HomeworkHours'] = st.number_input("Before HW Hours", 0.0, 20.0, 2.0)

with col3:
    st.subheader("Lingkungan Sekarang")
    env_now_opt = encoders['Now-Environment'].classes_
    user_input['Now-Environment'] = st.selectbox("Now Env.", env_now_opt)
    # Note: Now-ClassworkStress TIDAK diinput karena itu yang mau diprediksi
    user_input['Now-HomeworkStress'] = st.slider("Now HW Stress", 1, 6, 3)
    user_input['Now-HomeworkHours'] = st.number_input("Now HW Hours", 0.0, 20.0, 4.0)
    user_input['FamilyRelationships'] = st.slider("Family Rel. (-3 to 3)", -3, 3, 0)
    user_input['FriendRelationships'] = st.slider("Friend Rel. (-3 to 3)", -3, 3, 0)

# Tombol Eksekusi
if st.button("🔍 Analisis Risiko Stres"):
    # 1. Konversi dictionary ke DataFrame
    input_df = pd.DataFrame([user_input])
    
    # 2. Encoding Data Input (Penting!)
    try:
        for col, le in encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col])
        
        # 3. Pastikan urutan kolom SAMA PERSIS dengan X (Training data)
        # Ini mencegah error jika urutan kolom tertukar
        input_df = input_df[X.columns]
        
        # 4. Prediksi
        pred_label = model.predict(input_df)[0]
        pred_prob = model.predict_proba(input_df)[0]
        
        st.markdown("---")
        if pred_label == 1:
            st.error("⚠️ **HIGH STRESS RISK DETECTED**")
            st.write("Siswa ini diprediksi memiliki tingkat stres tinggi (Skor >= 4).")
            st.progress(int(pred_prob[1]*100))
            st.caption(f"Confidence Level: {pred_prob[1]:.2%}")
        else:
            st.success("✅ **LOW STRESS RISK**")
            st.write("Siswa ini diprediksi dalam kondisi stabil.")
            st.progress(int(pred_prob[0]*100))
            st.caption(f"Confidence Level: {pred_prob[0]:.2%}")
            
    except Exception as e:
        st.error(f"Terjadi error pada pemrosesan data: {e}")
        st.warning("Tips: Pastikan semua opsi input dipilih dengan benar.")
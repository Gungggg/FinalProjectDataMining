import streamlit as st
import pandas as pd
import numpy as np # noqa
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Analisis & Komparasi Model", layout="wide")

st.title("Dashboard Analisis: Clustering & Komparasi Model")
st.markdown("""
**Tujuan Dashboard:**
1.  **Clustering**: Mengelompokkan siswa berdasarkan karakteristik.
2.  **Klasifikasi (Tugas Utama)**: Menggunakan **Logistic Regression**.
3.  **Eksperimen (Argumentasi)**: Membandingkan dengan **Random Forest** + **Confusion Matrix** untuk evaluasi mendalam.
""")

# ==========================================
# 1. LOAD DATA
# ==========================================
st.header("1. Raw Data")
uploaded_file = 'responses.csv' 

@st.cache_data
def load_data():
    try:
        return pd.read_csv(uploaded_file)
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error("File 'responses.csv' tidak ditemukan!")
    st.stop()

with st.expander("Lihat Data Mentah"):
    st.dataframe(df.head())

# ==========================================
# 2. PREPROCESSING & FEATURE ENGINEERING
# ==========================================
st.header("2. Preprocessing & Feature Engineering")

df_clean = df.copy()

# 2.1 Handling Missing Values & Drop
if 'State' in df_clean.columns:
    df_clean = df_clean.drop(columns=['State'])

gender_mode = df_clean['Gender'].mode()[0]
df_clean['Gender'] = df_clean['Gender'].fillna(gender_mode)

# 2.2 Feature Engineering (Membuat Fitur Baru)
df_clean['Total_Support_System'] = df_clean['FamilyRelationships'] + df_clean['FriendRelationships']
df_clean['Total_Homework_Stress'] = df_clean['Before-HomeworkStress'] + df_clean['Now-HomeworkStress']

st.info("✅ Feature Engineering: Menambahkan kolom 'Total_Support_System' dan 'Total_Homework_Stress'.")

# 2.3 Encoding
encoders = {} 
categorical_cols = ['Category', 'Country', 'Gender', 'Before-Environment', 'Now-Environment']

df_encoded = df_clean.copy()

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    encoders[col] = le

st.write("Data siap digunakan (Encoded).")

# ==========================================
# 3. CLUSTERING (K-MEANS)
# ==========================================
st.header("3. Clustering (K-Means)")
st.caption("Pengelompokan siswa tanpa label (Unsupervised Learning).")

# Scaling khusus untuk clustering
scaler = StandardScaler()
num_cols_cluster = ['Age', 'Total_Support_System', 'Total_Homework_Stress', 'Now-HomeworkHours', 'Before-HomeworkHours']
X_cluster_scaled = scaler.fit_transform(df_encoded[num_cols_cluster])

col_k, col_viz = st.columns([1, 3])
with col_k:
    k = st.slider("Jumlah Cluster (K)", 2, 5, 3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_cluster_scaled)

with col_viz:
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(X_cluster_scaled)
    df_viz = df.copy()
    df_viz['Cluster'] = clusters
    df_viz['PCA1'] = pca_res[:, 0]
    df_viz['PCA2'] = pca_res[:, 1]

    fig_clust, ax_clust = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=df_viz, x='PCA1', y='PCA2', hue='Cluster', palette='viridis', s=100, ax=ax_clust)
    st.pyplot(fig_clust)

# ==========================================
# 4. MODELING & COMPARISON (INTI TUGAS)
# ==========================================
st.header("4. Klasifikasi & Komparasi Model")

# Fungsi Helper untuk Plot Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title(f"Confusion Matrix: {model_name}")
    ax.set_xlabel("Prediksi Model")
    ax.set_ylabel("Data Asli (Aktual)")
    ax.set_xticklabels(['Rendah', 'Tinggi'])
    ax.set_yticklabels(['Rendah', 'Tinggi'])
    return fig

# Persiapan Data
y = df_encoded['Now-ClassworkStress'].apply(lambda x: 1 if x >= 4 else 0)
X = df_encoded.drop(columns=['Now-ClassworkStress'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- MODEL 1: LOGISTIC REGRESSION ---
logreg = LogisticRegression(max_iter=2000, random_state=42)
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)

# --- MODEL 2: RANDOM FOREST ---
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

# TAMPILKAN PERBANDINGAN
col_model1, col_model2 = st.columns(2)

with col_model1:
    st.subheader("🅰️ Logistic Regression")
    st.metric(label="Akurasi", value=f"{acc_lr:.2%}")
    st.warning("Model Linear (Sederhana)")
    
    # Plot CM LogReg
    fig_cm_lr = plot_confusion_matrix(y_test, y_pred_lr, "Logistic Regression")
    st.pyplot(fig_cm_lr)
    
    with st.expander("Laporan Klasifikasi (LogReg)"):
        st.text(classification_report(y_test, y_pred_lr))

with col_model2:
    st.subheader("🅱️ Random Forest")
    st.metric(label="Akurasi", value=f"{acc_rf:.2%}", delta=f"{(acc_rf - acc_lr):.2%} vs LogReg")
    st.success("Model Non-Linear (Kompleks)")
    
    # Plot CM RF
    fig_cm_rf = plot_confusion_matrix(y_test, y_pred_rf, "Random Forest")
    st.pyplot(fig_cm_rf)
    
    with st.expander("Laporan Klasifikasi (Random Forest)"):
        st.text(classification_report(y_test, y_pred_rf))

st.caption("""
**Cara Membaca Confusion Matrix:**
* **Kotak Kiri-Atas (True Negative):** Asli Tidak Stres, Prediksi Tidak Stres (Benar ✅).
* **Kotak Kanan-Bawah (True Positive):** Asli Stres, Prediksi Stres (Benar ✅).
* **Kotak Kanan-Atas (False Positive):** Asli Tidak Stres, tapi diprediksi Stres (Salah ❌).
* **Kotak Kiri-Bawah (False Negative):** Asli Stres, tapi diprediksi Tidak Stres (Salah ❌ - Berbahaya).
""")

# ==========================================
# 5. SIMULASI PREDIKSI
# ==========================================
st.markdown("---")
st.header("5. Simulasi Prediksi")
st.write("Pilih model mana yang ingin digunakan untuk memprediksi data baru.")

# Pilihan Model
model_choice = st.radio("Pilih Model:", ("Logistic Regression (Standard)", "Random Forest (Optimized)"))
selected_model = logreg if model_choice == "Logistic Regression (Standard)" else rf

# Form Input
col1, col2, col3 = st.columns(3)
inputs = {}

with col1:
    st.subheader("Profil")
    inputs['Category'] = st.selectbox("Category", encoders['Category'].classes_)
    inputs['Country'] = st.selectbox("Country", encoders['Country'].classes_)
    inputs['Gender'] = st.selectbox("Gender", encoders['Gender'].classes_)
    inputs['Age'] = st.number_input("Age", 10, 30, 20)

with col2:
    st.subheader("Masa Lalu")
    inputs['Before-Environment'] = st.selectbox("Before Env.", encoders['Before-Environment'].classes_)
    inputs['Before-ClassworkStress'] = st.slider("Before Class Stress", 1, 6, 3)
    # Simpan ke variabel agar bisa dipakai hitung engineering
    bf_hw_stress = st.slider("Before HW Stress", 1, 6, 3)
    inputs['Before-HomeworkStress'] = bf_hw_stress
    inputs['Before-HomeworkHours'] = st.number_input("Before HW Hours", 0.0, 20.0, 2.0)

with col3:
    st.subheader("Sekarang")
    inputs['Now-Environment'] = st.selectbox("Now Env.", encoders['Now-Environment'].classes_)
    
    now_hw_stress = st.slider("Now HW Stress", 1, 6, 3)
    inputs['Now-HomeworkStress'] = now_hw_stress
    
    inputs['Now-HomeworkHours'] = st.number_input("Now HW Hours", 0.0, 20.0, 4.0)
    
    fam_rel = st.slider("Family Rel.", -3, 3, 0)
    inputs['FamilyRelationships'] = fam_rel
    
    friend_rel = st.slider("Friend Rel.", -3, 3, 0)
    inputs['FriendRelationships'] = friend_rel

if st.button("🚀 Prediksi Risiko Stres"):
    # 1. Buat DataFrame
    input_df = pd.DataFrame([inputs])
    
    # 2. FEATURE ENGINEERING (PENTING: Harus sama persis dengan training)
    input_df['Total_Support_System'] = input_df['FamilyRelationships'] + input_df['FriendRelationships']
    input_df['Total_Homework_Stress'] = input_df['Before-HomeworkStress'] + input_df['Now-HomeworkStress']
    
    # 3. Encoding
    try:
        for col, le in encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col].astype(str))
        
        # 4. Reorder Columns
        input_df = input_df[X.columns]
        
        # 5. Prediksi
        pred = selected_model.predict(input_df)[0]
        prob = selected_model.predict_proba(input_df)[0]
        
        st.markdown("---")
        st.write(f"**Menggunakan Model: {model_choice}**")
        
        if pred == 1:
            st.error(f"⚠️ **Stres TINGGI** (Confidence: {prob[1]:.2%})")
        else:
            st.success(f"✅ **Stres RENDAH** (Confidence: {prob[0]:.2%})")
            
    except Exception as e:
        st.error(f"Error: {e}")
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, silhouette_score

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Analisis Data Pipeline", layout="wide")

st.title("🔬 Dashboard Analisis Data & Pipeline Machine Learning")
st.markdown("""
Aplikasi ini dirancang untuk menvisualisasikan seluruh proses Data Mining:
mulai dari **Data Mentah**, **Pembersihan**, **Rekayasa Fitur**, hingga **Evaluasi Model**.
""")

# Membuat Tabulasi agar presentasi lebih rapi
tab1, tab2, tab3, tab4 = st.tabs([
    " 1. Data Pipeline (Proses)", 
    " 2. Clustering (K-Means)", 
    " 3. Komparasi Model", 
    " 4. Simulasi Prediksi"
])

# ==========================================
# LOAD DATA (GLOBAL)
# ==========================================
@st.cache_data
def load_data():
    try:
        return pd.read_csv('responses.csv')
    except FileNotFoundError:
        return None

df_raw = load_data()

if df_raw is None:
    st.error("File 'responses.csv' tidak ditemukan!")
    st.stop()

# ==========================================
# TAB 1: DATA PIPELINE (DETAIL PROSES)
# ==========================================
with tab1:
    st.header("Alur Pemrosesan Data (Data Pipeline)")
    
    # --- STEP 1: RAW DATA ---
    st.subheader("Langkah 1: Raw Data (Data Mentah)")
    st.write("Data asli yang dibaca langsung dari file CSV.")
    st.dataframe(df_raw.head(3))
    st.caption(f"Dimensi Data Awal: {df_raw.shape[0]} baris, {df_raw.shape[1]} kolom.")
    
    st.markdown("---")

    # --- STEP 2: DATA CLEANING ---
    st.subheader("Langkah 2: Data Cleaning (Pembersihan)")
    col_clean1, col_clean2 = st.columns(2)
    
    with col_clean1:
        st.markdown("**Masalah pada Raw Data:**")
        st.write("- Kolom `State` memiliki terlalu banyak variasi (kardinalitas tinggi) dan tidak relevan.")
        st.write("- Kolom `Gender` memiliki nilai kosong (NaN).")
        
    # Proses Cleaning
    df_clean = df_raw.copy()
    
    # 1. Drop State
    if 'State' in df_clean.columns:
        df_clean = df_clean.drop(columns=['State'])
    
    # 2. Imputasi Missing Value (Gender)
    gender_mode = df_clean['Gender'].mode()[0]
    df_clean['Gender'] = df_clean['Gender'].fillna(gender_mode)
    
    with col_clean2:
        st.markdown("**Tindakan Perbaikan:**")
        st.success("✅ Menghapus kolom `State`.")
        st.success(f"✅ Mengisi data kosong di `Gender` dengan modus: **'{gender_mode}'**.")
    
    with st.expander("Lihat Data Hasil Cleaning"):
        st.dataframe(df_clean.head(3))

    st.markdown("---")

    # --- STEP 3: FEATURE ENGINEERING ---
    st.subheader("Langkah 3: Feature Engineering (Rekayasa Fitur)")
    st.markdown("""
    Disini kita menciptakan kolom baru untuk menangkap pola yang lebih kuat (karena fitur tunggal korelasinya lemah).
    """)
    
    # Proses Feature Engineering
    # Gabungan skor hubungan sosial
    df_clean['Total_Support_System'] = df_clean['FamilyRelationships'] + df_clean['FriendRelationships']
    # Gabungan skor stres PR
    df_clean['Total_Homework_Stress'] = df_clean['Before-HomeworkStress'] + df_clean['Now-HomeworkStress']
    
    st.code("""
    # Rumus Fitur Baru:
    df['Total_Support_System'] = df['FamilyRelationships'] + df['FriendRelationships']
    df['Total_Homework_Stress'] = df['Before-HomeworkStress'] + df['Now-HomeworkStress']
    """, language='python')
    
    st.write("Contoh hasil penambahan kolom baru:")
    st.dataframe(df_clean[['FamilyRelationships', 'FriendRelationships', 'Total_Support_System', 'Total_Homework_Stress']].head(3))

    st.markdown("---")

    # --- STEP 4: ENCODING ---
    st.subheader("Langkah 4: Encoding (Penerjemahan ke Angka)")
    st.write("Mengubah data teks (Kategori) menjadi angka agar bisa dihitung oleh mesin.")
    
    encoders = {} 
    categorical_cols = ['Category', 'Country', 'Gender', 'Before-Environment', 'Now-Environment']
    df_encoded = df_clean.copy()

    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le
        
    c1, c2 = st.columns(2)
    with c1:
        st.info("Sebelum Encoding (Teks)")
        st.write(df_clean[categorical_cols].head(3))
    with c2:
        st.success("Sesudah Encoding (Angka)")
        st.write(df_encoded[categorical_cols].head(3))
    
    st.write("✅ **Data Pipeline Selesai!** Data ini siap masuk ke tahap Modeling.")

# ==========================================
# PREPARATION FOR MODELING (BACKEND)
# ==========================================
# Scaling khusus untuk Clustering (Karena K-Means sensitif terhadap jarak)
scaler = StandardScaler()
# Kita pilih fitur numerik + fitur hasil engineering tadi
num_cols_cluster = ['Age', 'Total_Support_System', 'Total_Homework_Stress', 'Now-HomeworkHours', 'Before-HomeworkHours']
X_cluster_scaled = scaler.fit_transform(df_encoded[num_cols_cluster])

# Split Data untuk Klasifikasi
y = df_encoded['Now-ClassworkStress'].apply(lambda x: 1 if x >= 4 else 0)
X = df_encoded.drop(columns=['Now-ClassworkStress'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# TAB 2: CLUSTERING
# ==========================================
with tab2:
    st.header("Analisis Clustering (Unsupervised)")
    st.write("Mengelompokkan siswa berdasarkan kemiripan karakteristik.")
    
    col_k, col_res = st.columns([1, 3])
    
    with col_k:
        k_val = st.slider("Pilih Jumlah Cluster (K)", 2, 6, 3)
        
        # MODEL K-MEANS
        kmeans = KMeans(n_clusters=k_val, random_state=42)
        clusters = kmeans.fit_predict(X_cluster_scaled)
        
        # METRIC: SILHOUETTE SCORE
        sil_score = silhouette_score(X_cluster_scaled, clusters)
        st.metric("Silhouette Score", f"{sil_score:.3f}")
        
        if sil_score > 0.5:
            st.success("Structure: STRONG")
        elif sil_score > 0.25:
            st.warning("Structure: MEDIUM")
        else:
            st.error("Structure: WEAK")
            
    with col_res:
        # Visualisasi PCA
        pca = PCA(n_components=2)
        pca_res = pca.fit_transform(X_cluster_scaled)
        
        df_viz = df_encoded.copy()
        df_viz['Cluster'] = clusters
        df_viz['PCA1'] = pca_res[:, 0]
        df_viz['PCA2'] = pca_res[:, 1]
        
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.scatterplot(data=df_viz, x='PCA1', y='PCA2', hue='Cluster', palette='viridis', s=80, ax=ax)
        plt.title(f"Visualisasi Sebaran Data ({k_val} Cluster)")
        st.pyplot(fig)
        
    st.subheader("Profil Rata-rata per Cluster")
    st.dataframe(df_viz.groupby('Cluster')[num_cols_cluster].mean())

# ==========================================
# TAB 3: KOMPARASI MODEL
# ==========================================
with tab3:
    st.header("Evaluasi & Komparasi Model Klasifikasi")
    st.write("Target Prediksi: **Apakah Siswa Mengalami Stres Tinggi pada Tugas Kelas?**")
    
    # --- TRAINING MODELS ---
    # 1. Logistic Regression
    logreg = LogisticRegression(max_iter=2000, random_state=42)
    logreg.fit(X_train, y_train)
    y_pred_lr = logreg.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    
    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    
    # --- VISUALISASI ---
    col_mod1, col_mod2 = st.columns(2)
    
    # Helper function plot CM
    def plot_cm(y_true, y_pred, title):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(3, 2.5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Prediksi')
        ax.set_ylabel('Aktual')
        return fig

    with col_mod1:
        st.subheader("A. Logistic Regression")
        st.metric("Akurasi", f"{acc_lr:.2%}")
        st.pyplot(plot_cm(y_test, y_pred_lr, "Confusion Matrix (LogReg)"))
        with st.expander("Lihat Detail Report"):
            st.text(classification_report(y_test, y_pred_lr))
            
    with col_mod2:
        st.subheader("B. Random Forest")
        diff = acc_rf - acc_lr
        st.metric("Akurasi", f"{acc_rf:.2%}", delta=f"{diff:.2%}")
        st.pyplot(plot_cm(y_test, y_pred_rf, "Confusion Matrix (Random Forest)"))
        with st.expander("Lihat Detail Report"):
            st.text(classification_report(y_test, y_pred_rf))
            
    st.info("""
    **Insight:** Perbedaan akurasi terjadi karena Random Forest mampu menangkap hubungan non-linear antar fitur yang gagal ditangkap oleh Logistic Regression.
    """)

# ==========================================
# TAB 4: PREDIKSI
# ==========================================
with tab4:
    st.header("Simulasi Prediksi Real-time")
    
    # Input Layout
    c1, c2, c3 = st.columns(3)
    inputs = {}
    
    with c1:
        st.markdown("### 1. Profil")
        inputs['Category'] = st.selectbox("Category", encoders['Category'].classes_)
        inputs['Country'] = st.selectbox("Country", encoders['Country'].classes_)
        inputs['Gender'] = st.selectbox("Gender", encoders['Gender'].classes_)
        inputs['Age'] = st.number_input("Age", 10, 30, 20)

    with c2:
        st.markdown("### 2. Masa Lalu")
        inputs['Before-Environment'] = st.selectbox("Before Env.", encoders['Before-Environment'].classes_)
        inputs['Before-ClassworkStress'] = st.slider("Before Class Stress", 1, 6, 3)
        # Variable temp untuk engineering
        bf_hw_s = st.slider("Before HW Stress", 1, 6, 3)
        inputs['Before-HomeworkStress'] = bf_hw_s
        inputs['Before-HomeworkHours'] = st.number_input("Before HW Hours", 0.0, 20.0, 2.0)

    with c3:
        st.markdown("### 3. Sekarang")
        inputs['Now-Environment'] = st.selectbox("Now Env.", encoders['Now-Environment'].classes_)
        
        now_hw_s = st.slider("Now HW Stress", 1, 6, 3)
        inputs['Now-HomeworkStress'] = now_hw_s
        inputs['Now-HomeworkHours'] = st.number_input("Now HW Hours", 0.0, 20.0, 4.0)
        
        fam_rel = st.slider("Family Rel.", -3, 3, 0)
        inputs['FamilyRelationships'] = fam_rel
        friend_rel = st.slider("Friend Rel.", -3, 3, 0)
        inputs['FriendRelationships'] = friend_rel
        
    # Pilihan Model
    model_choice = st.radio("Pilih Model untuk Prediksi:", ["Logistic Regression", "Random Forest"])
    active_model = logreg if model_choice == "Logistic Regression" else rf

    if st.button("🔍 Analisis Risiko"):
        # 1. Buat DataFrame
        input_df = pd.DataFrame([inputs])
        
        # 2. FEATURE ENGINEERING (Wajib sama dengan Tab 1)
        input_df['Total_Support_System'] = input_df['FamilyRelationships'] + input_df['FriendRelationships']
        input_df['Total_Homework_Stress'] = input_df['Before-HomeworkStress'] + input_df['Now-HomeworkStress']
        
        # 3. Encoding
        for col, le in encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col].astype(str))
                
        # 4. Reorder Columns
        input_df = input_df[X.columns]
        
        # 5. Prediksi
        pred = active_model.predict(input_df)[0]
        prob = active_model.predict_proba(input_df)[0]
        
        st.divider()
        if pred == 1:
            st.error("⚠️ **RISIKO TINGGI (High Stress)**")
            st.write(f"Model ({model_choice}) yakin {prob[1]:.1%} bahwa siswa ini mengalami tekanan tinggi.")
        else:
            st.success("✅ **RISIKO RENDAH (Low Stress)**")
            st.write(f"Model ({model_choice}) memprediksi kondisi siswa aman (Confidence: {prob[0]:.1%}).")
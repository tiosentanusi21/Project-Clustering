import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


def show():
    st.title("📈 Visualisasi Klastering (KMeans - 3 Klaster Sesuai Analisis)")

    if "data" not in st.session_state:
        st.warning("⚠️ Harap unggah data terlebih dahulu di menu 'Tabel'.")
        return

    df = st.session_state["data"].copy()

    # Tetapkan urutan fitur eksplisit agar konsisten dengan model ML utama
    features = [
        "Tingkat Kunjungan dan Perilaku Wisata",
        "Persepsi terhadap Potensi Geyser Cisolok sebagai Warisan UNESCO",
        "Kontribusi Ekonomi dan Pendapatan Daerah",
        "Strategi Pengembangan dan Optimalisasi Potensi"
    ]

    if not all(f in df.columns for f in features):
        st.error("Pastikan dataset memiliki kolom: " + ", ".join(features))
        return

    # Normalisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    # Klastering dengan 3 klaster sesuai analisis
    kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans_final.fit_predict(X_scaled)

    # Rata-rata tiap variabel per klaster
    summary = df.groupby('Cluster')[features].mean().round(2)
    st.write("### 📊 Rata-rata Tiap Variabel per Cluster")
    st.dataframe(summary)

    # Jumlah responden per cluster
    jumlah_responden = df['Cluster'].value_counts().sort_index()
    st.write("### 👥 Jumlah Responden per Cluster")
    for cluster, count in jumlah_responden.items():
        st.markdown(f"- **Cluster {cluster}**: {count} responden")

    # Histogram rata-rata per variabel
    st.write("### 📌 Histogram Rata-rata Tiap Variabel")
    for col in summary.columns:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(summary.index.astype(str), summary[col], color='cornflowerblue')
        ax.set_title(f'Rata-rata {col} per Cluster')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Rata-rata Skor')
        ax.set_ylim(0, summary.max().max() + 1)
        for i, val in enumerate(summary[col]):
            ax.text(i, val + 0.1, f"{val:.2f}", ha='center')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig)

    # Visualisasi PCA (2D)
    st.write("### 🧭 Visualisasi Klaster (PCA 2D)")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Cluster'], palette="Set2", s=60)
    ax.set_title("Visualisasi Klaster (PCA 2D)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.grid(True)
    st.pyplot(fig)

    # Elbow Method
    st.write("### 📐 Elbow Method untuk Menentukan Jumlah Cluster Optimal")
    wcss = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        wcss.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, 11), wcss, marker='o')
    ax.set_title("Elbow Method")
    ax.set_xlabel("Jumlah Cluster")
    ax.set_ylabel("WCSS")
    ax.grid(True)
    st.pyplot(fig)

    # Simpan hasil ke Excel (opsional)
    df_download = df.copy()
    st.download_button("📥 Unduh Data dengan Cluster", data=df_download.to_csv(index=False), file_name="Hasil_Clustering.csv", mime="text/csv")

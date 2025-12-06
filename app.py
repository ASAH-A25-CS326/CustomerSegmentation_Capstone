import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- 1. CONFIGURATION & FUNCTIONS ---
st.set_page_config(page_title="Dashboard Segmentasi Pelanggan", layout="wide")

@st.cache_data
def load_and_clean_data(file):
    """
    Fungsi untuk membaca dan membersihkan data transaksi retail.
    """
    # Membaca file
    if file.name.endswith('csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    # Standardisasi Nama Kolom
    df = df.rename(columns={
        'Customer ID': 'Customer_Id',
        'Price': 'UnitPrice',
    })
    
    # Validasi Kolom Minimum
    required_cols = ['InvoiceDate', 'Invoice', 'StockCode', 'Description', 'Quantity', 'UnitPrice', 'Customer_Id', 'Country']
    if not all(col in df.columns for col in required_cols):
        st.error(f"File tidak memiliki kolom yang dibutuhkan: {required_cols}")
        return pd.DataFrame()

    # Konversi tipe data
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df['Invoice'] = df['Invoice'].astype(str).str.strip()
    df['Description'] = df['Description'].astype(str).str.strip().str.upper()
    df['Country'] = df['Country'].astype(str).str.strip()
    
    # Cleaning: Hapus Missing, Duplikat, Cancelled, dan Deskripsi Sampah
    df = df.dropna(subset=['Description', 'Customer_Id'])
    df = df.drop_duplicates()
    df = df[~df['Invoice'].str.startswith('C')] # Hapus Cancelled
    
    unwanted_patterns = ['ADJUST', 'POSTAGE', 'CARRIAGE', 'BANK CHARGES', 'DOTCOM', 'TEST', 'SAMPLE']
    pattern = '|'.join(unwanted_patterns)
    df = df[~df['Description'].str.contains(pattern, case=False, na=False)]
    
    # Filter Nilai Positif
    df = df[(df['UnitPrice'] > 0) & (df['Quantity'] > 0)]
    
    # Hitung Total Belanja
    df['Amount'] = df['Quantity'] * df['UnitPrice']
    
    return df

@st.cache_data
def calculate_rfm(df):
    """
    Mengubah data transaksi menjadi tabel RFM per Customer.
    """
    if df.empty:
        return pd.DataFrame()

    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('Customer_Id').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'Invoice': 'nunique',
        'Amount': 'sum'
    }).reset_index()
    
    rfm.columns = ['Customer_Id', 'Recency', 'Frequency', 'Monetary']
    return rfm

# --- 2. SIDEBAR CONFIGURATION ---
st.sidebar.title("🔧 Kontrol Dashboard")

# A. File Upload
st.sidebar.subheader("1. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload dataset (CSV/Excel)", type=['csv', 'xlsx'])
use_dummy = st.sidebar.checkbox("Gunakan Data Dummy (Jika tidak ada file)", value=True)

# B. Clustering Config
st.sidebar.subheader("2. Parameter Clustering")
k_value = st.sidebar.radio(
    "Pilih Jumlah Cluster (K):",
    options=[3, 4, 5],
    index=1, 
    help="Jumlah segmen pelanggan yang ingin dibentuk."
)

st.sidebar.markdown("---")
st.sidebar.info("Dashboard ini menggunakan alur: Log Transform -> Scaling -> PCA -> K-Means")

# --- 3. MAIN DATA PROCESSING ---
st.title("🛍️ Dashboard Customer Segmentation (RFM + PCA)")

# Load Data
if uploaded_file is not None:
    df_clean = load_and_clean_data(uploaded_file)
    if df_clean.empty: st.stop()
    rfm_data = calculate_rfm(df_clean)
    st.success(f"✅ Data Berhasil Dimuat: {len(rfm_data)} Customer unik teridentifikasi.")
elif use_dummy:
    # Generate Dummy Data
    np.random.seed(42)
    rfm_data = pd.DataFrame({
        'Customer_Id': [str(i) for i in range(1, 1001)],
        'Recency': np.random.randint(1, 365, 1000),
        'Frequency': np.random.randint(1, 50, 1000),
        'Monetary': np.random.randint(100, 10000, 1000)
    })
    st.warning("⚠️ Menggunakan Data Dummy untuk Demo.")
else:
    st.info("👋 Silakan upload file dataset di sidebar kiri untuk memulai.")
    st.stop()

# --- 4. MODELING PIPELINE ---

# A. Log Transformation (Mengatasi Skewness)
rfm_log = rfm_data[['Recency', 'Frequency', 'Monetary']].apply(np.log1p)

# B. Standardization (Z-Score)
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)
rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=['Recency', 'Frequency', 'Monetary'])

# C. PCA (Reduksi Dimensi untuk Visualisasi 2D)
pca = PCA(n_components=2)
pca_components = pca.fit_transform(rfm_scaled)

# D. K-Means Clustering
kmeans = KMeans(n_clusters=k_value, random_state=42, n_init='auto')
clusters = kmeans.fit_predict(rfm_scaled)

# E. Gabungkan Hasil ke DataFrame Utama
rfm_final = rfm_data.copy()
rfm_final['Cluster'] = clusters.astype(str)
rfm_final['PC1'] = pca_components[:, 0]
rfm_final['PC2'] = pca_components[:, 1]

# Tambahkan Cluster ke data scaled untuk Snake Plot
rfm_scaled_df['Cluster'] = clusters.astype(str)

# --- 5. VISUALIZATION TABS ---
# Membuat 4 Tab agar semua visualisasi tertampung rapi
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Visualisasi 2D (PCA)", 
    "🧊 Visualisasi 3D (RFM)", 
    "📝 Interpretasi Segmen", 
    "🐍 Snake Plot"
])

# --- TAB 1: PCA 2D ---
with tab1:
    st.subheader(f"Peta Persebaran Segmen (PCA 2D)")
    st.markdown("Grafik ini menyederhanakan data RFM menjadi 2 dimensi. Titik yang berdekatan memiliki perilaku belanja yang mirip.")
    
    fig_pca = px.scatter(
        rfm_final,
        x="PC1",
        y="PC2",
        color="Cluster",
        hover_data=['Recency', 'Frequency', 'Monetary'],
        title=f"Cluster Separation (K={k_value})",
        color_discrete_sequence=px.colors.qualitative.Bold,
        opacity=0.8
    )
    st.plotly_chart(fig_pca, use_container_width=True)

# --- TAB 2: 3D RFM ---
with tab2:
    st.subheader("Visualisasi 3D Interaktif")
    
    use_log = st.checkbox("Gunakan Skala Logaritmik?", value=True, help="Disarankan ON agar data tidak menumpuk di satu titik.")
    
    if use_log:
        # Buat kolom log sementara untuk plotting
        plot_data = rfm_final.copy()
        for col in ['Recency', 'Frequency', 'Monetary']:
            plot_data[f'Log_{col}'] = np.log1p(plot_data[col])
        x_col, y_col, z_col = 'Log_Recency', 'Log_Frequency', 'Log_Monetary'
    else:
        plot_data = rfm_final
        x_col, y_col, z_col = 'Recency', 'Frequency', 'Monetary'

    fig_3d = px.scatter_3d(
        plot_data,
        x=x_col, y=y_col, z=z_col,
        color='Cluster',
        hover_data={'Recency':True, 'Frequency':True, 'Monetary':True, 
                    'Log_Recency':False, 'Log_Frequency':False, 'Log_Monetary':False, 
                    'Cluster':False, 'Customer_Id':True},
        title="3D RFM Distribution",
        color_discrete_sequence=px.colors.qualitative.Bold,
        opacity=0.6
    )
    fig_3d.update_layout(height=600)
    st.plotly_chart(fig_3d, use_container_width=True)

# --- TAB 3: INTERPRETASI ---
with tab3:
    st.subheader("Profil Statistik Segmen")
    
    # Hitung Statistik Rata-rata
    summary = rfm_final.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().reset_index().round(1)
    counts = rfm_final['Cluster'].value_counts().reset_index()
    counts.columns = ['Cluster', 'Count']
    summary = pd.merge(summary, counts, on='Cluster').sort_values('Monetary', ascending=False)
    
    # Rata-rata Global untuk Perbandingan
    avg_r = rfm_final['Recency'].mean()
    avg_f = rfm_final['Frequency'].mean()
    avg_m = rfm_final['Monetary'].mean()

    # Tampilkan Kartu Insight
    cols = st.columns(k_value)
    for i, (_, row) in enumerate(summary.iterrows()):
        # Logika Label Sederhana
        if row['Monetary'] > avg_m and row['Frequency'] > avg_f and row['Recency'] < avg_r:
            label, color = "💎 CHAMPION", "green"
        elif row['Recency'] > avg_r and row['Monetary'] > avg_m:
            label, color = "⚠️ AT RISK", "orange"
        elif row['Recency'] > avg_r and row['Monetary'] < avg_m:
            label, color = "💤 LOST", "red"
        else:
            label, color = "🌱 POTENTIAL", "blue"

        with cols[i]:
            st.markdown(f"### Cluster {row['Cluster']}")
            st.markdown(f":{color}[**{label}**]")
            st.metric("Populasi", int(row['Count']))
            st.caption(f"🛒 Avg Spend: ${row['Monetary']:,.0f}")
            st.caption(f"📅 Avg Recency: {row['Recency']} hari")
            st.divider()

    st.write("### Detail Statistik")
    st.dataframe(summary.style.highlight_max(axis=0, color='lightgreen'))

# --- TAB 4: SNAKE PLOT ---
with tab4:
    st.subheader("Snake Plot (Analisis Karakteristik)")
    st.markdown("Grafik ini menggunakan data yang sudah di-standarisasi (Z-Score) untuk membandingkan pola relatif antar cluster.")
    
    # Melt Data Scaled untuk Plotting Line Chart
    df_melt = pd.melt(
        rfm_scaled_df.reset_index(), 
        id_vars=['Cluster'], 
        value_vars=['Recency', 'Frequency', 'Monetary'], 
        var_name='Metric', 
        value_name='Value'
    )
    
    fig_snake = px.line(
        df_melt, 
        x="Metric", 
        y="Value", 
        color="Cluster", 
        title="Pola Standar Deviasi per Cluster",
        markers=True,
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    st.plotly_chart(fig_snake, use_container_width=True)
    st.info("💡 **Cara Baca:** Nilai 0 adalah rata-rata populasi. Titik di atas 0 berarti 'Di atas rata-rata', di bawah 0 berarti 'Di bawah rata-rata'.")
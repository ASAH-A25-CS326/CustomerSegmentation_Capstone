import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import google.generativeai as genai
import os

# =========================================================
# 1. CONFIG & DATA LOADING
# =========================================================
st.set_page_config(page_title="Integrated Customer Segmentation & CLV", layout="wide")

@st.cache_data
def load_and_clean_data(file_path):
    """
    Membaca dan membersihkan data transaksi (Support CSV/Excel).
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    # Standardisasi Nama Kolom
    col_map = {c.lower(): c for c in df.columns}
    if 'customer id' in col_map:
        df = df.rename(columns={col_map['customer id']: 'Customer_Id'})
    elif 'customer_id' in col_map:
        df = df.rename(columns={col_map['customer_id']: 'Customer_Id'})
        
    if 'price' in col_map:
        df = df.rename(columns={col_map['price']: 'UnitPrice'})
    
    # --- FIX BUG 'nan' ---
    df = df.dropna(subset=['Customer_Id'])
    df['Customer_Id'] = df['Customer_Id'].astype(str).str.strip()
    df = df[df['Customer_Id'].str.lower() != 'nan']
    # ---------------------

    # Konversi tipe data lain
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df['Invoice'] = df['Invoice'].astype(str).str.strip()
    
    # Cleaning Lanjutan
    df = df.drop_duplicates()
    df = df[~df['Invoice'].str.startswith('C')] # Hapus Cancelled
    
    # Filter Nilai Positif
    df = df[(df['UnitPrice'] > 0) & (df['Quantity'] > 0)]
    
    # Hitung Total Belanja
    df['Amount'] = df['Quantity'] * df['UnitPrice']
    
    return df

@st.cache_data
def calculate_rfm_clv(df):
    """
    Menghitung RFM dasar dan Menambahkan Fitur CLV (Historical).
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
    
    # --- CLV Features ---
    rfm["HistoricalCLV"] = rfm["Monetary"]

    # 1. Value-based Segmentation
    q30_val, q70_val = rfm["HistoricalCLV"].quantile([0.3, 0.7]).values
    def value_segment(x):
        if x >= q70_val: return "High CLV"
        elif x <= q30_val: return "Low CLV"
        return "Medium CLV"
    rfm["CLV_ValueSegment"] = rfm["HistoricalCLV"].apply(value_segment)

    # 2. Churn Risk Segmentation
    r_q30, r_q70 = rfm["Recency"].quantile([0.3, 0.7]).values
    def churn_risk(recency):
        if recency >= r_q70: return "High Risk"
        elif recency <= r_q30: return "Low Risk"
        return "Medium Risk"
    rfm["Churn_Risk"] = rfm["Recency"].apply(churn_risk)

    return rfm

# =========================================================
# 2. PAPER KNOWLEDGE BASE (10 SEGMENTASI)
# =========================================================
PAPER_SEGMENTS_RANKED = [
    {"name": "Churned Customers", "char": "Sudah lama tidak bertransaksi (RFM 111).", "strategy": "Win-Back Campaign eksklusif."},
    {"name": "Hibernating", "char": "Pola pembelian rendah & kurang terlibat.", "strategy": "Tawarkan produk relevan & diskon re-branding."},
    {"name": "About To Sleep", "char": "Recency menurun, berisiko tidur.", "strategy": "Bagikan info produk populer untuk engage ulang."},
    {"name": "Need Attention", "char": "Sering interaksi tapi nilai belanja kecil.", "strategy": "Limited Time Offer untuk naikkan nilai belanja."},
    {"name": "At Risk", "char": "Dulu aktif, sekarang jarang kembali.", "strategy": "Pendekatan personal (Email/WA)."},
    {"name": "Promising", "char": "Cukup aktif bertransaksi (Potensial).", "strategy": "Bangun Brand Awareness."},
    {"name": "Require Activation", "char": "Baru bergabung (New User).", "strategy": "Program loyalitas & hadiah first order."},
    {"name": "Potential Loyalist", "char": "Tanda positif jadi loyal.", "strategy": "Rekomendasi produk & member special."},
    {"name": "Loyal Customers", "char": "Belanja teratur & nilai tinggi.", "strategy": "Libatkan kegiatan perusahaan & Up-selling."},
    {"name": "Champions", "char": "Pelanggan terbaik (RFM Tertinggi).", "strategy": "Reward VIP & Brand Advocate."}
]

def assign_paper_based_labels(rfm_df, k_clusters):
    cluster_summary = rfm_df.groupby('Cluster').agg({
        'Recency': 'mean', 'Frequency': 'mean', 'Monetary': 'mean'
    }).reset_index()

    # Ranking Score (R rendah=Bagus, F&M tinggi=Bagus)
    def normalize(series):
        return (series - series.min()) / (series.max() - series.min() + 0.0001)

    norm_r = 1 - normalize(cluster_summary['Recency']) 
    norm_f = normalize(cluster_summary['Frequency'])
    norm_m = normalize(cluster_summary['Monetary'])
    
    cluster_summary['Score'] = (norm_r * 1.0) + (norm_f * 0.8) + (norm_m * 0.8)
    cluster_summary = cluster_summary.sort_values('Score').reset_index(drop=True)
    
    total_paper_segments = len(PAPER_SEGMENTS_RANKED)
    mapping_dict = {}
    
    for i, row in cluster_summary.iterrows():
        idx_mapped = int(i * (total_paper_segments / k_clusters))
        segment_info = PAPER_SEGMENTS_RANKED[min(idx_mapped, total_paper_segments - 1)]
        mapping_dict[str(row['Cluster'])] = segment_info
        
    return mapping_dict

def get_gemini_insight(api_key, cluster_name, row, global_avg):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash') 
        prompt = f"""
        use indonesian language,
        Act as Senior Marketing Analyst. Analyze segment: '{cluster_name}'.
        Data: Recency {row['Recency']:.1f} (Avg {global_avg['Recency']:.1f}), Frequency {row['Frequency']:.1f}, Monetary ${row['Monetary']:.0f}.
        Task: 1 Sentence behavior interpretation & 2 Specific tactics. Output plain text.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error AI: {str(e)}"

# =========================================================
# 3. SIDEBAR & SETUP
# =========================================================
st.sidebar.title("🔧 Dashboard Control")
DATA_FILE = 'online_retail_II.csv'

if not os.path.exists(DATA_FILE):
    st.error(f"File {DATA_FILE} tidak ditemukan!")
    st.stop()

k_value = st.sidebar.slider("Jumlah Cluster (K)", 3, 10, 5)
insight_method = st.sidebar.radio("Metode Interpretasi:", ["Rule-Based", "Gemini AI"])
api_key = st.sidebar.text_input("Gemini API Key:", type="password") if insight_method == "Gemini AI" else ""

# =========================================================
# 4. MAIN PROCESSING
# =========================================================
st.title("🛍️ Advanced Customer Segmentation")

# Load & Process
df_clean = load_and_clean_data(DATA_FILE)
rfm_df = calculate_rfm_clv(df_clean)

# ML Pipeline
rfm_log = rfm_df[['Recency', 'Frequency', 'Monetary']].apply(np.log1p)
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)
pca = PCA(n_components=2)
pca_components = pca.fit_transform(rfm_scaled)

kmeans = KMeans(n_clusters=k_value, random_state=42, n_init='auto')
clusters = kmeans.fit_predict(rfm_scaled)

rfm_df['Cluster'] = clusters.astype(str)
rfm_df['PC1'] = pca_components[:, 0]
rfm_df['PC2'] = pca_components[:, 1]
segment_map = assign_paper_based_labels(rfm_df, k_value)
rfm_df['Segment Name'] = rfm_df['Cluster'].apply(lambda x: segment_map[x]['name'])

# =========================================================
# 5. TABS
# =========================================================
tab_viz, tab_rfm_dist, tab_clv, tab_insight, tab_explore = st.tabs([
    "Visualisasi Cluster", 
    "Distribusi RFM",
    "CLV & Risk",
    "Insight", 
    "Explorer"
])

with tab_viz:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("2D PCA Plot")
        fig = px.scatter(rfm_df, x="PC1", y="PC2", color="Segment Name", 
                         color_discrete_sequence=px.colors.qualitative.Bold, title="Cluster Separation")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("3D RFM Plot")
        p_data = rfm_df.copy()
        for c in ['Recency','Frequency','Monetary']: p_data[f'Log_{c}'] = np.log1p(p_data[c])
        fig3 = px.scatter_3d(p_data, x='Log_Recency', y='Log_Frequency', z='Log_Monetary', color='Segment Name',
                             color_discrete_sequence=px.colors.qualitative.Bold, opacity=0.6)
        st.plotly_chart(fig3, use_container_width=True)

# --- TAB: DISTRIBUSI RFM (UPDATED VISUALIZATION) ---
with tab_rfm_dist:
    st.subheader("Average RFM by Cluster")
    
    # Hitung rata-rata per Cluster/Segment
    avg_rfm = rfm_df.groupby(['Cluster', 'Segment Name'])[['Recency', 'Frequency', 'Monetary']].mean().reset_index()
    
    # 1. Tabel Ringkasan (Formatted)
    st.markdown("**Tabel Rata-rata per Segment**")
    st.dataframe(
        avg_rfm.style.format("{:.2f}", subset=['Recency', 'Frequency', 'Monetary']), 
        use_container_width=True
    )
    
    st.divider()
    
    # 2. Bar Charts (Sesuai Referensi Gambar)
    col_h1, col_h2, col_h3 = st.columns(3)
    
    with col_h1:
        fig_r = px.bar(avg_rfm, x="Segment Name", y="Recency", color="Segment Name",
                       title="Average Recency by Cluster",
                       color_discrete_sequence=px.colors.qualitative.Bold)
        fig_r.update_layout(showlegend=False)
        st.plotly_chart(fig_r, use_container_width=True)
        
    with col_h2:
        fig_f = px.bar(avg_rfm, x="Segment Name", y="Frequency", color="Segment Name",
                       title="Average Frequency by Cluster",
                       color_discrete_sequence=px.colors.qualitative.Bold)
        fig_f.update_layout(showlegend=False)
        st.plotly_chart(fig_f, use_container_width=True)
        
    with col_h3:
        fig_m = px.bar(avg_rfm, x="Segment Name", y="Monetary", color="Segment Name",
                       title="Average Monetary by Cluster",
                       color_discrete_sequence=px.colors.qualitative.Bold)
        fig_m.update_layout(showlegend=False)
        st.plotly_chart(fig_m, use_container_width=True)

with tab_clv:
    st.subheader("Customer Lifetime Value (CLV) & Risiko Churn")
    
    # Penjelasan Edukatif
    with st.expander("📚 Tujuan", expanded=True):
        st.markdown("""
        Halaman ini membantu Anda memahami **Nilai Pelanggan** dan **Risiko Kehilangan Pelanggan**:
        
        1.  **Apa itu CLV (Historical)?**
            Ini adalah total uang yang sudah dibelanjakan pelanggan sejak pertama kali beli sampai sekarang. 
            * **High CLV:** "Pelanggan Sultan" (Banyak belanja).
            * **Low CLV:** "Pelanggan Hemat" (Sedikit belanja).
            
        2.  **Apa itu Churn Risk?**
            Prediksi kemungkinan pelanggan pergi meninggalkan bisnis Anda. Diukur dari kapan terakhir kali mereka belanja (*Recency*).
            * **High Risk:** Sudah lama sekali tidak belanja (Bahaya!).
            * **Low Risk:** Baru saja belanja (Aman).
        --------------------------------------------------------------------------------------------------------------------------------
        **Penjelasan Rule-Based:**
        * **CLV (Historical)** = total belanja historis (HistoricalCLV = Monetary)
        * **Kategori CLV** ditentukan otomatis per dataset:
            * **High CLV** = pelanggan di ≈30% teratas nilai HistoricalCLV (>= persentil 70 / q70)
            * **Low CLV** = pelanggan di ≈30% terbawah nilai HistoricalCLV (<= persentil 30 / q30)
            * **Medium CLV** = sisanya
        * **Churn Risk** ditentukan dari Recency (hari sejak transaksi terakhir) memakai persentil:
            * **High Risk** = ≈30% recency tertinggi (>= persentil 70 / r_q70) → lama tidak belanja
            * **Low Risk** = ≈30% recency terendah (<= persentil 30 / r_q30) → baru belanja
            * **Medium Risk** = sisanya
        """)
    
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 💎 Pelanggan paling bernilai")
        val_counts = rfm_df["CLV_ValueSegment"].value_counts().reset_index()
        val_counts.columns = ["CLV_ValueSegment", "Count"]
        
        fig_val = px.bar(val_counts, 
                         x="CLV_ValueSegment", 
                         y="Count", 
                         color="CLV_ValueSegment", 
                         title="Jumlah User berdasarkan Kategori Nilai Uang (CLV)",
                         color_discrete_map={"High CLV": "#00CC96", "Medium CLV": "#636EFA", "Low CLV": "#EF553B"})
        st.plotly_chart(fig_val, use_container_width=True)
        st.caption("ℹ️ **Tips:** Fokuskan budget marketing & layanan VIP untuk grup **High CLV**.")
        
    with c2:
        st.markdown("#### ⚠️ Pelanggan yang harus diselamatkan")
        matrix = rfm_df.groupby(["CLV_ValueSegment", "Churn_Risk"]).size().reset_index(name="Count")
        
        # Order categories for logic visual
        risk_order = ["Low Risk", "Medium Risk", "High Risk"]
        val_order = ["Low CLV", "Medium CLV", "High CLV"]
        
        fig_mx = px.scatter(matrix, x="Churn_Risk", y="CLV_ValueSegment", size="Count", color="Count", 
                            title="Matriks Prioritas (Semakin besar bulatan, semakin banyak orang)",
                            category_orders={"Churn_Risk": risk_order, "CLV_ValueSegment": val_order})
        st.plotly_chart(fig_mx, use_container_width=True)
        st.caption("ℹ️ **Tips:** Perhatikan titik **High CLV + High Risk**. Itu adalah pelanggan VIP yang hampir hilang! Segera hubungi mereka.")

with tab_insight:
    summary = rfm_df.groupby(['Cluster', 'Segment Name']).mean(numeric_only=True).reset_index().sort_values('Monetary', ascending=False)
    global_avg = rfm_df[['Recency', 'Frequency', 'Monetary']].mean()
    
    for idx, row in summary.iterrows():
        p_info = segment_map[row['Cluster']]
        with st.expander(f"📌 {row['Segment Name']}", expanded=True):
            
            # TAMPILKAN STATISTIK (Agar tidak hilang)
            c_stat1, c_stat2, c_stat3, c_txt = st.columns([1, 1, 1, 3])
            c_stat1.metric("Recency", f"{row['Recency']:.1f} Hari")
            c_stat2.metric("Frequency", f"{row['Frequency']:.1f}x")
            c_stat3.metric("Monetary", f"${row['Monetary']:,.0f}")
            
            with c_txt:
                if insight_method == "Rule-Based":
                    st.write(f"**Behavior:** {p_info['char']}")
                    st.info(f"**Tactic:** {p_info['strategy']}")
                elif api_key:
                    st.markdown(get_gemini_insight(api_key, row['Segment Name'], row, global_avg))
                else:
                    st.warning("Butuh API Key untuk AI.")

with tab_explore:
    st.subheader("Data Explorer")
    
    # --- Layout Filter Baris 1 ---
    r1_c1, r1_c2, r1_c3 = st.columns(3)
    
    with r1_c1:
        search_cust = st.text_input("🔍 Cari Customer ID:", placeholder="Contoh: 12345")
    
    with r1_c2:
        filter_cluster = st.multiselect("📂 Filter Cluster:", options=sorted(rfm_df['Cluster'].unique()))
        
    with r1_c3:
        filter_segment = st.multiselect("🏷️ Filter Segment Name:", options=sorted(rfm_df['Segment Name'].unique()))
    
    # --- Layout Filter Baris 2 ---
    r2_c1, r2_c2 = st.columns(2)
    
    with r2_c1:
        filter_clv = st.multiselect("💎 Filter CLV Value Segment:", options=sorted(rfm_df['CLV_ValueSegment'].unique()))
        
    with r2_c2:
        filter_churn = st.multiselect("⚠️ Filter Churn Risk:", options=sorted(rfm_df['Churn_Risk'].unique()))
    
    # --- Logic Filter ---
    df_show = rfm_df.copy()
    
    if search_cust:
        df_show = df_show[df_show['Customer_Id'].str.contains(search_cust, case=False)]
    
    if filter_cluster:
        df_show = df_show[df_show['Cluster'].isin(filter_cluster)]
        
    if filter_segment:
        df_show = df_show[df_show['Segment Name'].isin(filter_segment)]
        
    if filter_clv:
        df_show = df_show[df_show['CLV_ValueSegment'].isin(filter_clv)]
        
    if filter_churn:
        df_show = df_show[df_show['Churn_Risk'].isin(filter_churn)]
        
    # Tampilkan Data
    st.write(f"Menampilkan **{len(df_show)}** data pelanggan.")
    st.dataframe(df_show, use_container_width=True)

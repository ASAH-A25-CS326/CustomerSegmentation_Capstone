# 🛍️ Advanced Customer Segmentation & CLV Dashboard

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![Status](https://img.shields.io/badge/Status-Active-success)

Aplikasi dashboard interaktif untuk melakukan segmentasi pelanggan ritel menggunakan metode **RFM (Recency, Frequency, Monetary)** dan **Machine Learning (K-Means Clustering)**. Aplikasi ini juga mengintegrasikan analisis **Customer Lifetime Value (CLV)** dan **Generative AI (Google Gemini)** untuk memberikan strategi pemasaran yang dipersonalisasi.

## 🌟 Fitur Utama

### 1. 📊 Advanced Segmentation (K-Means + RFM)
- Menggunakan algoritma **K-Means Clustering** untuk mengelompokkan pelanggan.
- [cite_start]**Dynamic Labeling:** Cluster yang terbentuk (K=3 s/d 10) secara otomatis dipetakan ke **10 Profil Pelanggan** berdasarkan riset (Paper Analisis RFM), seperti *Champions, Hibernating, Loyal Customers*, dll.
- Visualisasi persebaran data menggunakan **PCA 2D** dan **3D Scatter Plot**.

### 2. 💰 CLV & Churn Risk Analysis
- **Value-based Segmentation:** Mengkategorikan pelanggan menjadi *High, Medium,* dan *Low CLV*.
- **Risk Matrix:** Matriks prioritas untuk mengidentifikasi pelanggan bernilai tinggi yang berisiko pergi (*High CLV + High Churn Risk*).

### 3. 🤖 AI-Powered Insights
- Terintegrasi dengan **Google Gemini AI** via API.
- Memberikan interpretasi perilaku dan **rekomendasi taktik marketing spesifik** untuk setiap segmen secara otomatis.

### 4. 📈 Comprehensive Visualization
- Bar chart perbandingan rata-rata Recency, Frequency, dan Monetary antar cluster.
- Histogram distribusi pelanggan.
- Data Explorer dengan fitur pencarian dan filtering canggih.

---

## 🛠️ Instalasi & Cara Menjalankan

Ikuti langkah-langkah ini untuk menjalankan aplikasi di komputer lokal Anda:

### 1. Clone Repository
```bash
git clone [https://github.com/USERNAME_ANDA/NAMA_REPO.git](https://github.com/USERNAME_ANDA/NAMA_REPO.git)
cd NAMA_REPO
```
### 2. Buat Virtual Environment (Opsional tapi Disarankan)
```bash
python -m venv myenv
# Windows
myenv\Scripts\activate
# Mac/Linux
source myenv/bin/activate
```
### 3. Install DependenciesPastikan Anda memiliki file requirements.txt.
```bash
pip install -r requirements.txt
```
### 4. Siapkan Dataset
Pastikan file dataset bernama online_retail_II.csv berada di dalam folder utama proyek.
### 5. Jalankan Aplikasi
```Bash
streamlit run app.py
```
#### 📂 Struktur Proyek
```text
📦 customer-segmentation-app
 ┣ 📜 app.py                 # Main application code
 ┣ 📜 online_retail_II.csv   # Dataset (Pastikan file ini ada)
 ┣ 📜 requirements.txt       # List library python
 ┗ 📜 README.md              # Dokumentasi proyek
```
#### 🧠 Logika Segmentasi (Rule-Based)
| Nama Segmen | Karakteristik Utama | Strategi Singkat |
| :--- | :--- | :--- |
| **Champions** 🏆 | RFM Tertinggi. Baru beli, sering beli, belanja banyak. | Reward VIP, Early Access. |
| **Loyal Customers** 💖 | Belanja teratur dengan nilai transaksi tinggi. | Upselling, libatkan di komunitas. |
| **Potential Loyalist** ↗️ | Tanda positif menjadi loyal. | Rekomendasi produk, membership. |
| **Promising** ✨ | Cukup aktif bertransaksi. | Brand awareness. |
| **Need Attention** ⚠️ | Sering interaksi tapi nilai belanja kecil. | Limited time offer. |
| **About To Sleep** 💤 | Recency menurun, berisiko tidur. | Info produk populer untuk engage ulang. |
| **At Risk** 🚨 | Dulu aktif, sekarang jarang kembali. | Pendekatan personal (Email/WA). |
| **Hibernating** 🐻 | Pola pembelian rendah & kurang terlibat. | Diskon relevan untuk re-branding. |
| **Churned Customers** 📉 | Sudah lama sekali tidak bertransaksi. | Win-back campaign eksklusif. |
| **Require Activation** 🆕 | Baru bergabung (New User). | Program loyalitas & hadiah first order. |

#### ⚙️ Teknologi yang Digunakan
- Language: Python 3.10+
- Web Framework: Streamlit
- Data Manipulation: Pandas, NumPyMachine Learning: Scikit-learn (KMeans, PCA, StandardScaler)
- Visualization: Plotly Express
- AI Integration: Google Generative AI (Gemini)

#### 📝 Referensi
Logika penamaan segmen dalam proyek ini diadaptasi dari paper penelitian:
> Analisis Segmentasi Menggunakan Metode RFM dalam Mengidentifikasi Perilaku Pelanggan (Utama & Ardiansah, 2023)21

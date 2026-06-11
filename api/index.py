from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import os

app = FastAPI(title="TripMate AI API - Dynamic Mode")

# Global Variables to store ML data in memory
df = pd.DataFrame()
tfidf = None
tfidf_matrix = None

# URL API Backend Laravel (Gunakan URL asli saat di hosting)
# Default menggunakan localhost untuk mode testing
LARAVEL_API_URL = os.getenv("LARAVEL_API_URL", "http://127.0.0.1:8000/api/destinations-data")

def sync_data_from_laravel():
    """Fungsi untuk menarik data dari Laravel dan melatih model TF-IDF secara instan"""
    global df, tfidf, tfidf_matrix
    
    try:
        print(f"Mengambil data dari: {LARAVEL_API_URL}")
        response = requests.get(LARAVEL_API_URL, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if len(data) == 0:
                print("⚠️ Peringatan: Data dari Laravel kosong!")
                return False
                
            # Konversi JSON ke DataFrame
            new_df = pd.DataFrame(data)
            
            # Pastikan kolom-kolom wajib ada (menggunakan nama JSON dari Laravel)
            for col in ['nama_wisata', 'kategori', 'provinsi', 'kota_kabupaten', 'price', 'deskripsi_bersih']:
                if col not in new_df.columns:
                    print(f"⚠️ Error: Kolom {col} tidak ditemukan dalam response API")
                    return False

            # Normalisasi tipe data dan penanganan missing values
            new_df['deskripsi_bersih'] = new_df['deskripsi_bersih'].fillna('')
            new_df['kategori'] = new_df['kategori'].fillna('').astype(str)
            new_df['provinsi'] = new_df['provinsi'].fillna('').astype(str)
            new_df['kota_kabupaten'] = new_df['kota_kabupaten'].fillna('').astype(str)
            new_df['price'] = pd.to_numeric(new_df['price'], errors='coerce').fillna(0).astype(int)

            # Membuat Metadata Soup (menggunakan nama kolom sesuai JSON web.php)
            new_df['metadata_soup'] = (new_df['kategori'] + ' ') * 3 + new_df['provinsi'] + ' ' + new_df['kota_kabupaten'] + ' ' + new_df['deskripsi_bersih']
            new_df['metadata_soup'] = new_df['metadata_soup'].str.lower()

            # Stopwords
            indonesian_stopwords = [
                'yang', 'di', 'dan', 'adalah', 'untuk', 'sebuah', 'ini', 'itu',
                'atau', 'pada', 'ke', 'dari', 'terdapat', 'menjadi', 'salah', 'satu', 'juga', 'ada'
            ]

            # Inisialisasi dan Latih TF-IDF
            new_tfidf = TfidfVectorizer(stop_words=indonesian_stopwords, ngram_range=(1, 2))
            new_matrix = new_tfidf.fit_transform(new_df['metadata_soup'])
            
            # Ganti state global secara aman (Atomic)
            df = new_df
            tfidf = new_tfidf
            tfidf_matrix = new_matrix
            
            print(f"Model TF-IDF berhasil dilatih ulang dengan {len(df)} destinasi wisata!")
            return True
        else:
            print(f"Gagal mengambil data: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"Error sinkronisasi: {str(e)}")
        return False

# Panggil fungsi sync saat server pertama kali menyala
@app.on_event("startup")
async def startup_event():
    sync_data_from_laravel()

# Format JSON dari Controller
class UserRequest(BaseModel):
    user_categories: Union[str, List[str]]
    user_province_input: Optional[str] = None
    max_budget: Optional[int] = None
    top_n: Optional[int] = 5

@app.get("/")
def read_root():
    return {
        "message": "API TripMate AI Dinamis Aktif!",
        "total_destinasi_tersedia": len(df) if not df.empty else 0
    }

@app.post("/reload-data")
def reload_data(background_tasks: BackgroundTasks):
    """Endpoint untuk memerintahkan AI me-reload data dari MySQL Laravel"""
    background_tasks.add_task(sync_data_from_laravel)
    return {"message": "Proses sinkronisasi data dari MySQL sedang berjalan di latar belakang."}

@app.post("/recommend")
def get_recommendations(req: UserRequest):
    if df.empty or tfidf is None:
        raise HTTPException(status_code=503, detail="Model belum siap atau data kosong. Silakan panggil /reload-data terlebih dahulu.")

    # 1. Inisialisasi Masking
    mask = pd.Series(True, index=df.index)

    # 2. Filter Lokasi
    if req.user_province_input:
        location_clean = req.user_province_input.lower().strip()
        mask = mask & (
            df['provinsi'].str.lower().str.contains(location_clean, na=False) |
            df['kota_kabupaten'].str.lower().str.contains(location_clean, na=False)
        )

    # 3. Filter Harga Mutlak
    if req.max_budget is not None:
        mask = mask & (df['price'] <= req.max_budget)

    # 2.5 Filter Kategori Ketat
    if req.user_categories:
        if isinstance(req.user_categories, list):
            cat_str = ", ".join(req.user_categories)
        else:
            cat_str = str(req.user_categories)
            
        if cat_str.strip():
            cat_list = [c.strip().lower() for c in cat_str.split(',')]
            cat_mask = pd.Series(False, index=df.index)
            for cat in cat_list:
                cat_mask = cat_mask | df['kategori'].str.lower().str.contains(cat, regex=False, na=False)
            mask = mask & cat_mask

    matched_indices = df[mask].index
    if len(matched_indices) == 0:
        return []

    df_filtered = df.loc[matched_indices].reset_index(drop=True)
    matrix_filtered = tfidf_matrix[matched_indices]

    # 4. Penyesuaian Input Kategori
    if isinstance(req.user_categories, list):
        user_input_cleaned = " ".join(req.user_categories).lower()
    else:
        user_input_cleaned = str(req.user_categories).lower()

    # 5. Prediksi Kemiripan (Cosine Similarity)
    user_vector = tfidf.transform([user_input_cleaned])
    similarity_scores = cosine_similarity(user_vector, matrix_filtered).flatten()

    # 6. Ambil Top-N Rekomendasi
    top_indices = similarity_scores.argsort()[::-1][:req.top_n]

    recommendations = df_filtered.iloc[top_indices].copy()
    recommendations['score_akurasi'] = similarity_scores[top_indices]

    # Pilih kolom untuk dikembalikan (sesuai JSON yang dulu diberikan model Colab)
    output_cols = ['nama_wisata', 'kategori', 'price', 'provinsi', 'kota_kabupaten', 'score_akurasi']
    hasil_akhir = recommendations[output_cols]

    return hasil_akhir.to_dict(orient='records')
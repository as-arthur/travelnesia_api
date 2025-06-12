from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import joblib
import pandas as pd
import numpy as np
import logging
from math import radians, sin, cos, sqrt, atan2
from typing import Optional, List, Dict
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_artifacts():
    """Memuat semua artifak yang diperlukan dari folder /artifacts."""
    artifacts = {}
    try:
        # Asumsi skrip ini ada di dalam folder, misal /app, dan artifacts ada di root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        artifacts_dir = os.path.join(current_dir, '..', 'artifacts')
        artifacts_dir = os.path.normpath(artifacts_dir)
        
        logger.info(f"Mencari artifacts di: {artifacts_dir}")
        if not os.path.exists(artifacts_dir):
            raise FileNotFoundError(f"Direktori artifacts tidak ditemukan: {artifacts_dir}")
            
        # Memuat semua file yang diperlukan
        artifacts['df_place'] = pd.read_csv(os.path.join(artifacts_dir, 'places_cleaned.csv'))
        artifacts['df_rating'] = pd.read_csv(os.path.join(artifacts_dir, 'ratings_cleaned.csv'))
        artifacts['model_svd'] = joblib.load(os.path.join(artifacts_dir, 'model_svd.pkl'))
        artifacts['trainset_full'] = joblib.load(os.path.join(artifacts_dir, 'trainset_full.pkl'))
        
        logger.info(">>> Semua artifacts berhasil dimuat. API siap melayani permintaan. <<<")
        return artifacts, True
        
    except Exception as e:
        logger.error(f"!!! FATAL ERROR saat memuat artifacts: {e} !!!", exc_info=True)
        return {}, False

ARTIFACTS, is_model_ready = load_artifacts()


#fungsi untuk menghitung jarak antar dua titik koordinat menggunakan rumus Haversine
def haversine_distance_calculator(lat1, lon1, lat2, lon2):
    """Menghitung jarak antar dua titik koordinat."""
    R = 6371
    try:
        lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, map(float, [lat1, lon1, lat2, lon2]))
    except (ValueError, TypeError, AttributeError):
        return float('inf')
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# --- Kumpulan Fungsi 'Getter' untuk setiap strategi ---

def get_popular_global(df_place, top_n=100):
    """Mengembalikan tempat terpopuler global berdasarkan rating."""
    return df_place.sort_values(by='Rating', ascending=False).head(top_n)

def get_popular_nearby(df_place, lat, lon, top_n=100):
    """Mengembalikan tempat terpopuler terdekat berdasarkan jarak lalu rating."""
    df_copy = df_place.copy()
    distances = df_copy.apply(lambda row: haversine_distance_calculator(lat, lon, row['Lat'], row['Long']), axis=1)
    df_copy['distance_km'] = distances
    return df_copy.sort_values(by=['distance_km', 'Rating'], ascending=[True, False]).head(top_n)

def get_trending_by_review(df_place, df_rating, top_n=100):
    """Mengembalikan tempat trending berdasarkan jumlah review."""
    review_counts = df_rating['Place_Id'].value_counts().reset_index()
    review_counts.columns = ['Place_Id', 'review_count']
    trending_places = pd.merge(df_place, review_counts, on='Place_Id', how='left')
    trending_places['review_count'] = trending_places['review_count'].fillna(0)
    return trending_places.sort_values(by='review_count', ascending=False).head(top_n)

def get_by_category(df_place, category, top_n=100):
    """Mengembalikan tempat berdasarkan kategori, diurutkan dari rating tertinggi."""
    filtered = df_place[df_place['Category'].str.lower() == category.lower()]
    return filtered.sort_values(by='Rating', ascending=False).head(top_n)

def get_personalized_global(user_id, model, trainset, df_place, df_rating, top_n=50):
    """Mengembalikan rekomendasi personal (CF) global yang sudah diperkaya dengan detail."""
    rated_by_user = set(df_rating[df_rating['User_Id'] == user_id]['Place_Id'])
    all_item_ids = [trainset.to_raw_iid(inner_id) for inner_id in trainset.all_items()]
    candidates = [item_id for item_id in all_item_ids if item_id not in rated_by_user]
    if not candidates: return pd.DataFrame()

    predictions = sorted([model.predict(uid=user_id, iid=item_id) for item_id in candidates], key=lambda x: x.est, reverse=True)
    recs_df = pd.DataFrame([(pred.iid, pred.est) for pred in predictions[:top_n]], columns=['Place_Id', 'predicted_rating'])
    return pd.merge(recs_df, df_place, on='Place_Id', how='left')

def get_hybrid_recommendations(user_id, model, trainset, df_place, lat, lon, radius_km, top_n, category=None):
    """
    Logika Hybrid v3.2: Memperbaiki bug pada 'Popular Global Filler' agar tetap menghormati filter kategori.
    """
    source_info = f"Personalized Nearby (Radius {radius_km}km)"
    place_pool = df_place.copy()

    # Tahap 1: Filter Kategori (jika ada)
    if category:
        place_pool = place_pool[place_pool['Category'].str.lower() == category.lower()]
        source_info += f" in Category '{category}'"
        if place_pool.empty:
            return pd.DataFrame(), f"Tidak ada tempat untuk kategori '{category}'."

    # Tahap 2: Filter Lokasi
    all_nearby = get_popular_nearby(place_pool, lat, lon, top_n=None)
    nearby_places = all_nearby[all_nearby['distance_km'] <= radius_km]
    if nearby_places.empty:
        return pd.DataFrame(), f"Tidak ada tempat dalam radius untuk filter yang diberikan."

    # Tahap 3: Personalisasi
    predictions = sorted([model.predict(uid=user_id, iid=pid) for pid in nearby_places['Place_Id']], key=lambda x: x.est, reverse=True)
    recs = nearby_places.copy()
    recs['predicted_rating'] = recs['Place_Id'].map({pred.iid: pred.est for pred in predictions})
    recs = recs.dropna(subset=['predicted_rating']).sort_values('predicted_rating', ascending=False)
    
    # --- LOGIKA FALLBACK BERTINGKAT ---

    # Fallback 1: Isi dengan Populer Terdekat (dalam kategori yang sama)
    if len(recs) < top_n:
        source_info += " + Popular Nearby Filler"
        existing_ids = recs['Place_Id'].tolist()
        needed = top_n - len(recs)
        filler_recs = nearby_places[~nearby_places['Place_Id'].isin(existing_ids)].head(needed)
        recs = pd.concat([recs, filler_recs], ignore_index=True)
        
    # Fallback 2: Isi dengan Populer Global (dalam kategori yang sama)
    if len(recs) < top_n:
        source_info += " + Popular Global Filler"
        existing_ids = recs['Place_Id'].tolist()
        needed = top_n - len(recs)
        
        # Gunakan 'place_pool' yang sudah difilter
        df_popular_global_filtered = get_popular_global(place_pool, top_n=len(place_pool))
        
        filler_recs_global = df_popular_global_filtered[~df_popular_global_filtered['Place_Id'].isin(existing_ids)].head(needed)
        
        # Pastikan kolom konsisten sebelum concat
        for col in ['distance_km', 'predicted_rating']:
            if col not in filler_recs_global.columns:
                filler_recs_global[col] = np.nan

        recs = pd.concat([recs, filler_recs_global], ignore_index=True)
        
    return recs.head(top_n), source_info

# --- Orkestrator Utama ---
def get_discovery_recommendations(user_id, category, location, view_mode, radius_km, top_n):
    """Orkestrator utama yang memanggil fungsi-fungsi helper yang sesuai."""
    df_p, df_r, model, ts = (ARTIFACTS[k] for k in ['df_place', 'df_rating', 'model_svd', 'trainset_full'])
    user_exists = (user_id is not None and user_id in df_r['User_Id'].values)
    
    recs, source = pd.DataFrame(), "Unknown"

    if user_exists and location:
        recs, source = get_hybrid_recommendations(user_id, model, ts, df_p, location['lat'], location['lon'], radius_km, top_n, category)
    elif view_mode == 'trending':
        recs, source = get_trending_by_review(df_p, df_r, 100), "Trending by Review Count"
    elif category:
        recs, source = get_by_category(df_p, category, 100), f"Category: {category}"
    elif user_exists:
        recs, source = get_personalized_global(user_id, model, ts, df_p, df_r, top_n), "Personalized Global"
    elif location:
        recs, source = get_popular_nearby(df_p, location['lat'], location['lon'], top_n), "Popular Nearby (Guest)"
        
    if recs.empty:
        recs, source = get_popular_global(df_p, top_n), "Popular Global (Fallback)"
        
    if location and 'distance_km' not in recs.columns:
        distances = recs.apply(lambda r: haversine_distance_calculator(location['lat'], location['lon'], r.get('Lat'), r.get('Long')), axis=1)
        recs['distance_km'] = distances
        if 'Personalized' not in source and 'Nearby' not in source:
            recs, source = recs.sort_values('distance_km'), source + " + Sorted by Location"
            
    for col in ['distance_km', 'predicted_rating', 'review_count']:
        if col not in recs.columns:
            recs[col] = np.nan
            
    return recs.head(top_n), source


# model response dan request schemas
class DiscoveryRequest(BaseModel):
    user_id: Optional[int] = None
    category: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    view_mode: Optional[str] = 'default'
    top_n: Optional [int] = 5
    radius_km: Optional[int] = 10

class Location(BaseModel):
    lat: float
    lon: float

class Context(BaseModel):
    distance_km: Optional[float] = None
    predicted_rating: Optional[float] = None
    review_count: Optional[int] = None

class Place(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    city: Optional[str] = None
    price: Optional[int] = None
    rating: Optional[float] = None
    image_url: Optional[str] = None
    location: Location
    context: Context

class DiscoveryResponse(BaseModel):
    status: str
    code: int
    message: str
    source: str
    count: int
    recommendations: List[Place]

class HealthResponse(BaseModel):
    status: str
    message: str
    artifacts_loaded: Dict[str, bool]

@app.post("/discovery", response_model=DiscoveryResponse, tags=["Recommendations"])
async def discovery_endpoint(request: DiscoveryRequest):
    """Endpoint utama untuk mendapatkan rekomendasi tempat wisata."""
    if not is_model_ready:
        raise HTTPException(status_code=503, detail="Layanan tidak tersedia: Model sedang tidak siap.")

    try:
        location_dict = {'lat': request.lat, 'lon': request.lon} if request.lat is not None and request.lon is not None else None
        
        recs_df, source_info = get_discovery_recommendations(
            user_id=request.user_id,
            category=request.category,
            location=location_dict,
            view_mode=request.view_mode,
            radius_km=request.radius_km,
            top_n=request.top_n
        )

        if recs_df.empty:
            return DiscoveryResponse(status="success", code=200, message="Tidak ada rekomendasi yang cocok ditemukan.", source=source_info, count=0, recommendations=[])

        response_list = []
        for _, row in recs_df.iterrows():
            row_dict = row.replace({np.nan: None}).to_dict()
            place = Place(
                id=int(row_dict.get('Place_Id')),
                name=row_dict.get('Place_Name'),
                description=row_dict.get('Description'),
                category=row_dict.get('Category'),
                city=row_dict.get('City'),
                price=int(row_dict.get('Price')) if row_dict.get('Price') is not None else None,
                rating=float(row_dict.get('Rating')) if row_dict.get('Rating') is not None else None,
                location=Location(lat=row_dict.get('Lat'), lon=row_dict.get('Long')),
                context=Context(
                    distance_km=float(row_dict.get('distance_km')) if row_dict.get('distance_km') is not None else None,
                    predicted_rating=float(row_dict.get('predicted_rating')) if row_dict.get('predicted_rating') is not None else None,
                    review_count=int(row_dict.get('review_count')) if row_dict.get('review_count') is not None else None
                )
            )
            response_list.append(place)
            
        return DiscoveryResponse(
            status="success",
            code=200,
            message="Rekomendasi berhasil diambil.",
            source=source_info,
            count=len(response_list),
            recommendations=response_list
        )

    except Exception as e:
        logger.error(f"Terjadi kesalahan tak terduga di endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Terjadi kesalahan internal pada server.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) # type 'ignore'
import pandas as pd 
import sqlite3
from pathlib import Path 
import torch
from sentence_transformers import SentenceTransformer, util

# Ayarlar
BASE_PATH = Path("data/raw/usda")
SURVEY_PATH = BASE_PATH / "survey"
FOUNDATION_PATH = BASE_PATH / "foundation"
# Veritabanının kaydedileceği klasör
PROCESSED_DATA_PATH = Path("data/processed")
# Klasör yoksa oluştur
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
# Analiz yapacağımız val etiketlerinin yolu
VAL_LABELS_PATH = Path("data/processed/yolo_dataset/labels/val")

TARGET_CLASSES = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 
    'beef_tartare', 'beet_salad', 'beignets', 'bibimbap', 
    'bread_pudding', 'breakfast_burrito'
]

EXCEPTIONS = {
    'beef_carpaccio': 'Beef, raw, lean',     # 'Fish, carp' bulmasını engellemek için 'Beef' ve 'Raw' vurgusu
    'beet_salad': 'Beets, salad',            # 'Beef' yerine 'Beets' (Pancar) vurgusu
    'beef_tartare': 'Steak tartare, raw',    # USDA'da bu isimle geçme ihtimali daha yüksek
    'baby_back_ribs': 'Pork ribs, cooked'    # Daha genel ve doğru bir kategori
}

# Modeli yükle 
print("🧠 NLP Modeli yükleniyor...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_average_mask_areas():
    """Val setindeki etiketlerden sınıf başına ortalama piksel alanını hesaplar."""
    VAL_LABELS_PATH = Path("data/processed/yolo_dataset/labels/val")
    class_areas = {i: [] for i in range(len(TARGET_CLASSES))}
    
    if not VAL_LABELS_PATH.exists():
        return {i: 40000 for i in range(len(TARGET_CLASSES))} # Fallback değeri

    for label_file in VAL_LABELS_PATH.glob("*.txt"):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3: continue
                
                cls_id = int(parts[0])
                coords = [float(x) for x in parts[1:]] # Normalize x,y çiftleri
                
                # Shoelace formülü ile normalize alan hesabı
                xs, ys = coords[0::2], coords[1::2]
                area = 0.5 * abs(sum(xs[i]*ys[i+1] - xs[i+1]*ys[i] for i in range(-1, len(xs)-1)))
                
                # 640x640 çözünürlüğe göre piksel alanına çevir
                pixel_area = area * (640 * 640) 
                class_areas[cls_id].append(pixel_area)

    return {cid: (sum(areas)/len(areas) if areas else 40000) for cid, areas in class_areas.items()}

def load_usda_data(folder_path):
    """Belirtilen klasördeki CSV'leri yükler."""
    print(f"📦 {folder_path.name} verileri yükleniyor...")
    # Dtype uyarısını engellemek için low_memory=False eklendi
    return {
        "food": pd.read_csv(folder_path / "food.csv", low_memory=False),
        "nutrient": pd.read_csv(folder_path / "food_nutrient.csv", low_memory=False),
        "portion": pd.read_csv(folder_path / "food_portion.csv", low_memory=False)
    }

def get_semantic_match(df, target_name, threshold=0.65):
    """Vektör benzerliği kullanarak anlamsal eşleşme yapar."""
    descriptions = df['description'].tolist()
    
    # Hedef kelimeyi ve veritabanı listesini vektöre çevir
    target_emb = model.encode(target_name, convert_to_tensor=True)
    desc_embs = model.encode(descriptions, convert_to_tensor=True, show_progress_bar=False)
    
    # Kosinüs Benzerliği hesapla
    cosine_scores = util.cos_sim(target_emb, desc_embs)[0]
    
    # En yüksek skoru ve indeksi bul
    best_idx = torch.argmax(cosine_scores).item()
    best_score = cosine_scores[best_idx].item()
    
    if best_score >= threshold:
        return df.iloc[best_idx], best_score
    return None, 0

def build_snapcal_db(survey_path, foundation_path, classes):
    print("Anlamsal Veri Madenciliği Başlıyor...")

    db_path = PROCESSED_DATA_PATH / "snapcal_local.db"

    # Referans alanları hesapla
    reference_areas = get_average_mask_areas()

    # SQLite Bağlantısını Kur
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Tabloyu oluştur
    cursor.execute('DROP TABLE IF EXISTS nutrition')
    cursor.execute('''
        CREATE TABLE nutrition (
                class_name TEXT PRIMARY KEY,
                source TEXT,
                usda_desc TEXT,
                calories REAL,
                protein REAL,
                fat REAL,
                carbs REAL,
                portion_g REAL,
                ref_area REAL
            )
        ''')

    # Kaynakları yükle
    sources = [
        {"name": "survey", "data": load_usda_data(survey_path), 
         "map": {208: 'calories', 203: 'protein', 204: 'fat', 205: 'carbs'}}, # Survey ID'leri
        {"name": "foundation", "data": load_usda_data(foundation_path), 
         "map": {1008: 'calories', 1003: 'protein', 1004: 'fat', 1005: 'carbs'}} # Foundation ID'leri
    ]


    for idx,cls in enumerate(classes):
        found = False
        search_term = EXCEPTIONS.get(cls, cls.replace("_", " "))

        for src in sources:
            if found: break 
            df_food = src["data"]["food"]

            # Arama değişkenlerini hazırla
            best_row = None
            current_score = 0.0 # Başlangıç değeri
            method_text = ""

            # 1. Aşama: Katı Arama
            match = df_food[df_food["description"].str.contains(search_term, case=False, na=False)]

            if not match.empty:
                best_row = match.iloc[0]
                current_score = 1.0  # Tam eşleşmede skor 1.00
                method_text = "🎯 Tam Eşleşme"
            else:
                # 2. Aşama: Anlamsal Arama
                print(f"🔍 Anlamsal arama yapılıyor: {cls}...")
                best_row, current_score = get_semantic_match(df_food, search_term)
                method_text = "🧠 Anlamsal"

            if best_row is not None:
                f_id = best_row["fdc_id"]
                n_df = src["data"]["nutrient"]
                p_df = src["data"]["portion"]

                # Makroları çek
                macro_vals = {v: 0.0 for v in src["map"].values()}
                
                for nid, name in src["map"].items():
                    val = n_df[(n_df["fdc_id"] == f_id) & (n_df["nutrient_id"] == nid)]["amount"]
                    if not val.empty: 
                        macro_vals[name] = float(val.iloc[0])

                
                portion = p_df[p_df["fdc_id"] == f_id]
                p_weight = float(portion.iloc[0]["gram_weight"]) if not portion.empty else 100.0

                ref_a = reference_areas.get(idx, 40000)
                # Veritabanına ekle
                cursor.execute('''INSERT INTO nutrition VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                    (cls, src["name"], best_row['description'], macro_vals['calories'], 
                    macro_vals['protein'], macro_vals['fat'], macro_vals['carbs'], p_weight, ref_a))
                
                print(f"✅ {cls} -> {best_row['description']} | Skor: {current_score:.2f} | (Kaynak: {src['name']}) | {method_text}")
                found = True
    
        if not found:
            print(f"❌ {cls} hiçbir kaynakta bulunamadı!")

    conn.commit()
    conn.close()
    print("\n🔥 Anlamsal Veritabanı başarıyla inşa edildi: snapcal_local.db")

if __name__ == "__main__":
    build_snapcal_db(SURVEY_PATH, FOUNDATION_PATH, TARGET_CLASSES)
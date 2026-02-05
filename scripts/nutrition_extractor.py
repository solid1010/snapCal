import pandas as pd 
import sqlite3
from pathlib import Path 

# Ayarlar
BASE_PATH = Path("data/raw/usda")
SURVEY_PATH = BASE_PATH / "survey"
FOUNDATION_PATH = BASE_PATH / "foundation"

TARGET_CLASSES = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 
    'beef_tartare', 'beet_salad', 'beignets', 'bibimbap', 
    'bread_pudding', 'breakfast_burrito'
]

def load_usda_data(folder_path):
    """Belirtilen klasördeki CSV'leri yükler."""
    print(f"📦 {folder_path.name} verileri yükleniyor...")
    return {
        "food": pd.read_csv(folder_path / "food.csv"),
        "nutrient": pd.read_csv(folder_path / "food_nutrient.csv"),
        "portion": pd.read_csv(folder_path / "food_portion.csv")
    }

def build_snapcal_db(survey__path,foundation_path,classes):
    print("Veri madenciliği başlıyor: Survey + Foundation birleştiriliyor...")

    # SQLite Bağlantısını Kur
    conn = sqlite3.connect("snapcal_local.db")
    cursor = conn.cursor()
    
    
    # Tabloyu oluştur (Eski tablo varsa siler, temiz kurulum yapar)
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
                portion_g REAL
            )
        ''')

    
    # Kaynakları yükle
    sources = [
        {"name": "survey", "data": load_usda_data(SURVEY_PATH)},
        {"name": "foundation", "data": load_usda_data(FOUNDATION_PATH)}
    ]

    macro_map = {1008: 'calories', 1003: 'protein', 1004: 'fat', 1005: 'carbs'}

    
    for cls in TARGET_CLASSES:
        found = False 
        search_term = cls.replace("_"," ")

        for src in sources:
            if found : 
                break 
            
            df_food = src["data"]["food"]

            # Arama yap

            match = df_food[df_food["description"].str.contains(search_term,case = False, na= False)]

            if not match.empty:
                best = match.iloc[0]
                f_id = best["fdc_id"]

                # Makroları çek
                n_df = src["data"]["nutrient"]

                macro_vals = {v: 0.0 for v in macro_map.values()}
                for nid, name in macro_map.items():
                    val = n_df[(n_df["fdc_id"] == f_id) & (n_df["nutrient_id"] == nid)]["amount"]
                    if not val.empty:
                        macro_vals[name] = float(val.iloc[0])


                p_df = src["data"]["portion"]
                portion = p_df[p_df["fdc_id"]==f_id]
                p_weight = float(portion.iloc[0]["gram_weight"]) if not portion.empty else 100.0


                cursor.execute('''INSERT INTO nutrition VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', 
                    (cls, src["name"], best['description'], nut_vals['calories'], 
                    nut_vals['protein'], nut_vals['fat'], nut_vals['carbs'], p_weight))
                
                print(f"✅ {cls} bulundu ({src['name']})")
                found = True

        if not found:
            print(f"❌ {cls} hiçbir kaynakta bulunamadı!")

    conn.commit()
    conn.close()
    print("\n🔥 Veritabanı başarıyla inşa edildi: snapcal_local.db")


if __name__ == "__main__":
    build_snapcal_db()
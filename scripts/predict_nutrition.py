import cv2
import sqlite3
from pathlib import Path
from ultralytics import YOLO

# 1. Yollar ve Ayarlar
# Kendi model yolunu buraya yaz (Örn: runs/detect/train/weights/best.pt)
MODEL_PATH = "/Users/alperen/Desktop/snapCal/runs/segment/runs/train/snapCal_v1_seg10/weights/best.pt"
DB_PATH = Path("data/processed/snapcal_local.db")

# 2. Modeli Yükle
print("Loading YOLO model...")
model = YOLO(MODEL_PATH)

# 3. Veritabanı Fonksiyonu
def get_nutrition_info(class_name):
    """Veritabanından yemeğin makrolarını çeker."""
    # 1. ADIM: İsimleri küçük harfe çevir ve sağındaki solundaki boşlukları temizle
    search_name = class_name.lower().strip()
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 2. ADIM: SQL sorgusunu da büyük/küçük harf duyarsız (LOWER) yapalım ki garanti olsun
        # SQL sorgusuna portion_g ve ref_area sütunlarını ekledik
        query = "SELECT usda_desc, calories, protein, fat, carbs, portion_g, ref_area FROM nutrition WHERE LOWER(class_name) = ?"
        
        cursor.execute(query, (search_name,))
        result = cursor.fetchone()
        conn.close()
        
        # DEBUG için terminale yazdıralım
        if result:
            print(f"✅ DB'den veri geldi: {search_name}")
        else:
            print(f"⚠️ DB'de '{search_name}' bulunamadı!")
            
        return result
    except Exception as e:
        print(f"❌ DB Hatası: {e}")
        return None

# 4. Kamerayı Başlat
cap = cv2.VideoCapture(0) # Web kamerası

print("📸 snapCal Canlı Tespit Başlıyor... (Çıkmak için 'q' tuşuna basın)")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # YOLO Tahmini (Hız için stream=True kullanılabilir)
    results = model(frame, conf=0.7, verbose=False)

    for r in results:
        if r.masks is not None: # Maske tespiti varsa
            for i, mask in enumerate(r.masks.data):
                # 1. Alan Hesabı (Piksel Sayısı)
                current_mask_area = int(mask.sum())
                
                cls_id = int(r.boxes.cls[i])
                cls_name = model.names[cls_id]
                nut_data = get_nutrition_info(cls_name)
                
                x1, y1, x2, y2 = map(int, r.boxes.xyxy[i])
                
                if nut_data:
                    # 2. Veriyi Parçala (7 Sütun)
                    desc, kcal_100, prot_100, fat_100, carb_100, p_weight, r_area = nut_data
                    
                    # 3. Porsiyon Ölçekleme (ref_area yoksa 40000 kullan)
                    ref_area_val = r_area if (r_area and r_area > 0) else 40000
                    scale_factor = current_mask_area / ref_area_val
                    
                    est_weight = p_weight * scale_factor
                    final_kcal = (kcal_100 / 100) * est_weight
                    
                    # UI Güncelleme
                    label = f"{cls_name.upper()}: ~{est_weight:.0f}g | {final_kcal:.0f} kcal"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Görüntüyü göster
    cv2.imshow("snapCal MVP - Real Time Nutrition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
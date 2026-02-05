import cv2
import sqlite3
from pathlib import Path
from ultralytics import YOLO

# 1. Yollar ve Ayarlar
# Kendi model yolunu buraya yaz (Örn: runs/detect/train/weights/best.pt)
MODEL_PATH = "/Users/alperen/Desktop/snapCal/runs/detect/runs/train/snapCal_v14/weights/best.pt" 
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
        query = "SELECT usda_desc, calories, protein, fat, carbs FROM nutrition WHERE LOWER(class_name) = ?"
        
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
        for box in r.boxes:
            # Sınıf adını al
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            
            # Besin verisini çek
            nut_data = get_nutrition_info(cls_name)
            
            print(f"YOLO Sınıfı: {cls_name} | DB'den Gelen Ham Veri: {nut_data}")
            
            # Kutu koordinatları
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Renk ve Çizim
            color = (0, 255, 0) # Yeşil
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            if nut_data:
                # Veri VARSA (Yeşil Kutu)
                desc, kcal, prot, fat, carb = nut_data
                label = f"{cls_name.upper()}: {kcal:.0f} kcal | P: {prot}g"
                color = (0, 255, 0)
                
                # Bilgi kutusu çizimi
                cv2.rectangle(frame, (x1, y1 - 45), (x1 + 250, y1), color, -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                # Veri YOKSA (Kırmızı Kutu ve Konsol Logu)
                print(f"⚠️ DB'de şu isim bulunamadı: '{cls_name}'")
                label = f"{cls_name} (VERI YOK)"
                color = (0, 0, 255)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Görüntüyü göster
    cv2.imshow("snapCal MVP - Real Time Nutrition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
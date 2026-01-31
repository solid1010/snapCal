import os
import shutil
import random
import json
from pathlib import Path
from ultralytics import YOLO

# 1. Sabit Yollar
RAW_DATA_PATH = Path("data/raw/food-101")
PROCESSED_PATH = Path("data/processed/yolo_dataset")

# 2. Odaklanacağımız 10 Sınıf
TARGET_CLASSES = [
    'Pizza', 'Hamburger', 'French fries', 'Steak', 'Sushi',
    'Caesar salad', 'Spaghetti bolognese', 'Chicken wings', 'Ice cream', 'Chocolate cake'
]


def setup_yolo_dirs():
    """YOLO eğitim formatı için gerekli klasörleri oluşturur."""
    for split in ['train', 'val']:
        (PROCESSED_PATH / 'images' / split).mkdir(parents=True, exist_ok=True)
        (PROCESSED_PATH / 'labels' / split).mkdir(parents=True, exist_ok=True)
    print(f"Klasör yapısı {PROCESSED_PATH} altında hazırlandı.")


def load_meta_classes():
    """Veri seti meta verilerini yükler ve filtreler."""
    # Bu fonksiyon, gerçek veri setine bağlı olarak uyarlanmalıdır.
    # Örneğin, bir CSV dosyasından etiketleri okuyabiliriz.
    metadata_path = RAW_DATA_PATH / "meta" / "train.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata.keys()

def resolve_image_path(images_dir: Path, rel_path: str) -> Path | None:
    """rel_path = 'Class_Name/123' -> images_dir/Class_Name/123.jpg veya varyant"""
    base = images_dir / f"{rel_path}.jpg"
    if base.exists():
        return base
    parts = rel_path.split("/", 1)
    if len(parts) != 2:
        return None
    cls, img_id = parts[0], parts[1]
    for folder in [cls, cls.replace(" ", "_"), cls.replace("_", " ")]:
        p = images_dir / folder / f"{img_id}.jpg"
        if p.exists():
            return p
    return None
    


def process_and_split_data(train_ratio=0.8):
    """Veri setini işler ve eğitim ve doğrulama setlerine böler."""
    
    for class_name in TARGET_CLASSES:
        class_dir= RAW_DATA_PATH / class_name
        
        images = list(class_dir.glob("*.jpg"))
        
        random.shuffle(images)
        split_index = int(len(images) * train_ratio)
        train_images = images[:split_index]
        val_images = images[split_index:]

        # Dosyaları kopyalama fonksiyonuna gönder
        copy_to_dest(train_images, "train", class_name)
        copy_to_dest(val_images, "val", class_name)



def copy_to_dest(image_list, split_type, class_name):
        """Görüntüleri ve etiketleri hedef klasörlere kopyalar."""
        for img_path in image_list:
             # Çakışmayı önlemek için yeni isim: "pizza_12345.jpg"
            new_name = f"{class_name}_{img_path.name}"
            
            # Hedef yol: data/processed/yolo_dataset/train/images/pizza_12345.jpg
            destination = PROCESSED_PATH / split_type / "images" / new_name
            
            # Kopyalama işlemi
            shutil.copy(img_path, destination)




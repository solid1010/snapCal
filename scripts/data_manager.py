import os
import shutil
import random
import json
from pathlib import Path
from ultralytics import YOLO

# 1. Sabit Yollar
RAW_DATA_PATH = Path("data/raw/food-101")
PROCESSED_PATH = Path("data/processed/yolo_dataset")

# 2. Odaklanacağımız maksimum sınıf sayısı
MAX_CLASSES = 10



def setup_yolo_dirs():
    """YOLO eğitim formatı için gerekli klasörleri oluşturur."""
    for split in ['train', 'val']:
        (PROCESSED_PATH / 'images' / split).mkdir(parents=True, exist_ok=True)
        (PROCESSED_PATH / 'labels' / split).mkdir(parents=True, exist_ok=True)
    print(f"Klasör yapısı {PROCESSED_PATH} altında hazırlandı.")


def normalize_class_name(class_name: str) -> str:
    """Sınıf adlarını küçük harf, boşluk ve alt çizgiye çevirir."""
    return class_name.lower().replace(" ", "_")


def load_meta_txt(metadata_path: Path,split_type: str) -> dict|None:
    """meta/train.txt veya meta/val.txt dosyasını yükler ve bir sözlük olarak döndürür."""
    metadata_path = metadata_path /"meta" / f"{split_type}.txt"
    
    if not metadata_path.exists():
        return None
    
    meta_dict = {}

    with open(metadata_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split("/", 1)
            if len(parts) != 2:
                continue
            cls, img_id = parts[0], parts[1]
            
            # Az sayıda kategori denediğimiz için filtreleme yapıyoruz.
            if cls not in meta_dict:
                continue
            # Sözlüğe ekle: {'pizza': ['123', '456']} yapısını kurar
            if cls not in meta_dict:
                meta_dict[cls] = []
            meta_dict[cls].append(line)
            
    return meta_dict

def load_meta_json(metadata_path: Path,split_type: str) -> dict|None:
    """meta/train.json veya meta/val.json dosyasını yükler ve bir sözlük olarak döndürür."""
    metadata_path = metadata_path / "meta" / f"{split_type}.json"
    if not metadata_path.exists():
        return None
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata
         

def load_meta(metadata_path: Path,split_type: str) -> dict|None:
    """Önce json dosyasını yükler, eğer yoksa txt dosyasını yükler."""
    data = load_meta_json(metadata_path,split_type) or load_meta_txt(metadata_path,split_type)
    if data is None:
        raise ValueError(f"Meta dosyası {metadata_path / f'{split_type}.json'} veya {metadata_path / f'{split_type}.txt'} bulunamadı.")
    return data
    
def filter_classes(meta_dict: dict,max_classes) -> dict:
    """Sınıf sayısını MAX_CLASSES sayısına kadar azaltır.
    max_classes:None ise tüm sınıfları döndürür , int ilk N sınıfı döndürür , liste ise belirtilen sınıfları döndürür """

    if max_classes is None:
        return meta_dict
    
    if isinstance(max_classes, list):
        norm_target_cls=[normalize_class_name(cls) for cls in max_classes]
        return {k: v for k, v in meta_dict.items() if normalize_class_name(v) in norm_target_cls}
    #int ise ilk N sınıfı döndürür
    classes=sorted(meta.keys())[:max_classes]
    return {k: meta[k] for k in classes}
    

def resolve_image_path(images_dir: Path, rel_path: str) -> Path | None:
    """rel_path = 'Class_Name/123' -> images_dir/Class_Name/123.jpg veya varyant"""
    base = images_dir /"images" / f"{rel_path}.jpg"
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
    """Veri setini işler ve eğitim ve doğrulama setlerine böler.
    Train test olarak ayrılmamış datasetlerde kullanılır."""

    
    for class_name in MAX_CLASSES:
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


def run_pipeline():
    images_dir = RAW_DATA_PATH / "images"
    meta_dir = RAW_DATA_PATH / "meta"
    if not meta_dir.exists():
        raise ValueError(f"Meta dosyası {meta_dir} bulunamadı.")

    # Meta dosyalarını yükle
    train_meta = load_meta(meta_dir, "train")
    test_meta = load_meta(meta_dir, "test")

    # Sadece MAX_CLASSES sayısı kadar sınıf seç
    train_meta = filter_classes(load_meta(meta_dir, "train"), MAX_CLASSES)
    test_meta = filter_classes(load_meta(meta_dir, "test"), MAX_CLASSES)

    class_names = sorted(train_meta.keys())
    class_to_id={c:i for i,c in enumerate(class_names)}

    print(f"Seçilen sınıflar {class_names} ve {len(class_names)} sınıf seçildi.")

    setup_yolo_dirs()
    model = YOLO("yolo11n.pt")

    def process_image(meta:dict,split_type:str):
        count=0
        for cls,rel_paths in meta.items():
            cid=class_to_id[cls]
            for rel_path in rel_paths:
                img_path=resolve_image_path(images_dir,rel_path)
                if img_path is None:
                    print(f"Görüntü {img_path} bulunamadı.")
                    continue
                new_name=rel_path.replace("/","_") + ("" if rel_path.endswith(".jpg") else ".jpg")
                if not new_name.lower().endswith(".jpg"):
                    new_name += ".jpg"

                dest_img=PROCESSED_PATH / split_type / "images" / new_name
                shutil.copy2(img_path,dest_img)

                results= model.predict(str(img_path),conf=0.25,verbose=False)
                lines=[]

                if results and len(results)>0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                    for row in results[0].boxes.xywhn.cpu().numpy():
                        lines.append(f"{cid} {row[0]:.6f} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f}")
                    
                else:
                    lines.append(f"{cid} 0.5 0.5 1.0 1.0")

                lbl_path = PROCESSED_PATH / "labels" / split_type / (Path(new_name).stem + ".txt")
                with open(lbl_path,"w",encoding="utf-8") as f:
                    f.write("\n".join(lines))
                count+=1
        return count

    train_count = process_image(train_meta,"train")
    val_count = process_image(test_meta,"val")

    #data.yaml

    yaml_path = PROCESSED_PATH / "data.yaml"
        
    with open(yaml_path, "w", encoding="utf-8") as f:
        # 1. Veri setinin ana dizini
        f.write(f"path: {PROCESSED_PATH.resolve()}\n")
        # 2. Eğitim resimlerinin yolu (ana dizine göre)
        f.write("train: images/train\n")
        # 3. Doğrulama resimlerinin yolu (ana dizine göre)
        f.write("val: images/val\n")
        # 4. Sınıf sayısı (nc: number of classes)
        f.write(f"nc: {len(class_names)}\n")
        # 5. Sınıf isimleri listesi
        f.write("names:\n")
        for i, name in enumerate(class_names):
            f.write(f"  {i}: {name}\n")

    print(f"Bitti.Train :{train_count}, Val: {val_count}")
    print(f"data.yaml: {yaml_path}")


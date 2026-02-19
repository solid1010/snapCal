import shutil
import random
import json
from pathlib import Path
from ultralytics import YOLO

# 1. Sabit Yollar
RAW_DATA_PATH = Path("data/raw/food-101")
PROCESSED_PATH = Path("data/processed/yolo_dataset")
# Modelleri 'models/' klasöründe topla
MODEL_PATH = Path("models/pretrained")
MODEL_PATH.mkdir(parents=True, exist_ok=True)


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
    metadata_path = metadata_path / f"{split_type}.txt"
    
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
            # Sözlüğe ekle: {'pizza': ['123', '456']} yapısını kurar
            if cls not in meta_dict:
                meta_dict[cls] = []
            meta_dict[cls].append(line)
            
    return meta_dict

def load_meta_json(metadata_path: Path,split_type: str) -> dict|None:
    """meta/train.json veya meta/val.json dosyasını yükler ve bir sözlük olarak döndürür."""
    metadata_path = metadata_path / f"{split_type}.json"
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
        return {k: v for k, v in meta_dict.items() if normalize_class_name(k) in norm_target_cls}
    #int ise ilk N sınıfı döndürür
    classes=sorted(meta_dict.keys())[:max_classes]
    return {k: meta_dict[k] for k in classes}

def generate_labels(img_path: Path | str, model, class_id: int, conf: float = 0.25) -> list[str]:
    """YOLO-seg ile görsel üzerinde poligon (maske) tahmini yapar."""
    # Segmentasyon modeli kullanılarak tahmin yapılır
    results = model.predict(str(img_path), conf=conf, verbose=False)
    lines = []
    
    # Eğer model bir maske (segmentasyon) bulduysa
    if results and len(results) > 0 and results[0].masks is not None:
        # Her bir nesnenin poligon noktalarını al (normalize edilmiş formatta)
        for mask in results[0].masks.xyn:
            # Poligon noktalarını düz bir string haline getir: "class x1 y1 x2 y2..."
            polygon_points = " ".join([f"{coord:.6f}" for pair in mask for coord in pair])
            lines.append(f"{class_id} {polygon_points}")
    else:
        # Maske bulunamazsa (fallback), kutuyu poligon gibi simüle et veya boş bırak
        # Şimdilik maske bulunamayan görselleri eğitime katmamak daha sağlıklı olabilir.
        pass
    
    return lines

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
    
def process_and_split_data(class_names: list[str], images_root: Path, train_ratio=0.8):
    """Veri setini işler ve eğitim ve doğrulama setlerine böler.
    Train test olarak ayrılmamış datasetlerde kullanılır.
    
    Args:
        class_names: İşlenecek sınıf isimleri listesi (örn. ['pizza', 'burger', ...])
        images_root: Sınıf klasörlerinin bulunduğu ana dizin (örn. data/raw/my_dataset/images)
        train_ratio: Eğitim seti oranı (varsayılan 0.8 -> %80 train, %20 val)
    """
    # Klasörleri oluştur
    setup_yolo_dirs()
    
    # YOLO modelini yükle
    model = YOLO(MODEL_PATH / "yolo11n-seg.pt")
    
    # Sınıf -> ID eşlemesi
    class_to_id = {c: i for i, c in enumerate(class_names)}
    
    for class_name in class_names:
        class_dir = images_root / class_name
        
        if not class_dir.exists():
            print(f"Uyarı: {class_dir} bulunamadı, atlanıyor.")
            continue
        
        images = list(class_dir.glob("*.jpg"))
        
        if not images:
            print(f"Uyarı: {class_name} klasöründe hiç görsel yok.")
            continue
        
        random.shuffle(images)
        split_index = int(len(images) * train_ratio)
        train_images = images[:split_index]
        val_images = images[split_index:]

        # Dosyaları kopyalama + etiketleme
        copy_to_dest(train_images, "train", class_name, model, class_to_id)
        copy_to_dest(val_images, "val", class_name, model, class_to_id)
    
    # data.yaml oluştur
    yaml_path = PROCESSED_PATH / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {PROCESSED_PATH.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write("names:\n")
        for i, name in enumerate(class_names):
            f.write(f"  {i}: {name}\n")
    
    print(f"İşlem tamamlandı: {yaml_path}")



def copy_to_dest(image_list, split_type, class_name, model, class_to_id):
    """Görüntüleri ve etiketleri hedef klasörlere kopyalar.
    Meta'sız datasetler için YOLO auto-labeling ile etiket oluşturur."""
    for img_path in image_list:
        # Çakışmayı önlemek için yeni isim: "pizza_12345.jpg"
        new_name = f"{class_name}_{img_path.name}"
        
        # Hedef yol: data/processed/yolo_dataset/images/train/pizza_12345.jpg
        dest_img = PROCESSED_PATH / "images" / split_type / new_name
        shutil.copy(img_path, dest_img)
        
        # YOLO ile bbox tahmini
        cid = class_to_id[class_name]
        lines = generate_labels(dest_img, model, cid)
        
        # Etiket dosyasını yaz
        lbl_path = PROCESSED_PATH / "labels" / split_type / (Path(new_name).stem + ".txt")
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def run_pipeline():
    images_dir = RAW_DATA_PATH / "images"
    meta_dir = RAW_DATA_PATH / "meta"
    if not meta_dir.exists():
        raise ValueError(f"Meta dosyası {meta_dir} bulunamadı.")

    # Meta dosyalarını yükle
    # Sadece MAX_CLASSES sayısı kadar sınıf seç
    train_meta = filter_classes(load_meta(meta_dir, "train"), MAX_CLASSES)
    test_meta = filter_classes(load_meta(meta_dir, "test"), MAX_CLASSES)

    class_names = sorted(train_meta.keys())
    class_to_id={c:i for i,c in enumerate(class_names)}

    print(f"Seçilen sınıflar {class_names} ve {len(class_names)} sınıf seçildi.")

    setup_yolo_dirs()
    model = YOLO(MODEL_PATH / "yolo11n-seg.pt")

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

                dest_img=PROCESSED_PATH / "images"/ split_type / new_name
                shutil.copy2(img_path,dest_img)

                lines = generate_labels(img_path, model, cid)
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

if __name__ == "__main__":
    run_pipeline()


from ultralytics import YOLO
import torch
import os

def train_model():
    # Donanım hızlandırma kontrolü
    if torch.cuda.is_available():
        device = 'cuda'
        print("Cuda destekli GPU bulundu. Eğitim GPU üzerinde gerçekleştirilecek.")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("MPS destekli cihaz bulundu. Eğitim MPS üzerinde gerçekleştirilecek.")
    else:
        device = 'cpu'
        print("GPU veya MPS destekli cihaz bulunamadı. Eğitim CPU üzerinde gerçekleştirilecek.")

    # Model yükleme
    model = YOLO("models/pretrained/yolo11n.pt")

    # Eğitimi başlat
    model.train(data="data/processed/yolo_dataset/data.yaml", 
                epochs=10,
                imgsz=640,
                batch=16,
                device=device,
                conf=0.25,        # Sadece %25+ emin olduğun kutuları NMS'e sok
                max_det=100,      # Resim başına maksimum 100 kutu ara (Food-101 için fazlasıyla yeterli)
                iou=0.45,         # Kutu çakışma eşiği
                name="snapCal_v1",
                project="runs/train",
                optimizer="auto",
                plots=True)
    
    print("Eğitim tamamlandı.")

if __name__ == "__main__":
    train_model()
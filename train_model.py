"""
Script: Training YOLOv8 untuk Deteksi Kematangan Strawberry
Model: YOLOv8n (nano) - ringan dan cepat
Dataset: dataset_bbox (sudah dikonversi ke format bounding box)
"""

from ultralytics import YOLO
import os

def main():
    print("🍓 Training YOLOv8 - Strawberry Ripeness Detection")
    print("=" * 55)
    
    # Load pre-trained YOLOv8n model
    model = YOLO("yolov8n.pt")
    
    # Path ke data.yaml
    data_yaml = os.path.abspath("dataset_bbox/data.yaml")
    print(f"📁 Dataset config: {data_yaml}")
    
    # Training
    results = model.train(
        data=data_yaml,
        epochs=3,          # 3 epoch untuk mempercepat demo (asli 50)
        imgsz=320,           # resolusi kecil untuk CPU
        batch=16,            # batch size
        name="strawberry_ripeness",  # nama experiment
        project="runs/detect",       # folder output
        patience=10,         # early stopping patience
        save=True,           # simpan checkpoint
        save_period=10,      # simpan setiap 10 epoch
        plots=True,          # generate plot training
        verbose=True,
    )
    
    print("\n✅ Training selesai!")
    print(f"📊 Best model: runs/detect/strawberry_ripeness/weights/best.pt")
    
    # Validasi
    print("\n🔍 Menjalankan validasi pada test set...")
    best_model = YOLO("runs/detect/strawberry_ripeness/weights/best.pt")
    val_results = best_model.val(
        data=data_yaml,
        split="test",
    )
    
    print(f"\n📈 Hasil Evaluasi (Test Set):")
    print(f"   mAP50:     {val_results.box.map50:.4f}")
    print(f"   mAP50-95:  {val_results.box.map:.4f}")
    

if __name__ == "__main__":
    main()

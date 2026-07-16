"""
Script: Konversi Label Segmentasi (Polygon) → Bounding Box (Detection)
Mengkonversi format YOLO segmentation ke YOLO detection untuk training YOLOv8 detect.
"""

import os
import shutil
import glob

SRC_DATASET = "Datasets-2"
DST_DATASET = "dataset_bbox"

def polygon_to_bbox(polygon_coords):
    """
    Mengkonversi koordinat polygon YOLO segmentation ke bounding box YOLO detection.
    Input: list of floats [x1, y1, x2, y2, ..., xn, yn] (normalized)
    Output: (x_center, y_center, width, height) (normalized)
    """
    xs = polygon_coords[0::2]  # ambil semua x (index genap)
    ys = polygon_coords[1::2]  # ambil semua y (index ganjil)
    
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    return x_center, y_center, width, height


def convert_label_file(src_path, dst_path):
    """Konversi satu file label dari segmentation ke bbox format."""
    with open(src_path, 'r') as f:
        lines = f.readlines()
    
    bbox_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        class_id = parts[0]
        coords = [float(x) for x in parts[1:]]
        
        if len(coords) < 4:
            continue  # skip invalid lines
        
        x_center, y_center, width, height = polygon_to_bbox(coords)
        
        # Clamp values to [0, 1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        bbox_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    with open(dst_path, 'w') as f:
        f.writelines(bbox_lines)


def process_split(split_name):
    """Process satu split (train/val/test)."""
    src_img_dir = os.path.join(SRC_DATASET, "images", split_name)
    src_lbl_dir = os.path.join(SRC_DATASET, "labels", split_name)
    dst_img_dir = os.path.join(DST_DATASET, "images", split_name)
    dst_lbl_dir = os.path.join(DST_DATASET, "labels", split_name)
    
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)
    
    # Copy images
    img_files = glob.glob(os.path.join(src_img_dir, "*"))
    for img_path in img_files:
        shutil.copy2(img_path, dst_img_dir)
    
    # Convert labels
    lbl_files = glob.glob(os.path.join(src_lbl_dir, "*.txt"))
    converted = 0
    for lbl_path in lbl_files:
        fname = os.path.basename(lbl_path)
        dst_path = os.path.join(dst_lbl_dir, fname)
        convert_label_file(lbl_path, dst_path)
        converted += 1
    
    print(f"  [{split_name}] Images: {len(img_files)}, Labels converted: {converted}")


def create_data_yaml():
    """Buat file data.yaml untuk YOLOv8 detection."""
    yaml_content = f"""path: {os.path.abspath(DST_DATASET)}
train: images/train
val: images/val
test: images/test

nc: 3
names:
  - Fullripe
  - Semiripe
  - Unripe
"""
    yaml_path = os.path.join(DST_DATASET, "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"\n✅ data.yaml created at: {yaml_path}")


def main():
    print("🍓 Konversi Segmentation → Bounding Box")
    print("=" * 50)
    
    # Buat dataset baru
    if os.path.exists(DST_DATASET):
        shutil.rmtree(DST_DATASET)
    
    for split in ["train", "val", "test"]:
        process_split(split)
    
    create_data_yaml()
    
    # Verifikasi
    print("\n📊 Verifikasi sample label (sebelum vs sesudah):")
    src_sample = glob.glob(os.path.join(SRC_DATASET, "labels", "train", "*.txt"))[0]
    dst_sample = os.path.join(DST_DATASET, "labels", "train", os.path.basename(src_sample))
    
    print(f"\n  [SEGMENTATION] {os.path.basename(src_sample)}:")
    with open(src_sample) as f:
        line = f.readline().strip()
        print(f"    {line[:100]}...")
    
    print(f"\n  [BBOX] {os.path.basename(dst_sample)}:")
    with open(dst_sample) as f:
        line = f.readline().strip()
        print(f"    {line}")
    
    print("\n✅ Konversi selesai! Dataset siap untuk training YOLOv8 detect.")


if __name__ == "__main__":
    main()
